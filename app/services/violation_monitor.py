"""
PPE Violation Monitor Service

Tracks continuous PPE violations and triggers alerts after 10 seconds.
Saves screenshots of violations to alerts folder.
"""
from dataclasses import dataclass
from typing import Dict, Optional
import time
import requests
from datetime import datetime
import logging
import cv2
import os
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ViolationState:
    """Tracks violation state for a single track_id"""
    track_id: int
    violation_start_time: Optional[float] = None
    violation_type: Optional[str] = None
    alert_sent: bool = False
    last_seen: float = 0.0


class ViolationMonitor:
    """
    Monitors PPE violations and sends alerts after continuous violations.
    
    Alert is triggered when:
    - Person is missing helmet/vest for >= 10 seconds continuously
    - Only one alert per continuous violation
    - New alerts allowed after person becomes compliant then violates again
    
    Reset conditions:
    - Person becomes compliant (puts on PPE)
    - Person disappears from frame
    - Track ID changes
    """
    
    def __init__(self, alert_url: str = "http://localhost:8000/api/ppe-alert", 
                 violation_threshold: float = 10.0,
                 enabled: bool = True,
                 save_screenshots: bool = True,
                 screenshots_dir: str = "alerts"):
        """
        Initialize violation monitor.
        
        Args:
            alert_url: POST endpoint for violation alerts
            violation_threshold: Seconds of continuous violation before alert (default: 10.0)
            enabled: Enable/disable violation monitoring
            save_screenshots: Enable/disable screenshot capture on violations
            screenshots_dir: Directory to save violation screenshots
        """
        self.alert_url = alert_url
        self.violation_threshold = violation_threshold
        self.enabled = enabled
        self.save_screenshots = save_screenshots
        self.screenshots_dir = Path(screenshots_dir)
        self.states: Dict[int, ViolationState] = {}
        
        # Create screenshots directory if it doesn't exist
        if self.save_screenshots:
            self.screenshots_dir.mkdir(exist_ok=True)
            logger.info(f"Screenshots will be saved to: {self.screenshots_dir.absolute()}")
        
    def update(self, track_id: int, has_helmet: bool, has_vest: bool, timestamp: float) -> None:
        """
        Update violation state for a track.
        
        Args:
            track_id: Unique person identifier
            has_helmet: True if person is wearing helmet
            has_vest: True if person is wearing vest
            timestamp: Current timestamp (time.time())
        """
        if not self.enabled:
            return
            
        # Determine current violation status
        is_compliant = has_helmet and has_vest
        
        # Determine violation type
        if not has_helmet and not has_vest:
            current_violation = "NO_BOTH"
        elif not has_helmet:
            current_violation = "NO_HELMET"
        elif not has_vest:
            current_violation = "NO_VEST"
        else:
            current_violation = None
        
        # Get or create state for this track
        if track_id not in self.states:
            self.states[track_id] = ViolationState(track_id=track_id, last_seen=timestamp)
        
        state = self.states[track_id]
        state.last_seen = timestamp
        
        # Handle state transitions
        if is_compliant:
            # Person is compliant - reset violation tracking
            if state.violation_start_time is not None:
                logger.info(f"Track {track_id} became compliant. Resetting violation state.")
            state.violation_start_time = None
            state.violation_type = None
            state.alert_sent = False
            
        else:
            # Person is in violation
            if state.violation_start_time is None:
                # New violation started
                state.violation_start_time = timestamp
                state.violation_type = current_violation
                state.alert_sent = False
                logger.info(f"Track {track_id} violation started: {current_violation}")
            else:
                # Ongoing violation - check if violation type changed
                if state.violation_type != current_violation:
                    # Violation type changed - restart timer
                    logger.info(f"Track {track_id} violation changed from {state.violation_type} to {current_violation}. Restarting timer.")
                    state.violation_start_time = timestamp
                    state.violation_type = current_violation
                    state.alert_sent = False
    
    def check_violations(self, camera_id: str = "default", frame=None) -> None:
        """
        Check all tracks for violations exceeding threshold and send alerts.
        
        Args:
            camera_id: Identifier for the camera/stream
            frame: Current video frame (numpy array) for screenshot capture
        """
        if not self.enabled:
            return
            
        current_time = time.time()
        
        for track_id, state in list(self.states.items()):
            # Check if track is still active (seen recently)
            if current_time - state.last_seen > 2.0:
                # Track disappeared - clean up
                logger.info(f"Track {track_id} disappeared. Removing from monitoring.")
                del self.states[track_id]
                continue
            
            # Check if violation meets threshold
            if state.violation_start_time is not None and not state.alert_sent:
                violation_duration = current_time - state.violation_start_time
                
                if violation_duration >= self.violation_threshold:
                    # Save screenshot if enabled and frame available
                    screenshot_path = None
                    if self.save_screenshots and frame is not None:
                        screenshot_path = self._save_screenshot(
                            frame, 
                            track_id, 
                            state.violation_type, 
                            camera_id
                        )
                    
                    # Trigger alert
                    self._send_alert(
                        track_id=track_id,
                        camera_id=camera_id,
                        violation_type=state.violation_type,
                        timestamp=datetime.utcnow().isoformat() + "Z",
                        screenshot_path=screenshot_path
                    )
                    state.alert_sent = True
                    logger.warning(f"Alert sent for track {track_id}: {state.violation_type} for {violation_duration:.1f}s")
    
    def _save_screenshot(self, frame, track_id: int, violation_type: str, camera_id: str) -> Optional[str]:
        """
        Save screenshot of violation to alerts folder.
        
        Args:
            frame: Video frame (numpy array)
            track_id: Track identifier
            violation_type: Type of violation
            camera_id: Camera identifier
            
        Returns:
            Relative path to saved screenshot or None if failed
        """
        try:
            # Create date-based subdirectory
            date_str = datetime.now().strftime("%Y-%m-%d")
            day_dir = self.screenshots_dir / date_str
            day_dir.mkdir(exist_ok=True)
            
            # Generate filename with timestamp
            timestamp_str = datetime.now().strftime("%H-%M-%S")
            filename = f"{camera_id}_track{track_id}_{violation_type}_{timestamp_str}.jpg"
            filepath = day_dir / filename
            
            # Save image
            cv2.imwrite(str(filepath), frame)
            
            # Return relative path
            relative_path = str(filepath.relative_to(self.screenshots_dir.parent))
            logger.info(f"Screenshot saved: {relative_path}")
            return relative_path
            
        except Exception as e:
            logger.error(f"Failed to save screenshot: {e}")
            return None
    
    def _send_alert(self, track_id: int, camera_id: str, violation_type: str, timestamp: str, screenshot_path: Optional[str] = None) -> None:
        """
        Send POST alert to configured endpoint.
        
        Args:
            track_id: Unique person identifier
            camera_id: Camera/stream identifier
            violation_type: Type of violation (NO_HELMET, NO_VEST, NO_BOTH)
            timestamp: ISO format timestamp
            screenshot_path: Path to saved screenshot (optional)
        """
        payload = {
            "track_id": track_id,
            "timestamp": timestamp,
            "camera_id": camera_id,
            "violation_type": violation_type,
            "screenshot_path": screenshot_path
        }
        
        try:
            response = requests.post(self.alert_url, json=payload, timeout=5)
            if response.status_code == 200:
                logger.info(f"Alert sent successfully: {payload}")
            else:
                logger.error(f"Alert failed with status {response.status_code}: {payload}")
        except requests.RequestException as e:
            logger.error(f"Failed to send alert: {e}. Payload: {payload}")
    
    def reset_track(self, track_id: int) -> None:
        """
        Reset violation state for a specific track.
        
        Args:
            track_id: Track to reset
        """
        if track_id in self.states:
            logger.info(f"Resetting track {track_id}")
            del self.states[track_id]
    
    def get_violation_status(self, track_id: int) -> Optional[Dict]:
        """
        Get current violation status for a track.
        
        Args:
            track_id: Track to query
            
        Returns:
            Dictionary with violation info or None if not in violation
        """
        if track_id not in self.states:
            return None
            
        state = self.states[track_id]
        if state.violation_start_time is None:
            return None
            
        return {
            "track_id": track_id,
            "violation_type": state.violation_type,
            "duration": time.time() - state.violation_start_time,
            "alert_sent": state.alert_sent
        }
