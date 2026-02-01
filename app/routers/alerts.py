"""
PPE Alert API Router

Receives and logs PPE violation alerts.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["alerts"])


class PPEAlert(BaseModel):
    """PPE violation alert model"""
    track_id: int
    timestamp: str
    camera_id: str
    violation_type: str  # NO_HELMET, NO_VEST, NO_BOTH
    screenshot_path: str = None  # Path to violation screenshot


# In-memory storage for demo purposes
# In production, store in database
alert_history: List[PPEAlert] = []


@router.post("/ppe-alert")
async def receive_ppe_alert(alert: PPEAlert):
    """
    Receive PPE violation alert.
    
    Args:
        alert: PPE violation alert data
        
    Returns:
        Success confirmation
    """
    logger.warning(f"🚨 PPE VIOLATION ALERT - Track {alert.track_id} - {alert.violation_type} - Camera: {alert.camera_id} - Time: {alert.timestamp}")
    
    # Store alert
    alert_history.append(alert)
    
    # Keep only last 1000 alerts
    if len(alert_history) > 1000:
        alert_history.pop(0)
    
    return {
        "status": "success",
        "message": f"Alert received for track {alert.track_id}",
        "alert": alert.dict()
    }


@router.get("/ppe-alerts")
async def get_alerts(limit: int = 100):
    """
    Get recent PPE violation alerts.
    
    Args:
        limit: Maximum number of alerts to return
        
    Returns:
        List of recent alerts
    """
    return {
        "total": len(alert_history),
        "alerts": alert_history[-limit:]
    }


@router.get("/ppe-alerts/stats")
async def get_alert_stats():
    """
    Get PPE violation statistics.
    
    Returns:
        Alert statistics by violation type
    """
    stats = {
        "total_alerts": len(alert_history),
        "NO_HELMET": sum(1 for a in alert_history if a.violation_type == "NO_HELMET"),
        "NO_VEST": sum(1 for a in alert_history if a.violation_type == "NO_VEST"),
        "NO_BOTH": sum(1 for a in alert_history if a.violation_type == "NO_BOTH"),
    }
    
    return stats


@router.delete("/ppe-alerts/clear")
async def clear_alerts():
    """
    Clear all PPE violation alerts.
    
    Returns:
        Confirmation message
    """
    global alert_history
    count = len(alert_history)
    alert_history.clear()
    logger.info(f"Cleared {count} alerts from history")
    
    return {
        "status": "success",
        "message": f"Cleared {count} alerts",
        "remaining": 0
    }
