"""
Test RTSP camera connection with multiple URL formats.
"""
import cv2
import time

# Camera credentials
IP = "192.168.100.157"
USERNAME = "admin"
PASSWORD = "Admin123"
PORT = "554"

# Common Dahua/DMSS RTSP URL formats
rtsp_urls = [
    # Format 1: Standard Dahua main stream
    f"rtsp://{USERNAME}:{PASSWORD}@{IP}:{PORT}/cam/realmonitor?channel=1&subtype=0",
    
    # Format 2: Dahua sub stream (lower quality, faster)
    f"rtsp://{USERNAME}:{PASSWORD}@{IP}:{PORT}/cam/realmonitor?channel=1&subtype=1",
    
    # Format 3: Alternative Dahua format
    f"rtsp://{USERNAME}:{PASSWORD}@{IP}:{PORT}/live",
    
    # Format 4: H.264 stream
    f"rtsp://{USERNAME}:{PASSWORD}@{IP}:{PORT}/h264",
    
    # Format 5: Streaming channels format
    f"rtsp://{USERNAME}:{PASSWORD}@{IP}:{PORT}/Streaming/Channels/101",
    
    # Format 6: Alternative with channel parameter
    f"rtsp://{USERNAME}:{PASSWORD}@{IP}:{PORT}/user={USERNAME}&password={PASSWORD}&channel=1&stream=0.sdp",
]

def test_rtsp_url(url, format_name):
    """Test a single RTSP URL."""
    print(f"\n{'='*70}")
    print(f"Testing {format_name}:")
    print(f"URL: {url.replace(PASSWORD, '****')}")
    print(f"{'='*70}")
    
    try:
        # Try to open the stream
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("❌ Failed to open stream")
            cap.release()
            return False
        
        print("⏳ Attempting to read frame (timeout: 10 seconds)...")
        start_time = time.time()
        
        # Try to read a frame with timeout
        ret, frame = cap.read()
        elapsed = time.time() - start_time
        
        if ret and frame is not None:
            h, w = frame.shape[:2]
            print(f"✅ SUCCESS! Connected to camera")
            print(f"   Resolution: {w}x{h}")
            print(f"   Frame read time: {elapsed:.2f}s")
            
            # Save test frame
            filename = f"test_frame_{format_name.replace(' ', '_').lower()}.jpg"
            cv2.imwrite(filename, frame)
            print(f"   Saved test frame: {filename}")
            
            cap.release()
            return True
        else:
            print(f"❌ Could not read frame (took {elapsed:.2f}s)")
            cap.release()
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("\n")
    print("="*70)
    print("  RTSP Camera Connection Test")
    print("="*70)
    print(f"Camera IP: {IP}")
    print(f"Username: {USERNAME}")
    print(f"Password: {'*' * len(PASSWORD)}")
    print(f"Testing {len(rtsp_urls)} different URL formats...")
    
    successful_urls = []
    
    for idx, url in enumerate(rtsp_urls, 1):
        format_name = f"Format {idx}"
        success = test_rtsp_url(url, format_name)
        
        if success:
            successful_urls.append((format_name, url))
    
    # Summary
    print("\n")
    print("="*70)
    print("  RESULTS SUMMARY")
    print("="*70)
    
    if successful_urls:
        print(f"\n✅ Found {len(successful_urls)} working URL(s):\n")
        for format_name, url in successful_urls:
            print(f"{format_name}:")
            print(f"  {url.replace(PASSWORD, '****')}\n")
        
        print("\n📝 Recommended URL for your multi_camera_ppe_demo.py:")
        print("-" * 70)
        best_url = successful_urls[0][1]
        print(f"""
cameras_config = [
    {{
        'id': 'dahua_camera_1',
        'source': '{best_url}'
    }},
]
""")
    else:
        print("\n❌ No working URLs found!")
        print("\nTroubleshooting steps:")
        print("1. Verify the camera IP is correct: ping {IP}")
        print("2. Check if port 554 is open on the camera")
        print("3. Verify username/password in DMSS app")
        print("4. Ensure you're on the same network as the camera")
        print("5. Try accessing the camera's web interface: http://{IP}")
        print("6. Check if the camera requires different credentials for RTSP")
    
    print("="*70)
    print("\n")

if __name__ == "__main__":
    main()
