import cv2

CREDS = "admin:ADMIN123"
IP    = "192.168.100.157"

urls = [
    # Dahua main stream formats
    f"rtsp://{CREDS}@{IP}:554/cam/realmonitor?channel=1&subtype=0",
    f"rtsp://{CREDS}@{IP}:554/cam/realmonitor?channel=1&subtype=1",
    # Hikvision formats
    f"rtsp://{CREDS}@{IP}:554/Streaming/Channels/101",
    f"rtsp://{CREDS}@{IP}:554/Streaming/Channels/1",
    # Generic
    f"rtsp://{CREDS}@{IP}:554/h264/ch1/main/av_stream",
    f"rtsp://{CREDS}@{IP}:554/onvif1",
    f"rtsp://{CREDS}@{IP}/onvif/device_service",
    # Channel 5 variations
    f"rtsp://{CREDS}@{IP}:554/cam/realmonitor?channel=5&subtype=0",
    f"rtsp://{CREDS}@{IP}:554/Streaming/Channels/501",
]

print("Testing RTSP URLs...\n")
for url in urls:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 4000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 4000)
    ok = cap.isOpened()
    ret, _ = cap.read() if ok else (False, None)
    status = "✅ OK + READ" if (ok and ret) else ("⚠ OPEN NO READ" if ok else "❌ FAIL")
    print(f"[{status}] {url}")
    cap.release()

print("\nDone.")
