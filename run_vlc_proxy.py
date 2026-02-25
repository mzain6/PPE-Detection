
import subprocess
import time
import os
import sys

# Configuration
VLC_PATH = r"D:\VLC\vlc.exe"
RTSP_URL = "rtsp://admin:Test1234@192.168.100.66:554/onvif1"
HTTP_PORT = 8888
HTTP_URL = f"http://localhost:{HTTP_PORT}/stream.mjpg"

def run_proxy():
    if not os.path.exists(VLC_PATH):
        print(f"❌ VLC not found at: {VLC_PATH}")
        return

    # VLC Command to transcode RTSP to HTTP MJPEG
    # -I dummy: No GUI
    # --sout: Stream output
    cmd = [
        VLC_PATH,
        "-I", "dummy",
        "-vvv",
        "--network-caching=300", # Moderate caching to prevent disconnects while keeping latency low
        "--rtsp-tcp", # Keep TCP for stability
        "--clock-jitter=0",
        "--clock-synchro=0",
        RTSP_URL,
        # Transcode to lower res/fps to prevent "stuck" frames
        f'--sout=#transcode{{vcodec=mjpg,scale=0.5,fps=15,acodec=none}}:http{{mux=mpjpeg,dst=:{HTTP_PORT}/stream.mjpg}}',
        ":sout-keep"
    ]

    print(f"🚀 Starting VLC Proxy...")
    print(f"   Source: {RTSP_URL}")
    print(f"   Target: {HTTP_URL}")
    print(f"   Command: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(cmd)
        print(f"\n✅ VLC Proxy Running (PID: {process.pid})")
        print("   Keep this window open!")
        print(f"   You can now view the stream at: {HTTP_URL}")
        
        # Keep prompt alive
        while True:
            time.sleep(1)
            if process.poll() is not None:
                print("❌ VLC exited unexpectedly!")
                break
    except KeyboardInterrupt:
        print("\nStopping Proxy...")
        process.terminate()

if __name__ == "__main__":
    run_proxy()
