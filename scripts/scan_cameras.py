
import socket
import threading
from queue import Queue
import time

# Configuration
SUBNET = "192.168.100."
PORT = 554  # RTSP
THREAD_COUNT = 50

def scan_ip(ip):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.5)
    result = sock.connect_ex((ip, PORT))
    sock.close()
    if result == 0:
        return True
    return False

def threader():
    while True:
        worker = q.get()
        if scan_ip(worker):
            print(f"[FOUND] Camera found at {worker}")
            found_cameras.append(worker)
        q.task_done()

q = Queue()
found_cameras = []

print(f"Scanning {SUBNET}1-254 for RTSP cameras on port {PORT}...")

for x in range(THREAD_COUNT):
    t = threading.Thread(target=threader)
    t.daemon = True
    t.start()

for worker in range(1, 255):
    q.put(f"{SUBNET}{worker}")

q.join()

print("\n--- Scan Complete ---")
if found_cameras:
    print(f"Found {len(found_cameras)} cameras:")
    for cam in found_cameras:
        print(f" - {cam}")
else:
    print("No cameras found.")
