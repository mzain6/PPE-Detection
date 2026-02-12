
import socket
import threading
from queue import Queue

# Common local subnets
SUBNETS = [
    "192.168.100.",  # Prioritized based on ipconfig
    "192.168.1.",
    "192.168.0."
]

port = 554 # RTSP Port
print_lock = threading.Lock()

def portscan(ip):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5) 
    try:
        con = s.connect((ip, port))
        with print_lock:
            print(f"[FOUND] RTSP Service Open: {ip}")
        con.close()
        return True
    except:
        pass
    return False

def worker():
    while True:
        ip = q.get()
        portscan(ip)
        q.task_done()

q = Queue()

print("Scanning local network for Dahua DVRs (Port 554)...")
print("This fits generic networks. If your IP is unique (e.g., 10.0.0.x), edit the script.")

# Start Threads
for x in range(50):
    t = threading.Thread(target=worker)
    t.daemon = True
    t.start()

# Add jobs
for subnet in SUBNETS:
    for i in range(1, 255):
        q.put(subnet + str(i))

q.join()
print("Scan Complete.")
print("If you see an IP above, put it in 'run_cameras.py' as RTSP_IP.")
print("If nothing found, check your PC's IP address and update the SUBNETS list.")









