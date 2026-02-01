import requests
import time
import sys

BASE_URL = "http://127.0.0.1:8000"

def test_stability(iterations=100):
    print(f"Starting stability test on {BASE_URL} for {iterations} iterations...")
    
    # 1. Health check
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=2)
        if r.status_code != 200:
            print("❌ Health check failed")
            return
        print("✅ Health check passed")
    except Exception as e:
        print(f"❌ Could not connect to server: {e}")
        return

    # 2. Loop detect
    success = 0
    errors = 0
    start_time = time.time()
    
    for i in range(iterations):
        try:
            # We use source='0' (default) which might fail if no camera, 
            # but we check if it returns 503/504 vs crash (500)
            r = requests.get(f"{BASE_URL}/detect?source=0", timeout=5)
            if r.status_code == 200:
                success += 1
                sys.stdout.write(".")
            elif r.status_code in [503, 504]:
                # Expected if no camera connected, counting as "handled error"
                sys.stdout.write("w") 
                success += 1
            else:
                errors += 1
                sys.stdout.write("E")
        except Exception:
            errors += 1
            sys.stdout.write("X")
        sys.stdout.flush()
        time.sleep(0.1)

    duration = time.time() - start_time
    print(f"\n\nTest Finished in {duration:.2f}s")
    print(f"Success/Handled: {success}")
    print(f"Errors/Crashes: {errors}")
    
    if errors == 0:
        print("✅ Stability Test Passed")
    else:
        print("❌ Stability Test Failed")

if __name__ == "__main__":
    test_stability()
