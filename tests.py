import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import numpy as np
import time

BASE_URL = "http://127.0.0.1:5000"
NUM_THREADS = 5
TOTAL_REQUESTS = 100

def random_predict_payload():
    return {"features": np.random.uniform(0, 10, size=4).tolist()}

def call_endpoint(i):
    endpoint_choice = random.choice(["health", "info", "predict"])
    url = f"{BASE_URL}/{endpoint_choice}"
    try:
        if endpoint_choice == "predict":
            resp = requests.post(url, json=random_predict_payload(), timeout=10)
        else:
            resp = requests.get(url, timeout=10)
        try:
            data = resp.json()
        except Exception:
            data = resp.text
        return f"{i} -> {endpoint_choice} | {resp.status_code} | {data}"
    except Exception as e:
        return f"{i} -> {endpoint_choice} | ERROR | {str(e)}"

start_time = time.time()

with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    futures = [executor.submit(call_endpoint, i) for i in range(TOTAL_REQUESTS)]
    for future in as_completed(futures):
        print(future.result())

end_time = time.time()
print(f"\nAll requests completed in {end_time - start_time:.2f} seconds")
