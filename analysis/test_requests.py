import requests
import random
import time

URL = "http://127.0.0.1:5001/predict"

def random_request():
    return {
        "carat": round(random.uniform(0.2, 2.5), 2),
        "cut": random.choice(["Ideal", "Premium", "Good", "Very Good", "Fair"]),
        "color": random.choice(["D", "E", "F", "G", "H", "I", "J"]),
        "clarity": random.choice(["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2"]),
        "depth": round(random.uniform(55, 65), 2),
        "table": round(random.uniform(50, 70), 2),
        "x": round(random.uniform(3, 9), 2),
        "y": round(random.uniform(3, 9), 2),
        "z": round(random.uniform(2, 6), 2)
    }

for i in range(100):
    r = requests.post(URL, json=random_request())
    if r.status_code != 200:
        print("Ошибка:", r.text)
    time.sleep(0.1)

print("✅ 100 запросов отправлены")

