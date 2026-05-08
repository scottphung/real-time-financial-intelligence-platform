import json
import time
import requests

from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

URL = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"

import random

sleep_time = 5

while True:
    response = requests.get(URL)

    if response.status_code == 200:
        data = response.json()

        message = {
            "source": "coingecko",
            "symbol": "BTCUSD",
            "price": data["bitcoin"]["usd"],
            "timestamp": time.time()
        }

        producer.send("financial-stream", value=message)

        print("Sent:", message)

        sleep_time = 5  # reset on success

    else:
        print("Skipping due to rate limit")

        sleep_time = min(sleep_time * 2, 60)  # exponential backoff (important)

    time.sleep(sleep_time)