import json
from kafka import KafkaConsumer
from processing.storage.parquet_writer import write_to_parquet
from processing.ml.signal_generator import generate_signal

from collections import deque
import numpy as np

# Kafka consumer
consumer = KafkaConsumer(
    "financial-stream",
    bootstrap_servers="localhost:9092",
    auto_offset_reset="latest",
    value_deserializer=lambda x: json.loads(x.decode("utf-8"))
)

print("🚀 Stream Processor Started... Listening to Kafka")

# sliding window
price_window = deque(maxlen=10)


def transform_event(event):
    try:
        price = float(event.get("price"))
        price_window.append(price)

        features = {
            "source": event.get("source"),
            "symbol": event.get("symbol"),
            "price": price,
            "timestamp": event.get("timestamp"),
        }

        # return
        if len(price_window) > 1:
            features["return"] = (price_window[-1] - price_window[-2]) / price_window[-2]
        else:
            features["return"] = 0.0

        # moving average
        features["moving_avg_10"] = float(np.mean(price_window))

        # volatility
        if len(price_window) > 2:
            features["volatility"] = float(np.std(price_window))
        else:
            features["volatility"] = 0.0

        # anomaly
        features["is_anomaly"] = abs(features["return"]) > 0.002

        return features

    except Exception as e:
        print("Bad event:", event, "Error:", e)
        return None


for message in consumer:
    raw_event = message.value

    processed = transform_event(raw_event)

    if processed:
        print("✅ Clean Event:", processed)

        # 💾 store
        write_to_parquet(processed)

        # 📡 ML SIGNAL (REAL-TIME)
        signal = generate_signal(processed)

        print("📡 SIGNAL:", signal)