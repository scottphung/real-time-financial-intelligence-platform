from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'financial-stream',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print("Listening for messages...")

for message in consumer:
    print(message.value)