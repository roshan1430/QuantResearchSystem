import argparse
import json
import random
import time
from datetime import datetime, timezone

import numpy as np
from confluent_kafka import Producer


class ScientificSignalProducer:
    def __init__(self, brokers: str, topic: str) -> None:
        self.producer = Producer({"bootstrap.servers": brokers})
        self.topic = topic
        self.step = 0
        self.base_temp = 4.2
        self.base_pressure = 1.0
        self.base_vibration = 0.05
        self.drift = 0.0

    def next_point(self, anomaly_probability: float, enable_drift: bool) -> dict:
        self.step += 1
        if enable_drift:
            self.drift += 0.0008

        temp = self.base_temp + self.drift + np.sin(self.step / 25.0) * 0.08 + np.random.normal(0, 0.04)
        pressure = self.base_pressure + self.drift * 0.25 + np.random.normal(0, 0.015)
        vibration = self.base_vibration + np.cos(self.step / 17.0) * 0.01 + np.random.normal(0, 0.004)

        is_true_anomaly = random.random() < anomaly_probability
        anomaly_type = "none"
        if is_true_anomaly:
            anomaly_type = random.choice(["spike", "oscillation", "thermal_runup"])
            if anomaly_type == "spike":
                temp += 1.5
                vibration += 0.2
            elif anomaly_type == "oscillation":
                vibration += 0.35
            else:
                temp += 0.9
                pressure += 0.15

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sensor_id": "cryo_pump_A1",
            "temperature_k": round(float(temp), 5),
            "pressure_atm": round(float(pressure), 5),
            "vibration_mms": round(float(vibration), 5),
            "is_true_anomaly": int(is_true_anomaly),
            "anomaly_type": anomaly_type,
        }

    def run(self, rate_hz: int, anomaly_probability: float, enable_drift: bool) -> None:
        sleep_interval = 1.0 / rate_hz
        while True:
            payload = self.next_point(anomaly_probability=anomaly_probability, enable_drift=enable_drift)
            self.producer.produce(self.topic, key=payload["sensor_id"], value=json.dumps(payload))
            self.producer.poll(0)
            time.sleep(sleep_interval)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brokers", default="kafka:9092")
    parser.add_argument("--topic", default="raw_metrics")
    parser.add_argument("--rate", default=4, type=int)
    parser.add_argument("--anomaly-prob", default=0.03, type=float)
    parser.add_argument("--drift", action="store_true")
    args = parser.parse_args()
    ScientificSignalProducer(args.brokers, args.topic).run(args.rate, args.anomaly_prob, args.drift)


if __name__ == "__main__":
    main()
