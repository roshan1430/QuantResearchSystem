import argparse
import json
import logging

from confluent_kafka import Consumer, Producer

from app.feature_engineering import FeatureEngineeringEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


class StreamProcessor:
    def __init__(self, brokers: str, in_topic: str, out_topic: str) -> None:
        self.consumer = Consumer(
            {
                "bootstrap.servers": brokers,
                "group.id": "feature-engineering-consumer",
                "auto.offset.reset": "latest",
            }
        )
        self.consumer.subscribe([in_topic])
        self.producer = Producer({"bootstrap.servers": brokers})
        self.out_topic = out_topic
        self.engine = FeatureEngineeringEngine()

    def run(self) -> None:
        while True:
            message = self.consumer.poll(1.0)
            if message is None:
                continue
            if message.error():
                logging.error("Kafka consumer error: %s", message.error())
                continue
            payload = json.loads(message.value().decode("utf-8"))
            engineered = self.engine.ingest(payload)
            if engineered is None:
                continue
            self.producer.produce(
                self.out_topic,
                key=payload["sensor_id"],
                value=json.dumps(engineered, default=str),
            )
            self.producer.poll(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brokers", default="kafka:9092")
    parser.add_argument("--in-topic", default="raw_metrics")
    parser.add_argument("--out-topic", default="processed_features")
    args = parser.parse_args()
    StreamProcessor(args.brokers, args.in_topic, args.out_topic).run()


if __name__ == "__main__":
    main()
