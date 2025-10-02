#!/usr/bin/env python3
"""
Weather Anomaly Event Monitor
Simple real-time monitor for Kafka events from station agents.
No event consumption - just monitoring for debugging/visibility.
"""

import json
import argparse
from datetime import datetime
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import os
from dotenv import load_dotenv

load_dotenv()

# ANSI color codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


class EventMonitor:
    """Monitor Kafka events without consuming them"""
    
    def __init__(self, 
                 bootstrap_servers: str = "localhost:9092",
                 topic: str = "weather-anomalies",
                 group_id: str = None):
        """
        Initialize monitor.
        
        Args:
            bootstrap_servers: Kafka broker address
            topic: Topic to monitor
            group_id: Consumer group (None = new unique group each time)
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id or f"monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.consumer = None
        self.event_count = 0
        self.station_counts = {}
    
    def connect(self):
        """Connect to Kafka"""
        try:
            print(f"{CYAN}Connecting to Kafka...{RESET}")
            print(f"  Broker: {self.bootstrap_servers}")
            print(f"  Topic: {self.topic}")
            print(f"  Group: {self.group_id}\n")
            
            self.consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset='latest',  # Only new messages
                enable_auto_commit=False,  # Don't commit - just monitor
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None
            )
            
            print(f"{GREEN}✓ Connected successfully{RESET}\n")
            return True
            
        except KafkaError as e:
            print(f"{RED}✗ Kafka connection failed: {e}{RESET}")
            return False
        except Exception as e:
            print(f"{RED}✗ Error: {e}{RESET}")
            return False
    
    def start_monitoring(self):
        """Start monitoring events"""
        if not self.consumer:
            if not self.connect():
                return
        
        print("=" * 80)
        print(f"{BOLD}{CYAN}WEATHER ANOMALY EVENT MONITOR{RESET}")
        print("=" * 80)
        print(f"Monitoring topic: {YELLOW}{self.topic}{RESET}")
        print(f"Waiting for events... (Ctrl+C to stop)\n")
        
        try:
            for message in self.consumer:
                self._display_event(message)
                
        except KeyboardInterrupt:
            print(f"\n\n{CYAN}Monitoring stopped{RESET}")
            self._print_summary()
        except Exception as e:
            print(f"\n{RED}Error during monitoring: {e}{RESET}")
        finally:
            if self.consumer:
                self.consumer.close()
    
    def _display_event(self, message):
        """Display a single event"""
        self.event_count += 1
        
        event = message.value
        key = message.key
        partition = message.partition
        offset = message.offset
        timestamp = datetime.fromtimestamp(message.timestamp / 1000.0)
        
        # Update station counts
        station_id = event.get('data', {}).get('station_id', 'unknown')
        self.station_counts[station_id] = self.station_counts.get(station_id, 0) + 1
        
        # Display header
        print("=" * 80)
        print(f"{BOLD}{GREEN}EVENT #{self.event_count}{RESET} | "
            f"{CYAN}Partition: {partition}{RESET} | "
            f"{CYAN}Offset: {offset}{RESET}")
        print(f"{DIM}Received: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
        print("-" * 80)
        
        # Event metadata
        event_type = event.get('event_type', 'UNKNOWN')
        source = event.get('source', 'unknown')
        correlation_id = event.get('correlation_id', 'N/A')
        
        print(f"{YELLOW}Type:{RESET} {event_type}")
        print(f"{YELLOW}Source:{RESET} {source}")
        print(f"{YELLOW}Correlation ID:{RESET} {correlation_id}")
        print(f"{YELLOW}Partition Key:{RESET} {key}")
        
        # Event data
        data = event.get('data', {})
        
        if data:
            print(f"\n{MAGENTA}Station Information:{RESET}")
            print(f"  ID: {data.get('station_id', 'N/A')}")
            
            metadata = data.get('station_metadata', {})
            if metadata:
                print(f"  Location: {metadata.get('location', 'N/A')}")
                print(f"  Coordinates: {metadata.get('latitude', 'N/A')}, "
                    f"{metadata.get('longitude', 'N/A')}")
                print(f"  Altitude: {metadata.get('altitude', 'N/A')}m")
            
            # Anomalous features with sensor info
            anomalies = data.get('confirmed_anomalies', [])
            if anomalies:
                print(f"\n{RED}Confirmed Anomalies ({len(anomalies)}):{RESET}")
                for i, anomaly in enumerate(anomalies, 1):
                    param = anomaly.get('parameter', 'Unknown')
                    param_code = anomaly.get('parameter_code', '')
                    value = anomaly.get('value', 'N/A')
                    sensor = anomaly.get('sensor_brand', 'Unknown')
                    
                    print(f"  {i}. {param}: {value}")
                    print(f"     Sensor: {sensor}")
                    
                    # Get calibration info from sensor_info
                    sensor_info = metadata.get('sensor_info', {})
                    if param_code in sensor_info:
                        last_cal = sensor_info[param_code].get('last_calibration', 'N/A')
                        print(f"     Last Calibration: {last_cal}")
        
        print("=" * 80)
        print()
    
    def _print_summary(self):
        """Print monitoring summary"""
        print("\n" + "=" * 80)
        print(f"{BOLD}{CYAN}MONITORING SUMMARY{RESET}")
        print("=" * 80)
        print(f"Total events monitored: {self.event_count}")
        
        if self.station_counts:
            print(f"\nEvents per station:")
            for station_id, count in sorted(self.station_counts.items()):
                print(f"  {station_id}: {count}")
        
        print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Monitor weather anomaly events from Kafka"
    )
    parser.add_argument(
        "--broker", "-b",
        default="10.33.205.40:9093",  # Default to external listener
        help="Kafka bootstrap servers"
    )
    parser.add_argument(
        "--topic", "-t",
        default="weather-anomalies",
        help="Kafka topic to monitor"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Show all messages from beginning"
    )
    
    args = parser.parse_args()
    
    monitor = EventMonitor(
        bootstrap_servers=args.broker,
        topic=args.topic
    )
    
    if args.all:
        monitor.consumer = KafkaConsumer(
            args.topic,
            bootstrap_servers=args.broker,
            group_id=monitor.group_id,
            auto_offset_reset='earliest',
            enable_auto_commit=False,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            key_deserializer=lambda k: k.decode('utf-8') if k else None
        )
    
    monitor.start_monitoring()

if __name__ == "__main__":
    main()