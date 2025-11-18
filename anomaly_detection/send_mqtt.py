#!/usr/bin/env python3
"""
MQTT Weather Data Simulator
Simulates sending weather station data to MQTT broker

Usage:
    python mqtt_simulator.py --data "2024-08-19 07:40:00+00,30.92,49.15,994.5651,1.511,172.2,530.6,0.0"
    python mqtt_simulator.py --data "2024-08-19 07:40:00+00,30.92,49.15,994.5651,1.511,172.2,530.6,0.0" --broker 10.33.205.40 --topic station/test
    python mqtt_simulator.py --file data.csv --broker 10.33.205.40
"""

import argparse
import json
import time
from datetime import datetime
import paho.mqtt.client as mqtt


def parse_raw_data(raw_line):
    """Parse raw CSV line into structured data"""
    parts = raw_line.strip().split(",")
    
    if len(parts) != 8:
        raise ValueError(f"Expected 8 fields, got {len(parts)}")
    
    timestamp_str, tt, rh, pp, ws, wd, sr, rr = parts
    
    # Convert timestamp - handle both '+00' and '+00:00' formats
    # Replace '+00' with '+00:00' for proper ISO format
    if timestamp_str.endswith('+00') and not timestamp_str.endswith('+00:00'):
        timestamp_str = timestamp_str[:-3] + '+00:00'
    
    timestamp = datetime.fromisoformat(timestamp_str)
    
    # Format data
    data = {
        "date": timestamp.strftime("%Y-%m-%d %H:%M:%S+00"),
        "tt": round(float(tt), 2),
        "rh": round(float(rh), 1), 
        "pp": round(float(pp), 4),
        "ws": round(float(ws), 3),
        "wd": round(float(wd), 1),
        "sr": round(float(sr), 1),
        "rr": round(float(rr), 1)
    }
    
    return data


def publish_data(broker, port, topic, data):
    """Publish data to MQTT broker"""
    try:
        client = mqtt.Client()
        client.connect(broker, port, 60)
        
        payload = json.dumps(data)
        result = client.publish(topic, payload)
        
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"✓ Published to {topic}:")
            print(f"  {payload}")
        else:
            print(f"✗ Failed to publish (code: {result.rc})")
        
        client.disconnect()
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Simulate MQTT weather data transmission",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Send single data point
  python mqtt_simulator.py --data "2024-08-19 07:40:00+00,30.92,49.15,994.5651,1.511,172.2,530.6,0.0"
  
  # Send from file
  python mqtt_simulator.py --file data.csv
  
  # Custom broker and topic
  python mqtt_simulator.py --data "..." --broker 192.168.1.100 --topic station/custom
  
  # Send with interval (for file)
  python mqtt_simulator.py --file data.csv --interval 2
        """
    )
    
    # Data source (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--data",
        type=str,
        help="Single raw data line (CSV format: timestamp,tt,rh,pp,ws,wd,sr,rr)"
    )
    data_group.add_argument(
        "--file",
        type=str,
        help="CSV file containing multiple data lines"
    )
    
    # MQTT configuration
    parser.add_argument(
        "--broker",
        type=str,
        default="10.33.205.40",
        help="MQTT broker address (default: 10.33.205.40)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=1883,
        help="MQTT broker port (default: 1883)"
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="station/aws_diy_stageof_yogyakarta",
        help="MQTT topic (default: station/aws_diy_stageof_yogyakarta)"
    )
    
    # File-specific options
    parser.add_argument(
        "--interval",
        type=float,
        default=0,
        help="Interval in seconds between messages when using --file (default: 0)"
    )
    parser.add_argument(
        "--skip-header",
        action="store_true",
        help="Skip first line of CSV file (header)"
    )
    
    args = parser.parse_args()
    
    print(f"MQTT Simulator")
    print(f"Broker: {args.broker}:{args.port}")
    print(f"Topic: {args.topic}")
    print("-" * 50)
    
    # Single data point
    if args.data:
        try:
            data = parse_raw_data(args.data)
            publish_data(args.broker, args.port, args.topic, data)
        except Exception as e:
            print(f"✗ Error parsing data: {e}")
            return 1
    
    # Multiple data points from file
    elif args.file:
        try:
            with open(args.file, 'r') as f:
                lines = f.readlines()
            
            if args.skip_header:
                lines = lines[1:]
            
            print(f"Found {len(lines)} data lines\n")
            
            success_count = 0
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = parse_raw_data(line)
                    if publish_data(args.broker, args.port, args.topic, data):
                        success_count += 1
                    
                    if args.interval > 0 and i < len(lines):
                        time.sleep(args.interval)
                        
                except Exception as e:
                    print(f"✗ Error on line {i}: {e}")
            
            print(f"\n{'='*50}")
            print(f"Summary: {success_count}/{len(lines)} messages published")
            
        except FileNotFoundError:
            print(f"✗ File not found: {args.file}")
            return 1
        except Exception as e:
            print(f"✗ Error reading file: {e}")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())