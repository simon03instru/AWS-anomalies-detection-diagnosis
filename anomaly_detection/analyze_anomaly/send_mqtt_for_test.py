import paho.mqtt.client as mqtt
import json
import csv
from datetime import datetime
import time
import argparse
import sys

# MQTT broker info
BROKER = "10.33.205.40"
PORT = 1883
TOPIC = "station/aws_diy_pakem"

def on_connect(client, userdata, flags, rc):
    """Callback for when the client connects to the broker"""
    if rc == 0:
        print("Connected to MQTT broker successfully")
    else:
        print(f"Failed to connect, return code {rc}")

def on_publish(client, userdata, mid):
    """Callback for when a message is published"""
    print(f"Message published (ID: {mid})")

def on_disconnect(client, userdata, rc):
    """Callback for when the client disconnects"""
    if rc != 0:
        print(f"Unexpected disconnection (code {rc})")
    else:
        print("Disconnected from MQTT broker")

def parse_timestamp(timestamp_str):
    """
    Parse timestamp in multiple formats:
    - Old format: 2025-09-23T15:56:00.688089
    - New format: 2024-12-25 01:20:00+00 or 2024-12-28 10:10:00+00
    """
    timestamp_str = timestamp_str.strip()
    
    # Try multiple parsing strategies using strptime
    formats = [
        "%Y-%m-%d %H:%M:%S%z",     # 2024-12-25 01:20:00+0000
        "%Y-%m-%dT%H:%M:%S.%f",    # 2025-09-23T15:56:00.688089
        "%Y-%m-%d %H:%M:%S",       # 2024-12-25 01:20:00 (no timezone)
    ]
    
    # First, normalize the timezone format from +00 to +0000
    normalized_ts = timestamp_str
    if '+' in normalized_ts:
        # Split on the + to handle timezone
        parts = normalized_ts.rsplit('+', 1)
        if len(parts) == 2 and len(parts[1]) == 2:
            # Convert +00 to +0000
            normalized_ts = parts[0] + '+' + parts[1] + '00'
    
    for fmt in formats:
        try:
            dt = datetime.strptime(normalized_ts, fmt)
            return dt
        except ValueError:
            continue
    
    raise ValueError(f"Could not parse timestamp '{timestamp_str}' (normalized: '{normalized_ts}')")

def detect_format(csv_file):
    """
    Detect CSV format by checking the first row
    Returns: 'new' or 'old'
    """
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        first_row = next(reader, None)
        
        if first_row is None:
            raise ValueError("CSV file is empty")
        
        # Check if 'is_anomaly' column exists
        if 'is_anomaly' in first_row:
            return 'new'
        else:
            return 'old'

def parse_and_publish(csv_file, broker=BROKER, port=PORT, topic=TOPIC):
    """Read CSV file and publish each row to MQTT"""
    
    # Detect format
    print(f"Detecting CSV format...")
    try:
        csv_format = detect_format(csv_file)
        print(f"✓ Detected format: {csv_format}")
    except Exception as e:
        print(f"Error detecting format: {e}")
        return
    
    # Create MQTT client
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_publish = on_publish
    client.on_disconnect = on_disconnect
    
    try:
        # Connect to broker
        print(f"Connecting to MQTT broker at {broker}:{port}...")
        client.connect(broker, port, 60)
        client.loop_start()  # Start network loop in background
        
        # Give connection time to establish
        time.sleep(1)
        
        # Read and publish CSV data
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            
            for row_num, row in enumerate(reader, start=2):  # Start from 2 (after header)
                try:
                    # Parse values from CSV row
                    timestamp_str = row['date']
                    
                    # Handle missing/null values
                    def safe_float(value, default=0.0):
                        """Convert value to float, return default if missing or invalid"""
                        if value is None or value == '' or value == 'nan' or str(value).lower() == 'nan':
                            return default
                        try:
                            return float(value)
                        except (ValueError, TypeError):
                            return default
                    
                    tt = safe_float(row.get('tt'))
                    rh = safe_float(row.get('rh'))
                    pp = safe_float(row.get('pp'))
                    ws = safe_float(row.get('ws'))
                    wd = safe_float(row.get('wd'))
                    sr = safe_float(row.get('sr'))
                    rr = safe_float(row.get('rr'))
                    
                    # Parse timestamp (handles both formats)
                    timestamp = parse_timestamp(timestamp_str)
                    
                    # Format data
                    data = {
                        "date": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "tt": round(tt, 2),
                        "rh": round(rh, 1),
                        "pp": round(pp, 4),
                        "ws": round(ws, 3),
                        "wd": round(wd, 1),
                        "sr": round(sr, 1),
                        "rr": round(rr, 1)
                    }
                    
                    # Add is_anomaly if it exists (new format)
                    if 'is_anomaly' in row and row['is_anomaly']:
                        try:
                            data['is_anomaly'] = int(float(row['is_anomaly']))
                        except (ValueError, TypeError):
                            data['is_anomaly'] = 0
                    
                    # Publish to MQTT
                    payload = json.dumps(data)
                    result = client.publish(topic, payload)
                    
                    if result.rc == mqtt.MQTT_ERR_SUCCESS:
                        print(f"Row {row_num}: Published - {payload}")
                    else:
                        print(f"Row {row_num}: Failed to publish (error code: {result.rc})")
                    
                    # Small delay between messages to avoid overwhelming the broker
                    time.sleep(0.5)
                    
                except ValueError as e:
                    print(f"Row {row_num}: Error parsing data - {e}")
                except Exception as e:
                    print(f"Row {row_num}: Unexpected error - {e}")
        
        print("\n✓ All rows processed successfully")
        
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found")
        sys.exit(1)
    except ConnectionRefusedError:
        print(f"Error: Could not connect to MQTT broker at {broker}:{port}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        # Clean up
        time.sleep(1)
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Publish weather data from CSV to MQTT broker',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mqtt_publisher.py data.csv
  python mqtt_publisher.py data.csv --broker 192.168.1.100
  python mqtt_publisher.py data.csv --broker 10.33.205.40 --port 1883 --topic station/weather
        """
    )
    
    parser.add_argument(
        'csv_file',
        help='Path to CSV file with weather data'
    )
    parser.add_argument(
        '--broker',
        default=BROKER,
        help=f'MQTT broker address (default: {BROKER})'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=PORT,
        help=f'MQTT broker port (default: {PORT})'
    )
    parser.add_argument(
        '--topic',
        default=TOPIC,
        help=f'MQTT topic to publish to (default: {TOPIC})'
    )
    
    args = parser.parse_args()
    
    parse_and_publish(args.csv_file, args.broker, args.port, args.topic)