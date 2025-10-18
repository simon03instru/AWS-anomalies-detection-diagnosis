import paho.mqtt.client as mqtt
import json
import csv
from datetime import datetime
import time

# MQTT broker info
BROKER = "10.33.205.40"
PORT = 1883
TOPIC = "station/aws_diy_stageof_yogyakarta"

# CSV file path
CSV_FILE = "test_dataset.csv"  # Change this to your CSV file path

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

def parse_and_publish(csv_file):
    """Read CSV file and publish each row to MQTT"""
    
    # Create MQTT client
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_publish = on_publish
    client.on_disconnect = on_disconnect
    
    try:
        # Connect to broker
        client.connect(BROKER, PORT, 60)
        client.loop_start()  # Start network loop in background
        
        # Read and publish CSV data
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            
            for row_num, row in enumerate(reader, start=2):  # Start from 2 (after header)
                try:
                    # Parse values from CSV row
                    timestamp_str = row['date']
                    tt = float(row['tt'])
                    rh = float(row['rh'])
                    pp = float(row['pp'])
                    ws = float(row['ws'])
                    wd = float(row['wd'])
                    sr = float(row['sr'])
                    rr = float(row['rr'])
                    
                    # Convert timestamp to standard format
                    timestamp = datetime.fromisoformat(timestamp_str)
                    
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
                    
                    # Publish to MQTT
                    payload = json.dumps(data)
                    result = client.publish(TOPIC, payload)
                    
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
        
        print("All rows processed successfully")
        
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    parse_and_publish(CSV_FILE)