"""
Weather Anomaly Investigation MCP Server
Provides tools for database queries and MQTT messaging for weather anomaly detection.
Configuration values are fixed and not exposed to the agent.
"""
import sqlite3
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

# ============================================
# FIXED CONFIGURATION - NOT EXPOSED TO AGENT
# ============================================
DEFAULT_DB_PATH = os.getenv(
    "DB_PATH", 
    "/Users/simonsiagian/Library/CloudStorage/OneDrive-UniversityofAdelaide/Research Project/experiment_code/Anomaly_detectin_ori/weather_anomaly_monitor/data/weather_data.db"
)
DEFAULT_MQTT_BROKER = os.getenv("MQTT_BROKER", "test.mosquitto.org")
DEFAULT_MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
DEFAULT_MQTT_TOPIC = os.getenv("MQTT_TOPIC", "weather/investigation")

# Initialize MCP server
mcp = FastMCP("weather_anomaly_tools")

# Station metadata
STATION_METADATA = {
    'features': {
        'tt': {'name': 'Temperature', 'unit': '°C', 'sensor': 'thermometer', 
               'brand': 'Vaisala HMP155', 'range': '-40 to 60°C'},
        'rh': {'name': 'Relative Humidity', 'unit': '%', 'sensor': 'hygrometer', 
               'brand': 'Vaisala HMP155', 'range': '0-100%'},
        'pp': {'name': 'Atmospheric Pressure', 'unit': 'hPa', 'sensor': 'barometer', 
               'brand': 'Vaisala PTB330', 'range': '850-1100 hPa'},
        'ws': {'name': 'Wind Speed', 'unit': 'm/s', 'sensor': 'anemometer', 
               'brand': 'Young 05103-L', 'range': '0-50 m/s'},
        'wd': {'name': 'Wind Direction', 'unit': '°', 'sensor': 'wind vane', 
               'brand': 'Young 05103-L', 'range': '0-360°'},
        'sr': {'name': 'Solar Radiation', 'unit': 'W/m²', 'sensor': 'pyranometer', 
               'brand': 'Kipp & Zonen CMP3', 'range': '0-1500 W/m²'},
        'rr': {'name': 'Rainfall', 'unit': 'mm', 'sensor': 'rain gauge', 
               'brand': 'Texas Electronics TR-525M', 'range': '0-100 mm/h'}
    },
    'station_info': {
        'id': 'AWS_DKI_UNIV_INDO',
        'latitude': -34.9285,
        'longitude': 138.6007,
        'altitude': 48.0,
        'location': 'Depok, Indodnesia',
        'last_calibration': '2024-03-15',
        'calibration_interval': '6 months',
        'maintenance_contact': 'station-maintenance@weather.org'
    }
}

def ensure_database_exists(db_path: str):
    """Ensure database and sample data exist."""
    try:
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Create table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS weather_data (
                    timestamp TEXT PRIMARY KEY,
                    tt REAL,
                    rh REAL,
                    pp REAL,
                    ws REAL,
                    wd REAL,
                    sr REAL,
                    rr REAL
                )
            """)
            
            # Insert sample data if table is empty
            cursor.execute("SELECT COUNT(*) FROM weather_data")
            if cursor.fetchone()[0] == 0:
                sample_data = [
                    ('2024-09-24T12:00:00', 22.5, 65.0, 1013.2, 5.2, 180.0, 800.0, 0.0),
                    ('2024-09-24T11:00:00', 21.8, 68.0, 1013.5, 4.8, 175.0, 750.0, 0.0),
                    ('2024-09-24T10:00:00', 20.2, 72.0, 1014.0, 4.2, 170.0, 650.0, 0.0),
                    ('2024-09-24T09:00:00', 18.5, 78.0, 1014.2, 3.8, 165.0, 500.0, 0.0),
                    ('2024-09-24T08:00:00', 17.0, 82.0, 1014.5, 3.2, 160.0, 300.0, 0.0),
                ]
                
                cursor.executemany("""
                    INSERT INTO weather_data 
                    (timestamp, tt, rh, pp, ws, wd, sr, rr) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, sample_data)
                
                conn.commit()
                
    except Exception as e:
        print(f"Database initialization error: {e}")


@mcp.tool()
async def get_data_from_db(
    features: str = "tt,rh,pp,ws,wd,sr,rr",
    limit: int = 5
) -> str:
    """
    Get latest weather records from the database.
    
    Parameters:
    - features: Comma-separated list of features to retrieve (e.g., 'tt,rh,pp')
    - limit: Number of latest records to retrieve (default: 5)
    
    Returns:
    - JSON string containing the latest weather records
    
    Available features:
    - tt: Temperature (°C)
    - rh: Relative Humidity (%)
    - pp: Atmospheric Pressure (hPa)
    - ws: Wind Speed (m/s)
    - wd: Wind Direction (°)
    - sr: Solar Radiation (W/m²)
    - rr: Rainfall (mm)
    """
    # Use fixed database path
    db_path = DEFAULT_DB_PATH
    
    try:
        # Ensure database exists
        ensure_database_exists(db_path)
        
        # Clean and validate features
        feature_list = [f.strip() for f in features.split(',')]
        valid_features = ['tt', 'rh', 'pp', 'ws', 'wd', 'sr', 'rr']
        feature_list = [f for f in feature_list if f in valid_features]
        
        if not feature_list:
            return json.dumps({
                "error": "No valid features provided",
                "valid_features": valid_features
            })
        
        # Build SQL query
        columns = ['timestamp'] + feature_list
        query = f"SELECT {', '.join(columns)} FROM weather_data ORDER BY timestamp DESC LIMIT {limit}"
        
        # Execute query
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            
            if not rows:
                return json.dumps({"error": "No data found in database"})
            
            # Format results
            results = []
            for row in rows:
                record = dict(zip(columns, row))
                results.append(record)
            
            # Calculate statistics for each feature
            statistics = {}
            for feature in feature_list:
                values = [r[feature] for r in results if r.get(feature) is not None]
                if values:
                    statistics[feature] = {
                        "mean": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "latest": values[0] if values else None,
                        "feature_name": STATION_METADATA['features'][feature]['name'],
                        "unit": STATION_METADATA['features'][feature]['unit']
                    }
            
            response = {
                "records": results,
                "statistics": statistics,
                "count": len(results)
            }
            
            return json.dumps(response, indent=2)
            
    except sqlite3.Error as e:
        return json.dumps({"error": f"Database error: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"Error retrieving data: {str(e)}"})


# @mcp.tool()
# async def send_anomalous_status(
#     timestamp: str,
#     anomalous_features: str
# ) -> str:
#     """
#     Send anomalous weather status to MQTT broker with station metadata.
    
#     Parameters:
#     - timestamp: Timestamp of the anomaly (ISO format, e.g., "2024-09-25T14:30:00")
#     - anomalous_features: JSON string with format {"feature_code": value}
#       Example: {"tt": 35.2, "rh": 25.0}
    
#     Returns:
#     - Success message with confirmation details
    
#     Feature codes:
#     - tt: Temperature, rh: Humidity, pp: Pressure, ws: Wind Speed, 
#     - wd: Wind Direction, sr: Solar Radiation, rr: Rainfall
#     """
#     # Use fixed MQTT configuration
#     mqtt_broker = DEFAULT_MQTT_BROKER
#     mqtt_port = DEFAULT_MQTT_PORT
#     mqtt_topic = DEFAULT_MQTT_TOPIC
    
#     try:
#         # Parse anomalous features
#         if isinstance(anomalous_features, str):
#             features_data = json.loads(anomalous_features)
#         else:
#             features_data = anomalous_features
            
#         # Build anomalous features list with metadata
#         processed_features = []
#         for feature_code, current_value in features_data.items():
#             feature_info = STATION_METADATA['features'].get(feature_code, {})
            
#             processed_feature = {
#                 "feature": feature_info.get('name', feature_code.upper()),
#                 "current_value": current_value,
#                 #"unit": feature_info.get('unit', 'N/A'),
#                 "sensor_brand": feature_info.get('brand', f'Unknown sensor for {feature_code}'),
#                 #"sensor_range": feature_info.get('range', 'N/A')
#             }
#             processed_features.append(processed_feature)
        
#         # Build complete message with station metadata
#         message = {
#             "timestamp": timestamp,
#             "station_metadata": {
#                 "station_id": STATION_METADATA['station_info']['id'],
#                 "location": STATION_METADATA['station_info']['location'],
#                 "latitude": STATION_METADATA['station_info']['latitude'],
#                 "longitude": STATION_METADATA['station_info']['longitude'],
#                 "altitude": STATION_METADATA['station_info']['altitude'],
#                 "last_calibration": STATION_METADATA['station_info']['last_calibration'],
#                 "maintenance_contact": STATION_METADATA['station_info']['maintenance_contact']
#             },
#             "anomalous_features": processed_features
#         }
        
#         message_json = json.dumps(message, indent=2)
        
#         # Try to send via MQTT
#         try:
#             import paho.mqtt.client as mqtt
            
#             client = mqtt.Client()
            
#             # Connection callback
#             connection_result = {"success": False}
#             def on_connect(client, userdata, flags, rc):
#                 if rc == 0:
#                     connection_result["success"] = True
#                 else:
#                     connection_result["error"] = f"Connection failed with code {rc}"
            
#             client.on_connect = on_connect
            
#             # Connect and publish
#             client.connect(mqtt_broker, mqtt_port, 60)
#             client.loop_start()
            
#             # Wait for connection
#             import time
#             timeout = 5
#             start_time = time.time()
#             while not connection_result["success"] and time.time() - start_time < timeout:
#                 if "error" in connection_result:
#                     raise Exception(connection_result["error"])
#                 time.sleep(0.1)
            
#             if not connection_result["success"]:
#                 raise Exception("Connection timeout")
            
#             # Publish message
#             result = client.publish(mqtt_topic, message_json, qos=1)
#             result.wait_for_publish(timeout=10)
            
#             client.loop_stop()
#             client.disconnect()
            
#             if result.rc != mqtt.MQTT_ERR_SUCCESS:
#                 raise Exception(f"Publish failed with return code: {result.rc}")
            
#             return json.dumps({
#                 "status": "success",
#                 "message": "Anomaly report sent to MQTT successfully",
#                 "broker": mqtt_broker,
#                 "topic": mqtt_topic,
#                 "timestamp": timestamp,
#                 "anomalous_features_count": len(processed_features),
#                 "reported_features": list(features_data.keys())
#             }, indent=2)
            
#         except ImportError:
#             # Fallback: save to file if MQTT library not available
#             fallback_file = f"mqtt_fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
#             with open(fallback_file, "w") as f:
#                 json.dump({
#                     "timestamp": datetime.now().isoformat(),
#                     "message": message,
#                     "topic": mqtt_topic,
#                     "broker": mqtt_broker
#                 }, f, indent=2)
            
#             return json.dumps({
#                 "status": "fallback",
#                 "message": f"MQTT library not available. Message saved to {fallback_file}",
#                 "file": fallback_file
#             }, indent=2)
            
#     except json.JSONDecodeError as e:
#         return json.dumps({
#             "status": "error",
#             "error": f"Invalid JSON format for anomalous_features: {str(e)}"
#         })
#     except Exception as e:
#         # Fallback: save to file on any error
#         fallback_file = f"mqtt_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
#         try:
#             with open(fallback_file, "w") as f:
#                 json.dump({
#                     "timestamp": datetime.now().isoformat(),
#                     "message": message if 'message' in locals() else anomalous_features,
#                     "topic": mqtt_topic,
#                     "broker": mqtt_broker,
#                     "error": str(e)
#                 }, f, indent=2)
            
#             return json.dumps({
#                 "status": "error",
#                 "error": str(e),
#                 "fallback_file": fallback_file,
#                 "message": f"MQTT failed, saved to {fallback_file}"
#             })
#         except:
#             return json.dumps({
#                 "status": "error",
#                 "error": f"Failed to send MQTT and save fallback: {str(e)}"
#             })

@mcp.tool()
async def publish_anomaly_to_kafka(
    timestamp: str,
    anomalous_features: str,
    trend_analysis: str = None
) -> str:
    """
    Publish confirmed anomaly to Kafka event broker with station metadata and trend analysis.
    
    Parameters:
    - timestamp: Timestamp of the anomaly (ISO format, e.g., "2024-09-25T14:30:00")
    - anomalous_features: JSON string with format {"feature_code": value}
      Example: {"tt": 35.2, "rh": 25.0}
    - trend_analysis: Optional JSON string with trend insights for each anomalous feature
      Example: {"tt": "Rising 3.5°C over last 5 hours, exceeding normal range",
                "rh": "Dropped 15% in 2 hours, unusual for this time period"}
    
    Returns:
    - Success message with confirmation details
    
    Feature codes:
    - tt: Temperature, rh: Humidity, pp: Pressure, ws: Wind Speed, 
    - wd: Wind Direction, sr: Solar Radiation, rr: Rainfall
    """
    # Fixed Kafka configuration (not exposed to agent)
    kafka_broker = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "10.33.205.40:9093")
    kafka_topic = os.getenv("KAFKA_TOPIC", "weather-anomalies")
    
    try:
        # Parse anomalous features
        if isinstance(anomalous_features, str):
            features_data = json.loads(anomalous_features)
        else:
            features_data = anomalous_features
        
        # Parse trend analysis if provided
        trend_data = {}
        if trend_analysis:
            if isinstance(trend_analysis, str):
                trend_data = json.loads(trend_analysis)
            else:
                trend_data = trend_analysis
            
        # Build anomalous features list with metadata and trends
        confirmed_anomalies = []
        for feature_code, current_value in features_data.items():
            feature_info = STATION_METADATA['features'].get(feature_code, {})
            
            confirmed_anomaly = {
                "parameter": feature_info.get('name', feature_code.upper()),
                "parameter_code": feature_code,
                "value": current_value,
                "unit": feature_info.get('unit', 'N/A'),
                "sensor_brand": feature_info.get('brand', f'Unknown sensor for {feature_code}'),
                "sensor_range": feature_info.get('range', 'N/A')
            }
            
            # Add trend analysis if available
            if feature_code in trend_data:
                confirmed_anomaly["trend_analysis"] = trend_data[feature_code]
            
            confirmed_anomalies.append(confirmed_anomaly)
        
        # Build Kafka event payload
        event_data = {
            "timestamp": timestamp,
            "station_id": STATION_METADATA['station_info']['id'],
            "station_metadata": {
                "location": STATION_METADATA['station_info']['location'],
                "latitude": STATION_METADATA['station_info']['latitude'],
                "longitude": STATION_METADATA['station_info']['longitude'],
                "altitude": STATION_METADATA['station_info']['altitude'],
                "sensor_info": {
                    feature_code: {
                        "brand": STATION_METADATA['features'][feature_code]['brand'],
                        "last_calibration": STATION_METADATA['station_info']['last_calibration']
                    }
                    for feature_code in features_data.keys()
                }
            },
            "confirmed_anomalies": confirmed_anomalies,
            "validation_timestamp": datetime.now().isoformat(),
            "has_trend_analysis": bool(trend_data)
        }
        
        # Create Kafka event envelope
        kafka_event = {
            "event_type": "WEATHER_ANOMALY_CONFIRMED",
            "timestamp": datetime.now().isoformat(),
            "source": f"StationAgent-{STATION_METADATA['station_info']['id']}",
            "data": event_data,
            "correlation_id": f"anomaly-{STATION_METADATA['station_info']['id']}-{int(datetime.now().timestamp())}"
        }
        
        # Try to publish to Kafka
        try:
            from kafka import KafkaProducer
            
            producer = KafkaProducer(
                bootstrap_servers=kafka_broker,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None
            )
            
            # Publish with station_id as key for partitioning
            future = producer.send(
                kafka_topic,
                key=STATION_METADATA['station_info']['id'],
                value=kafka_event
            )
            
            # Wait for confirmation (with timeout)
            record_metadata = future.get(timeout=10)
            
            producer.flush()
            producer.close()
            
            return json.dumps({
                "status": "success",
                "message": "Anomaly published to Kafka successfully",
                "kafka_broker": kafka_broker,
                "topic": kafka_topic,
                "partition": record_metadata.partition,
                "offset": record_metadata.offset,
                "timestamp": timestamp,
                "station_id": STATION_METADATA['station_info']['id'],
                "anomalous_features_count": len(confirmed_anomalies),
                "trend_analysis_included": bool(trend_data),
                "correlation_id": kafka_event["correlation_id"]
            }, indent=2)
            
        except ImportError:
            # Fallback: save to file if Kafka library not available
            fallback_file = f"kafka_fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(fallback_file, "w") as f:
                json.dump(kafka_event, f, indent=2)
            
            return json.dumps({
                "status": "fallback",
                "message": f"Kafka library not available. Event saved to {fallback_file}",
                "file": fallback_file,
                "install_command": "pip install kafka-python"
            }, indent=2)
            
    except json.JSONDecodeError as e:
        return json.dumps({
            "status": "error",
            "error": f"Invalid JSON format: {str(e)}"
        })
    except Exception as e:
        # Fallback: save to file on any error
        fallback_file = f"kafka_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            error_event = kafka_event if 'kafka_event' in locals() else {
                "error": "Failed to create event",
                "anomalous_features": anomalous_features
            }
            
            with open(fallback_file, "w") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "event": error_event,
                    "error": str(e)
                }, f, indent=2)
            
            return json.dumps({
                "status": "error",
                "error": str(e),
                "fallback_file": fallback_file,
                "message": f"Kafka publish failed, saved to {fallback_file}"
            })
        except:
            return json.dumps({
                "status": "error",
                "error": f"Failed to publish to Kafka and save fallback: {str(e)}"
            })

# Run the MCP server
if __name__ == "__main__":
    mcp.run(transport='stdio')