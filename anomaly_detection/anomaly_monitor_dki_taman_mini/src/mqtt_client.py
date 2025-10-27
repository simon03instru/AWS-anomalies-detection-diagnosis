import json
import logging
from datetime import datetime
from typing import Callable
import paho.mqtt.client as mqtt

class MQTTClient:
    """Handles MQTT connection and message processing"""
    
    def __init__(self, broker: str, port: int, topic: str, client_id: str, 
                 message_callback: Callable, qos: int = 1):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.qos = qos
        self.message_callback = message_callback
        
        self.client = mqtt.Client(client_id=client_id, clean_session=True)
        self.connected = False
        self.logger = logging.getLogger(__name__)
        
        self._setup_callbacks()
        
    def _setup_callbacks(self):
        """Setup MQTT client callbacks"""
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        
    def _on_connect(self, client, userdata, flags, rc):
        """Handle MQTT connection"""
        if rc == 0:
            self.connected = True
            result = client.subscribe(self.topic, qos=self.qos)
            if result[0] == mqtt.MQTT_ERR_SUCCESS:
                self.logger.info(f"Connected and subscribed to {self.topic}")
            else:
                self.logger.error(f"Failed to subscribe to {self.topic}")
        else:
            self.connected = False
            self.logger.error(f"MQTT connection failed with code: {rc}")
            
    def _on_disconnect(self, client, userdata, rc):
        """Handle MQTT disconnection"""
        self.connected = False
        if rc != 0:
            self.logger.warning("Unexpected MQTT disconnection")
            
    def _on_message(self, client, userdata, msg):
        """Handle incoming MQTT messages - supports CSV format"""
        try:
            message_str = msg.payload.decode('utf-8').strip()
            
            # Try JSON first
            try:
                data = json.loads(message_str)
            except json.JSONDecodeError:
                # If JSON fails, parse as CSV
                # Format: timestamp,tt,rh,pp,ws,wd,sr,rr
                parts = message_str.split(',')
                if len(parts) >= 8:
                    try:
                        data = {
                            'date': parts[0].strip(),
                            'tt': float(parts[1]),
                            'rh': float(parts[2]),
                            'pp': float(parts[3]),
                            'ws': float(parts[4]),
                            'wd': float(parts[5]),
                            'sr': float(parts[6]),
                            'rr': float(parts[7])
                        }
                    except (ValueError, IndexError) as e:
                        self.logger.error(f"Failed to parse CSV: {message_str}, Error: {e}")
                        return
                else:
                    self.logger.error(f"Invalid message format (not enough fields): {message_str}")
                    return
            
            timestamp = self._extract_timestamp(data)
            self.message_callback(data, timestamp)
            
        except Exception as e:
            self.logger.error(f"Error processing MQTT message: {e}")
            
    def _extract_timestamp(self, data: dict) -> datetime:
        """Extract timestamp from message or use current time"""
        # Try different timestamp field names (CSV uses 'date', JSON uses 'timestamp')
        timestamp_str = None
        for field in ['date', 'timestamp', 'time', 'dt']:
            if field in data:
                timestamp_str = data[field]
                break
        
        if timestamp_str:
            try:
                timestamp_str = str(timestamp_str).strip()
                
                # Remove timezone if present
                for tz_marker in ['+00', '+00:00', 'Z', '+0000']:
                    if tz_marker in timestamp_str:
                        timestamp_str = timestamp_str.split(tz_marker)[0]
                        break
                
                # Try different formats
                formats = [
                    '%Y-%m-%dT%H:%M:%S.%f',      # 2024-12-31T23:50:00.123456
                    '%Y-%m-%dT%H:%M:%S',         # 2024-12-31T23:50:00
                    '%Y-%m-%d %H:%M:%S.%f',      # 2024-12-31 23:50:00.123456
                    '%Y-%m-%d %H:%M:%S',         # 2024-12-31 23:50:00
                ]
                
                for fmt in formats:
                    try:
                        return datetime.strptime(timestamp_str, fmt)
                    except ValueError:
                        continue
                
                self.logger.warning(f"Could not parse timestamp: {timestamp_str}, using current time")
                
            except Exception as e:
                self.logger.warning(f"Error extracting timestamp: {e}")
        
        # Fallback to current time
        return datetime.now()
        
    def connect(self) -> bool:
        """Connect to MQTT broker"""
        try:
            result = self.client.connect(self.broker, self.port, keepalive=60)
            if result == mqtt.MQTT_ERR_SUCCESS:
                self.client.loop_start()
                return True
            return False
        except Exception as e:
            self.logger.error(f"MQTT connection error: {e}")
            return False
            
    def disconnect(self):
        """Disconnect from MQTT broker"""
        if self.connected:
            self.client.loop_stop()
            self.client.disconnect()