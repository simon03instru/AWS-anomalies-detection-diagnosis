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
        """Handle incoming MQTT messages"""
        try:
            message_str = msg.payload.decode('utf-8')
            data = json.loads(message_str)
            timestamp = self._extract_timestamp(data)
            self.message_callback(data, timestamp)
        except Exception as e:
            self.logger.error(f"Error processing MQTT message: {e}")
            
    def _extract_timestamp(self, data: dict) -> datetime:
        """Extract timestamp from message or use current time"""
        if 'timestamp' in data:
            try:
                timestamp_str = data['timestamp']
                formats = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S.%f']
                for fmt in formats:
                    try:
                        return datetime.strptime(timestamp_str, fmt)
                    except ValueError:
                        continue
            except Exception:
                pass
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