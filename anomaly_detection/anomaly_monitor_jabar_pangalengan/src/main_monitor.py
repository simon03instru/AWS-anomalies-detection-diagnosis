import logging
import queue
import threading
import time
import json
import asyncio
import csv
from datetime import datetime
from typing import Dict, Callable, Any, Optional
from pathlib import Path

from mqtt_client import MQTTClient
from data_processor import PersistentDataProcessor
from data_storage import WeatherDataStorage
from anomaly_detector import AnomalyDetectionEngine
from output_handler import OutputHandler


class AnomalyScoreRecorder:
    """Records all anomaly scores to a CSV file"""
    
    def __init__(self, score_file: str = "logs/anomaly_scores.csv"):
        self.score_file = score_file
        self.lock = threading.Lock()
        
        # Create logs directory if it doesn't exist
        Path(score_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV file with headers if it doesn't exist
        if not Path(score_file).exists():
            with open(score_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'anomaly_score', 'threshold', 'is_anomaly'])
    
    def record_score(self, timestamp: datetime, anomaly_score: float, threshold: float, is_anomaly: bool):
        """Record a single anomaly score"""
        with self.lock:
            try:
                with open(self.score_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        timestamp.isoformat(),
                        f"{anomaly_score:.8f}",
                        f"{threshold:.8f}",
                        is_anomaly
                    ])
            except Exception as e:
                logging.error(f"Error recording score: {e}")


class WeatherAnomalyMonitor:
    """Updated monitor with anomaly score recording"""
    
    def __init__(self, config, anomaly_agent=None, score_file: str = "logs/anomaly_scores.csv"):
        self.config = config
        self.running = False
        self.data_queue = queue.Queue()
        
        # Agent for processing anomalies
        self.anomaly_agent = anomaly_agent
        
        # Score recorder
        self.score_recorder = AnomalyScoreRecorder(score_file)
        self.score_file = score_file
        
        # Counters for visibility
        self.readings_processed = 0
        self.anomalies_detected = 0
        self.last_reading_time = None
        
        # Initialize components
        self.mqtt_client = None
        self.data_processor = None
        self.anomaly_engine = None
        self.output_handler = None
        self.scaler = None
        self.data_storage = None
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, config.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self._initialize_components()
        
    def set_agent(self, agent):
        """Set or update the anomaly processing agent"""
        self.anomaly_agent = agent
        self.logger.info(f"Agent set: {type(agent).__name__}")
        
    def _initialize_components(self):
        """Initialize all system components with persistent storage"""
        from mqtt_client import MQTTClient
        from anomaly_detector import AnomalyDetectionEngine
        from output_handler import OutputHandler
        from scaler_manager import WeatherDataScaler
        
        # Initialize data storage
        storage_path = getattr(self.config, 'STORAGE_DB_PATH', 'weather_data.db')
        csv_path = getattr(self.config, 'STORAGE_CSV_PATH', 'weather_data.csv')
        self.data_storage = WeatherDataStorage(storage_path, csv_path)
        
        # Initialize scaler if enabled
        if hasattr(self.config, 'USE_ROBUST_SCALER') and self.config.USE_ROBUST_SCALER:
            try:
                self.scaler = WeatherDataScaler(
                    dataset_path=self.config.DATASET_PATH,
                    feature_names=self.config.WEATHER_FEATURES,
                    scaler_save_path=getattr(self.config, 'SCALER_SAVE_PATH', None)
                )
                self.logger.info("RobustScaler initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize scaler: {e}")
                self.scaler = None
        
        # Initialize persistent data processor
        self.data_processor = PersistentDataProcessor(
            feature_names=self.config.WEATHER_FEATURES,
            win_size=self.config.WIN_SIZE,
            storage=self.data_storage,
            scaler=self.scaler
        )
        
        # Initialize anomaly detection engine
        self.anomaly_engine = AnomalyDetectionEngine(
            model_config=self.config.MODEL_CONFIG,
            checkpoint_path=self.config.MODEL_CHECKPOINT_PATH,
            feature_names=self.config.WEATHER_FEATURES,
            win_size=self.config.WIN_SIZE,
            use_fixed_threshold=getattr(self.config, 'USE_FIXED_THRESHOLD', True),
            fixed_threshold=getattr(self.config, 'FIXED_THRESHOLD', 0.1),
            threshold_percentile=getattr(self.config, 'ANOMALY_THRESHOLD_PERCENTILE', 95.0),
            temperature=getattr(self.config, 'TEMPERATURE_SCALE', 50)
        )
        
        # Initialize output handler - ensure JSON format
        self.output_handler = OutputHandler(
            output_format='json'
        )
        
        # Initialize MQTT client
        self.mqtt_client = MQTTClient(
            broker=self.config.MQTT_BROKER,
            port=self.config.MQTT_PORT,
            topic=self.config.MQTT_TOPIC,
            client_id=self.config.MQTT_CLIENT_ID,
            message_callback=self._on_mqtt_message,
            qos=getattr(self.config, 'MQTT_QOS', 1)
        )
        
        # Print storage status to stderr
        status = self.data_processor.get_storage_status()
        import sys
        print(f"Data Storage Status:", file=sys.stderr)
        print(f"  Database: {status['database_path']}", file=sys.stderr)
        print(f"  CSV: {status['csv_path']}", file=sys.stderr)
        print(f"  Total records: {status['total_records']}", file=sys.stderr)
        print(f"  Ready for detection: {status['has_enough_data']}", file=sys.stderr)
        print(f"  Agent configured: {self.anomaly_agent is not None}", file=sys.stderr)
        print(f"  Score recording to: {self.score_file}", file=sys.stderr)

    def _call_agent_sync(self, anomaly_json: str):
        """Call agent synchronously"""
        try:
            if hasattr(self.anomaly_agent, 'analyze'):
                # Parse JSON back to dict for agent
                anomaly_data = json.loads(anomaly_json)
                result = self.anomaly_agent.analyze_sync(anomaly_data)
                self.logger.info(f"Agent processed anomaly: {result}")
                # Print to stderr for visibility
                import sys
                print(f"[AGENT RESULT] {result[:200]}...", file=sys.stderr, flush=True)
            elif callable(self.anomaly_agent):
                # Agent is a simple function
                result = self.anomaly_agent(anomaly_json)
                self.logger.info(f"Agent function processed anomaly: {result}")
        except Exception as e:
            self.logger.error(f"Error calling agent: {e}")
            import sys
            print(f"AGENT ERROR: {e}", file=sys.stderr)

    def update_threshold(self, new_threshold: float):
        """Update the anomaly detection threshold during runtime"""
        if self.anomaly_engine:
            self.anomaly_engine.update_threshold(new_threshold)
            
    def _on_mqtt_message(self, data: Dict, timestamp: datetime):
        """Handle incoming MQTT messages"""
        self.data_queue.put((data, timestamp))
        self.last_reading_time = timestamp
        
    def _process_data(self):
        """Main data processing loop - RECORDS ALL SCORES"""
        while self.running:
            try:
                # Get data from queue with timeout
                reading, timestamp = self.data_queue.get(timeout=1.0)
                
                # Add reading to buffer (with scaling if enabled)
                has_enough_data = self.data_processor.add_reading(reading, timestamp)
                
                if has_enough_data:
                    # Get current data window (already scaled)
                    data_window = self.data_processor.get_current_window()
                    
                    if data_window is not None:
                        # Run anomaly detection
                        anomaly_score, feature_contributions = self.anomaly_engine.detect_anomaly(data_window)
                        
                        # Check if it's an anomaly
                        is_anomaly = self.anomaly_engine.is_anomaly(anomaly_score)
                        
                        # ===== RECORD SCORE TO CSV =====
                        self.score_recorder.record_score(
                            timestamp=timestamp,
                            anomaly_score=anomaly_score,
                            threshold=self.anomaly_engine.threshold,
                            is_anomaly=is_anomaly
                        )
                        
                        # Update counters
                        self.readings_processed += 1
                        if is_anomaly:
                            self.anomalies_detected += 1
                        
                        # Format feature contributions
                        feature_contrib_dict = {
                            name: float(score) for name, score in 
                            zip(self.config.WEATHER_FEATURES, feature_contributions)
                        }
                        
                        # Format output as JSON for both normal and anomaly cases
                        result_json = self.output_handler.format_anomaly_output(
                            timestamp=timestamp,
                            reading=reading,
                            anomaly_score=anomaly_score,
                            threshold=self.anomaly_engine.threshold,
                            feature_contributions=feature_contrib_dict
                        )
                        
                        # Add status field to JSON
                        result_dict = json.loads(result_json)
                        result_dict['status'] = 'ANOMALY' if is_anomaly else 'NORMAL'
                        result_dict['sequence_number'] = self.readings_processed
                        result_json = json.dumps(result_dict, indent=2)
                        
                        # Output JSON to stdout for ALL readings
                        print(result_json, flush=True)
                        
                        # Print summary line to stderr for easy tracking
                        import sys
                        status_emoji = "🔴" if is_anomaly else "🟢"
                        print(f"{status_emoji} [{timestamp.strftime('%H:%M:%S')}] #{self.readings_processed} | Score: {anomaly_score:.4f} | Status: {'ANOMALY' if is_anomaly else 'NORMAL'} | Total Anomalies: {self.anomalies_detected}", 
                              file=sys.stderr, flush=True)
                        
                        # Log to file
                        status = "ANOMALY" if is_anomaly else "NORMAL"
                        self.logger.info(f"Score: {anomaly_score:.6f}, Threshold: {self.anomaly_engine.threshold}, Status: {status}")
                        
                        # Pass to agent for processing ONLY if it's an anomaly
                        if is_anomaly and self.anomaly_agent:
                            import sys
                            print(f"[AGENT] 🤖 Starting analysis for anomaly #{self.anomalies_detected} at {timestamp.strftime('%H:%M:%S')}", file=sys.stderr, flush=True)
                            self._call_agent_sync(result_json)
                            print(f"[AGENT] ✅ Analysis complete", file=sys.stderr, flush=True)
                else:
                    # Log building status to stderr
                    storage_status = self.data_processor.get_storage_status()
                    total_count = storage_status['total_records']
                    needed = self.data_processor.win_size
                    import sys
                    print(f"⏳ [{timestamp.strftime('%H:%M:%S')}] Building storage: {total_count}/{needed} samples", 
                          file=sys.stderr, flush=True)
                
                self.data_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing data: {e}")
                import sys
                import traceback
                print(f"❌ ERROR: {e}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
                
    def get_system_status(self) -> Dict:
        """Get current system status including agent information"""
        status = {
            'running': self.running,
            'mqtt_connected': self.mqtt_client.connected if self.mqtt_client else False,
            'mqtt_broker': self.config.MQTT_BROKER,
            'mqtt_topic': self.config.MQTT_TOPIC,
            'queue_size': self.data_queue.qsize(),
            'threshold': self.anomaly_engine.threshold if self.anomaly_engine else None,
            'scaling_enabled': self.scaler is not None,
            'feature_names': self.config.WEATHER_FEATURES,
            'win_size': self.config.WIN_SIZE,
            'agent_configured': self.anomaly_agent is not None,
            'agent_type': type(self.anomaly_agent).__name__ if self.anomaly_agent else None,
            'readings_processed': self.readings_processed,
            'anomalies_detected': self.anomalies_detected,
            'score_file': self.score_file
        }
        
        # Get storage status
        if self.data_processor:
            storage_status = self.data_processor.get_storage_status()
            status.update({
                'total_records': storage_status['total_records'],
                'database_path': storage_status['database_path'],
                'csv_path': storage_status['csv_path'],
                'has_enough_data': storage_status['has_enough_data']
            })
            
            if self.last_reading_time:
                status['last_reading_time'] = self.last_reading_time.isoformat()
        else:
            status.update({
                'total_records': 0,
                'database_path': 'N/A',
                'csv_path': 'N/A',
                'has_enough_data': False
            })
        
        return status

    def get_score_summary(self) -> Dict:
        """Get summary statistics from recorded scores"""
        try:
            if not Path(self.score_file).exists():
                return {'error': 'No scores recorded yet'}
            
            import pandas as pd
            df = pd.read_csv(self.score_file)
            
            if len(df) == 0:
                return {'error': 'No scores recorded yet'}
            
            return {
                'total_scores': len(df),
                'total_anomalies': df['is_anomaly'].sum(),
                'anomaly_rate': f"{(df['is_anomaly'].sum() / len(df) * 100):.2f}%",
                'min_score': df['anomaly_score'].min(),
                'max_score': df['anomaly_score'].max(),
                'mean_score': df['anomaly_score'].mean(),
                'median_score': df['anomaly_score'].median(),
                'std_score': df['anomaly_score'].std()
            }
        except Exception as e:
            return {'error': str(e)}

    def start(self):
        """Start the monitoring system"""
        if self.running:
            self.logger.warning("Monitor is already running")
            return False
            
        self.logger.info("Starting Weather Anomaly Monitor...")
        self.logger.info(f"Using threshold: {self.anomaly_engine.threshold}")
        self.logger.info(f"Scaling enabled: {self.scaler is not None}")
        self.logger.info(f"Features: {self.config.WEATHER_FEATURES}")
        self.logger.info(f"Agent configured: {self.anomaly_agent is not None}")
        self.logger.info(f"Recording scores to: {self.score_file}")
        
        # Connect to MQTT
        if not self.mqtt_client.connect():
            self.logger.error("Failed to connect to MQTT broker")
            return False
            
        # Start processing thread
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_data, daemon=True)
        self.processing_thread.start()
        
        self.logger.info(f"Monitoring started - listening to {self.config.MQTT_TOPIC}")
        
        # Print system status to stderr
        import sys
        print("\n" + "="*70, file=sys.stderr)
        print("🌦️  WEATHER ANOMALY MONITOR STARTED", file=sys.stderr)
        print("="*70, file=sys.stderr)
        status = self.get_system_status()
        for key, value in status.items():
            print(f"  {key}: {value}", file=sys.stderr)
        print("="*70, file=sys.stderr)
        print("⏱️  Waiting for weather data...", file=sys.stderr)
        print("📊 Showing ALL results (NORMAL + ANOMALY)", file=sys.stderr)
        print(f"📝 Recording ALL scores to: {self.score_file}", file=sys.stderr)
        print("💡 Tip: Use 'tmux' or redirect to file for logging", file=sys.stderr)
        print("="*70 + "\n", file=sys.stderr, flush=True)
        
        return True
        
    def stop(self):
        """Stop the monitoring system"""
        self.logger.info("Stopping Weather Anomaly Monitor...")
        self.running = False
        
        if self.mqtt_client:
            self.mqtt_client.disconnect()
            
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join()
            
        # Print final summary
        import sys
        print("\n" + "="*70, file=sys.stderr)
        print("📊 FINAL SUMMARY", file=sys.stderr)
        print("="*70, file=sys.stderr)
        print(f"  Total readings processed: {self.readings_processed}", file=sys.stderr)
        print(f"  Total anomalies detected: {self.anomalies_detected}", file=sys.stderr)
        if self.readings_processed > 0:
            print(f"  Anomaly rate: {(self.anomalies_detected/self.readings_processed)*100:.2f}%", file=sys.stderr)
        print(f"  Scores saved to: {self.score_file}", file=sys.stderr)
        
        # Show score statistics
        score_summary = self.get_score_summary()
        if 'error' not in score_summary:
            print("\n  Score Statistics:", file=sys.stderr)
            print(f"    Min: {score_summary['min_score']:.6f}", file=sys.stderr)
            print(f"    Max: {score_summary['max_score']:.6f}", file=sys.stderr)
            print(f"    Mean: {score_summary['mean_score']:.6f}", file=sys.stderr)
            print(f"    Median: {score_summary['median_score']:.6f}", file=sys.stderr)
            print(f"    Std Dev: {score_summary['std_score']:.6f}", file=sys.stderr)
        
        print("="*70, file=sys.stderr, flush=True)
        
        self.logger.info("Monitor stopped")
        
    def run_forever(self):
        """Run the monitor in an infinite loop"""
        if not self.start():
            return
            
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        finally:
            self.stop()
            
    def run_with_status_updates(self, status_interval: int = 300):
        """Run the monitor with periodic status updates and heartbeat"""
        if not self.start():
            return
            
        last_status_time = time.time()
        heartbeat_count = 0
        
        try:
            while True:
                time.sleep(10)  # Check every 10 seconds
                
                # Show heartbeat every 30 seconds
                heartbeat_count += 1
                if heartbeat_count % 3 == 0:  # Every 30 seconds
                    import sys
                    uptime = int(time.time() - (last_status_time if last_status_time else time.time()))
                    print(f"💓 [{datetime.now().strftime('%H:%M:%S')}] Monitor alive | Uptime: {uptime//60}m | Processed: {self.readings_processed} | Anomalies: {self.anomalies_detected}", 
                          file=sys.stderr, flush=True)
                
                # Detailed status update
                current_time = time.time()
                if current_time - last_status_time >= status_interval:
                    import sys
                    print(f"\n{'='*70}", file=sys.stderr)
                    print(f"📋 STATUS UPDATE ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})", file=sys.stderr)
                    print('='*70, file=sys.stderr)
                    status = self.get_system_status()
                    print(f"  🔢 Readings processed: {status.get('readings_processed', 0)}", file=sys.stderr)
                    print(f"  🚨 Anomalies detected: {status.get('anomalies_detected', 0)}", file=sys.stderr)
                    print(f"  📦 Storage: {status.get('total_records', 0)}/{status['win_size']} records", file=sys.stderr)
                    print(f"  📨 Queue size: {status['queue_size']}", file=sys.stderr)
                    print(f"  🌐 MQTT: {'✅ Connected' if status['mqtt_connected'] else '❌ Disconnected'}", file=sys.stderr)
                    print(f"  🤖 Agent: {'✅ Configured' if status['agent_configured'] else '❌ Not configured'}", file=sys.stderr)
                    print(f"  🎯 Threshold: {status['threshold']}", file=sys.stderr)
                    print(f"  📝 Score file: {status['score_file']}", file=sys.stderr)
                    if 'last_reading_time' in status:
                        print(f"  ⏰ Last reading: {status['last_reading_time']}", file=sys.stderr)
                    
                    # Score statistics
                    score_summary = self.get_score_summary()
                    if 'error' not in score_summary:
                        print(f"\n  📊 Score Statistics:", file=sys.stderr)
                        print(f"    Mean: {score_summary['mean_score']:.6f}", file=sys.stderr)
                        print(f"    Range: {score_summary['min_score']:.6f} - {score_summary['max_score']:.6f}", file=sys.stderr)
                    
                    print('='*70 + '\n', file=sys.stderr, flush=True)
                    last_status_time = current_time
                    
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        finally:
            self.stop()

    def test_with_dummy_anomaly(self):
        """Test the system by injecting a dummy anomaly"""
        if not self.running:
            import sys
            print("❌ Monitor not running. Start the monitor first.", file=sys.stderr)
            return
            
        dummy_anomaly = {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S+00'),
            'tt': 55.0,  'rh': 15.0,  'pp': 950.0,  'ws': 45.0,
            'wd': 180.0, 'sr': 1500.0, 'rr': 0.0
        }
        
        import sys
        print("🧪 Injecting test anomaly...", file=sys.stderr, flush=True)
        self._on_mqtt_message(dummy_anomaly, datetime.now())
        print("✅ Test anomaly injected. Agent should process it.", file=sys.stderr, flush=True)


# Example usage:
if __name__ == "__main__":
    import sys
    import os
    
    # Add src directory to Python path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))
    
    from config import Config
    
    from Anomaly_detectin_ori.weather_anomaly_monitor.station_agent import WeatherAnomalyAgent
    station_agent = WeatherAnomalyAgent(llm_name ="gpt-4-turbo")
    

    # Create monitor with agent and score recording
    monitor = WeatherAnomalyMonitor(
        Config, 
        anomaly_agent=station_agent,
        score_file="logs/anomaly_scores.csv"  # Customize path if needed
    )
    
    # Run monitor with status updates every 5 minutes
    monitor.run_with_status_updates(status_interval=300)