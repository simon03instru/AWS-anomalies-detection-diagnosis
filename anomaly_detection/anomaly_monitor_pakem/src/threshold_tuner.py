# threshold_tuner.py - Helper script for finding the right threshold
import time
from config import Config
from main_monitor import WeatherAnomalyMonitor

def test_thresholds():
    """Test different threshold values to find the right sensitivity"""
    
    test_thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    
    print("Threshold Testing Tool")
    print("This will test different threshold values with your current data stream")
    print("Send some normal and anomalous data to see which threshold works best")
    print("-" * 60)
    
    monitor = WeatherAnomalyMonitor(Config)
    
    for threshold in test_thresholds:
        print(f"\nTesting threshold: {threshold}")
        print("Press Enter to continue to next threshold, or Ctrl+C to stop")
        
        # Update threshold
        monitor.update_threshold(threshold)
        
        # Run for a short time
        if monitor.start():
            try:
                time.sleep(30)  # Test for 30 seconds
            except KeyboardInterrupt:
                break
            finally:
                monitor.stop()
        
        try:
            input()  # Wait for user input
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    test_thresholds()