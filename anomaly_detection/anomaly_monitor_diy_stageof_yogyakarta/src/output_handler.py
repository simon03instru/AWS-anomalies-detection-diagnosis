import json
from datetime import datetime
from typing import Dict, Any, List, Tuple
import logging

class OutputHandler:
    """Handles formatting and output of anomaly detection results"""
    
    def __init__(self, output_format: str = "json"):
        self.output_format = output_format.lower()
        self.logger = logging.getLogger(__name__)
        
    def format_anomaly_output(self, timestamp: datetime, reading: Dict, 
                            anomaly_score: float, threshold: float,
                            feature_contributions: Dict[str, float]) -> str:
        """Format anomaly data for output - only top 3 anomalous features and their data"""
        
        # Get top 3 contributing features
        top_3_features = self._get_top_3_features(feature_contributions)
        
        # Extract only the weather data for top 3 features
        top_3_weather_data = {}
        for feature_info in top_3_features:
            feature_name = feature_info["name"]
            if feature_name in reading:
                top_3_weather_data[feature_name] = reading[feature_name]
        
        anomaly_data = {
            "timestamp": timestamp.isoformat(),
            "anomaly_score": round(anomaly_score, 6),
            "weather_data": top_3_weather_data,  # Only top 3 features' data
            "top_3_contributing_features": top_3_features
        }
        
        if self.output_format == "json":
            return json.dumps(anomaly_data, separators=(',', ':'))
        elif self.output_format == "csv":
            return self._format_csv(anomaly_data)
        else:
            return str(anomaly_data)
    
    def _get_top_3_features(self, feature_contributions: Dict[str, float]) -> List[Dict[str, Any]]:
        """Get top 3 contributing features with their scores"""
        if not feature_contributions:
            return []
        
        # Sort features by contribution score (descending)
        sorted_features = sorted(feature_contributions.items(), key=lambda x: x[1], reverse=True)
        
        # Take top 3 and format them
        top_3 = []
        for feature_name, score in sorted_features[:3]:
            top_3.append({
                "name": feature_name,
                "score": round(score, 6)
            })
        
        return top_3
            
    def _format_csv(self, data: Dict[str, Any]) -> str:
        """Format data as CSV row"""
        # Extract top feature for CSV
        top_feature = data['top_3_contributing_features'][0] if data['top_3_contributing_features'] else {"name": "unknown", "score": 0.0}
        
        # Create CSV with weather data fields
        weather_values = ",".join([str(v) for v in data['weather_data'].values()])
        
        return f"{data['timestamp']},{data['anomaly_score']},{weather_values},{top_feature['name']},{top_feature['score']}"
        
    def print_anomaly(self, formatted_output: str):
        """Print anomaly to stdout"""
        print(formatted_output, flush=True)