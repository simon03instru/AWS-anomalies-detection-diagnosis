#!/usr/bin/env python3
"""
Weather Anomaly Investigation Agent using CAMEL AI Platform with MCP Tools
Analyzes weather data for anomalies using MCP server tools via MCPToolkit.
Clean version with minimal debug output - logs to file only.
ENHANCED: Now includes automatic evaluation metrics calculation.
EVALUATION MODE: Skips MQTT publishing, focuses on detection accuracy.
"""

import sys
import json
import asyncio
import argparse
import logging
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import re

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.toolkits.mcp_toolkit import MCPToolkit

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


# ANSI color codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"

# Initialize local LLM model
ollama_model = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="gpt-oss:120b",
    url="http://10.33.205.34:11440/v1",
    model_config_dict={
        "temperature": 0,
        "max_tokens": 16384,
    },
)


class EvaluationMetrics:
    """Calculate and track evaluation metrics for the agent"""
    
    def __init__(self):
        self.results = []
        self.confusion_matrix = {
            'true_positives': defaultdict(int),
            'false_positives': defaultdict(int),
            'false_negatives': defaultdict(int),
            'true_negatives': defaultdict(int)
        }
        
    def add_result(self, true_anomalies: Set[str], predicted_anomalies: Set[str], 
                   all_features: Set[str], case_id: str = None):
        """
        Add a single evaluation result
        
        Args:
            true_anomalies: Set of actual anomalous features (ground truth)
            predicted_anomalies: Set of features predicted as anomalous by agent
            all_features: Set of all possible features in the data
            case_id: Optional identifier for this case
        """
        result = {
            'case_id': case_id,
            'true_anomalies': sorted(list(true_anomalies)),
            'predicted_anomalies': sorted(list(predicted_anomalies)),
            'true_positives': sorted(list(true_anomalies & predicted_anomalies)),
            'false_positives': sorted(list(predicted_anomalies - true_anomalies)),
            'false_negatives': sorted(list(true_anomalies - predicted_anomalies)),
            'true_negatives': sorted(list(all_features - true_anomalies - predicted_anomalies))
        }
        
        self.results.append(result)
        
        # Update confusion matrix per feature
        for feature in result['true_positives']:
            self.confusion_matrix['true_positives'][feature] += 1
        for feature in result['false_positives']:
            self.confusion_matrix['false_positives'][feature] += 1
        for feature in result['false_negatives']:
            self.confusion_matrix['false_negatives'][feature] += 1
        for feature in result['true_negatives']:
            self.confusion_matrix['true_negatives'][feature] += 1
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate overall and per-feature metrics"""
        if not self.results:
            return {
                'error': 'No results to calculate metrics from'
            }
        
        # Overall metrics (micro-averaged)
        total_tp = sum(self.confusion_matrix['true_positives'].values())
        total_fp = sum(self.confusion_matrix['false_positives'].values())
        total_fn = sum(self.confusion_matrix['false_negatives'].values())
        total_tn = sum(self.confusion_matrix['true_negatives'].values())
        
        # Calculate overall metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn) if (total_tp + total_fp + total_fn + total_tn) > 0 else 0
        
        # Per-feature metrics
        all_features = set()
        for key in self.confusion_matrix.values():
            all_features.update(key.keys())
        
        per_feature_metrics = {}
        for feature in all_features:
            tp = self.confusion_matrix['true_positives'][feature]
            fp = self.confusion_matrix['false_positives'][feature]
            fn = self.confusion_matrix['false_negatives'][feature]
            tn = self.confusion_matrix['true_negatives'][feature]
            
            feat_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            feat_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            feat_f1 = 2 * (feat_precision * feat_recall) / (feat_precision + feat_recall) if (feat_precision + feat_recall) > 0 else 0
            
            per_feature_metrics[feature] = {
                'precision': round(feat_precision, 4),
                'recall': round(feat_recall, 4),
                'f1_score': round(feat_f1, 4),
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'true_negatives': tn
            }
        
        return {
            'overall': {
                'accuracy': round(accuracy, 4),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1_score': round(f1, 4),
                'total_cases': len(self.results),
                'total_true_positives': total_tp,
                'total_false_positives': total_fp,
                'total_false_negatives': total_fn,
                'total_true_negatives': total_tn
            },
            'per_feature': per_feature_metrics,
            'detailed_results': self.results
        }
    
    def print_metrics_summary(self):
        """Print a formatted summary of metrics"""
        metrics = self.calculate_metrics()
        
        if 'error' in metrics:
            print(f"{RED}{metrics['error']}{RESET}")
            return
        
        print(f"\n{BOLD}{CYAN}{'='*70}{RESET}")
        print(f"{BOLD}{CYAN}EVALUATION METRICS SUMMARY{RESET}")
        print(f"{BOLD}{CYAN}{'='*70}{RESET}\n")
        
        # Overall metrics
        overall = metrics['overall']
        print(f"{BOLD}{GREEN}Overall Performance (Micro-averaged):{RESET}")
        print(f"  Total Cases Evaluated: {overall['total_cases']}")
        print(f"  Accuracy:   {BOLD}{overall['accuracy']*100:.2f}%{RESET}")
        print(f"  Precision:  {BOLD}{overall['precision']*100:.2f}%{RESET}")
        print(f"  Recall:     {BOLD}{overall['recall']*100:.2f}%{RESET}")
        print(f"  F1 Score:   {BOLD}{overall['f1_score']*100:.2f}%{RESET}")
        print(f"\n  Confusion Matrix (Total):")
        print(f"    True Positives:  {overall['total_true_positives']}")
        print(f"    False Positives: {overall['total_false_positives']}")
        print(f"    False Negatives: {overall['total_false_negatives']}")
        print(f"    True Negatives:  {overall['total_true_negatives']}")
        
        # Per-feature metrics
        if metrics['per_feature']:
            print(f"\n{BOLD}{BLUE}Per-Feature Performance:{RESET}")
            print(f"{'Feature':<10} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'TP':<6} {'FP':<6} {'FN':<6}")
            print(f"{'-'*70}")
            
            for feature, feat_metrics in sorted(metrics['per_feature'].items()):
                print(f"{feature:<10} "
                      f"{feat_metrics['precision']*100:>6.2f}%     "
                      f"{feat_metrics['recall']*100:>6.2f}%     "
                      f"{feat_metrics['f1_score']*100:>6.2f}%     "
                      f"{feat_metrics['true_positives']:<6} "
                      f"{feat_metrics['false_positives']:<6} "
                      f"{feat_metrics['false_negatives']:<6}")
        
        # Show detailed results
        print(f"\n{BOLD}{BLUE}Detailed Case Results:{RESET}")
        print(f"{'Case ID':<15} {'True':<20} {'Predicted':<20} {'Match':<8}")
        print(f"{'-'*70}")
        for result in metrics['detailed_results']:
            case_id = str(result['case_id'])[:14]
            true_str = ', '.join(result['true_anomalies']) if result['true_anomalies'] else 'none'
            pred_str = ', '.join(result['predicted_anomalies']) if result['predicted_anomalies'] else 'none'
            match = '✓' if set(result['true_anomalies']) == set(result['predicted_anomalies']) else '✗'
            print(f"{case_id:<15} {true_str:<20} {pred_str:<20} {match:<8}")
        
        print(f"\n{BOLD}{CYAN}{'='*70}{RESET}\n")
    
    def save_metrics_to_file(self, filepath: str):
        """Save detailed metrics to a JSON file"""
        metrics = self.calculate_metrics()
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"{GREEN}Metrics saved to: {filepath}{RESET}")


class ThoughtProcessLogger:
    """Silent logger that only writes to file"""
    
    def __init__(self, log_file: str):
        self.log_entries = []
        self.log_file = log_file
        
        # Setup file logger
        self.file_logger = logging.getLogger('thought_process')
        self.file_logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.file_logger.handlers.clear()
        
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # File handler
        fh = logging.FileHandler(log_file, mode='a')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        self.file_logger.addHandler(fh)
    
    def log_step(self, step_type: str, content: str):
        """Log a step silently to file only"""
        timestamp = datetime.now().isoformat()
        entry = {
            "timestamp": timestamp,
            "step_type": step_type,
            "content": content
        }
        self.log_entries.append(entry)
        self.file_logger.info(f"[{step_type}] {content}")
    
    def log_tool_call(self, tool_name: str, arguments: Dict[str, Any]):
        """Log tool call to file"""
        args_str = json.dumps(arguments, indent=2)
        content = f"Calling tool: {tool_name}\nArguments:\n{args_str}"
        self.log_step("TOOL_CALL", content)

    def log_tool_result(self, tool_name: str, result: str, truncate: int = 200):
        """Log tool result to file - skip detailed logging for get_data_from_db"""
        # Skip detailed logging for database queries (too verbose)
        if tool_name == "get_data_from_db":
            # Just log a summary for database queries
            try:
                result_data = json.loads(result) if isinstance(result, str) else result
                record_count = len(result_data.get("records", [])) if isinstance(result_data, dict) else 0
                result_preview = f"[Database query returned {record_count} records, {len(result)} characters total]"
            except:
                result_preview = f"[Database query executed - {len(result)} characters returned]"
        else:
            # For other tools, show a preview
            result_preview = result[:truncate] + "..." if len(result) > truncate else result
        
        content = f"Tool '{tool_name}' returned: {result_preview}"
        self.log_step("TOOL_RESULT", content)

    
    def log_reasoning(self, reasoning: str):
        """Log agent's reasoning to file"""
        self.log_step("REASONING", reasoning)
    
    def log_decision(self, decision: str):
        """Log agent's decision to file"""
        self.log_step("DECISION", decision)
    
    def log_error(self, error: str):
        """Log an error to file"""
        self.log_step("ERROR", error)
    
    def log_analysis_start(self, anomaly_data: Dict[str, Any]):
        """Log the start of analysis to file"""
        content = f"Starting analysis of anomaly data:\n{json.dumps(anomaly_data, indent=2)}"
        self.log_step("ANALYSIS_START", content)
    
    def log_analysis_end(self, result: str):
        """Log the COMPLETE analysis result to file"""
        # Log the full result, not just the length
        separator = "=" * 80
        content = f"Analysis complete.\n\n{separator}\nFULL ANALYSIS RESULT:\n{separator}\n\n{result}\n\n{separator}"
        self.log_step("ANALYSIS_RESULT", content)
        
        # Also log a summary for quick reference
        summary = f"Result length: {len(result)} characters, {len(result.split())} words"
        self.log_step("ANALYSIS_SUMMARY", summary)
    
    def save_summary(self, output_file: str):
        """Save a summary of the thought process to a JSON file"""
        summary = {
            "session_start": self.log_entries[0]["timestamp"] if self.log_entries else None,
            "session_end": self.log_entries[-1]["timestamp"] if self.log_entries else None,
            "total_steps": len(self.log_entries),
            "steps": self.log_entries
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)


class WeatherAnomalyAgent:
    """Weather Anomaly Investigation Agent using CAMEL AI with MCP server tools"""
    
    def __init__(
        self, 
        mcp_config_path: str,
        model_platform: str = "openai",
        model_type: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        thought_logger: Optional[ThoughtProcessLogger] = None,
        evaluation_mode: bool = False
    ):
        self.mcp_config_path = mcp_config_path
        self.model_platform = model_platform
        self.model_type = model_type
        self.temperature = temperature
        self.thought_logger = thought_logger
        self.mcp_toolkit = None
        self.agent = None
        self.evaluation_mode = evaluation_mode
        self.evaluator = EvaluationMetrics() if evaluation_mode else None
        
        if self.thought_logger:
            mode_str = "EVALUATION MODE" if evaluation_mode else "PRODUCTION MODE"
            self.thought_logger.log_step("INITIALIZATION", f"Initializing Weather Anomaly Agent - {mode_str}...")
    
    async def initialize(self):
        """Initialize MCP toolkit and CAMEL agent asynchronously"""
        try:
            # Initialize MCP toolkit
            if self.thought_logger:
                self.thought_logger.log_step("INITIALIZATION", f"Loading MCP config: {self.mcp_config_path}")
            
            self.mcp_toolkit = MCPToolkit(config_path=self.mcp_config_path)
            
            # Connect to MCP server
            if self.thought_logger:
                self.thought_logger.log_step("INITIALIZATION", "Connecting to MCP server...")
            
            await self.mcp_toolkit.connect()
            
            # Get tools
            tools = self.mcp_toolkit.get_tools()
            tool_names = [tool.get_function_name() for tool in tools]
            
            if self.thought_logger:
                self.thought_logger.log_step(
                    "TOOL_DISCOVERY",
                    f"Connected! Available tools: {', '.join(tool_names)}"
                )
            
            # Filter out MQTT tool in evaluation mode
            if self.evaluation_mode:
                tools = [t for t in tools if 'send_anomalous_status' not in t.get_function_name()]
                if self.thought_logger:
                    self.thought_logger.log_step(
                        "EVALUATION_MODE",
                        "MQTT publishing disabled for evaluation - focusing on detection only"
                    )
            
            # Create CAMEL agent
            if self.thought_logger:
                self.thought_logger.log_step("INITIALIZATION", "Creating CAMEL agent...")
            
            self.agent = self._create_agent(tools)
            
            if self.thought_logger:
                self.thought_logger.log_step("INITIALIZATION", "Agent initialization complete")
            
            return True
            
        except Exception as e:
            error_msg = f"Error during initialization: {e}"
            if self.thought_logger:
                self.thought_logger.log_error(error_msg)
            import traceback
            traceback.print_exc()
            return False
    
    def _create_agent(self, tools):
        """Create the CAMEL agent with MCP tools."""
        try:
            # Different system messages for evaluation vs production
            if self.evaluation_mode:
                system_message = """You are a weather sensor Anomaly Investigation Agent in EVALUATION MODE. Your task is to identify anomalous features WITHOUT publishing to MQTT.

                PROCESS:
                1) IDENTIFY: Extract top 3 anomalous features from provided data. Timestamp in received data is UTC, and local time is UTC+7.

                2) RETRIEVE: Get latest 20 records for ALL parameters using get_data_from_db(features="tt,rh,pp,ws,wd,sr,rr", limit=20)

                3) IMMEDIATE SENSOR FAULT CHECK:
                - Invalid sentinels (e.g., -9999, 9999): Mark as anomalous
                - Physically impossible values: Mark as anomalous
                    * rh: must be 0-100%, values at exact 0% or 100% are highly suspicious
                    * wd: must be 0-360°
                    * tt: must be within reasonable range for location (-50°C to 60°C)
                    * ws: cannot be negative
                    * pp: typical range 950-1050 hPa
                - Stuck readings: same exact value for >6 intervals (60min): Mark as anomalous

                4) CORRELATION CHECK (secondary validation):
                For features not caught above, check correlations:
                - tt (Temperature): Should correlate negatively with rh, positively with sr
                - rh (Humidity): Should correlate negatively with tt, positively with rr
                - ws (Wind Speed): Should correlate negatively with pp
                - pp (Pressure): Should correlate negatively with ws
                - sr (Solar Radiation): can change rapidly (300 units/10min is normal); low at night is expected
                - wd (Wind Direction): can change rapidly; do NOT use correlation; only check range (0-360°)
                
                Note: Broken correlations SUPPORT anomaly detection but intact correlations do NOT rule out sensor faults

                5) ANALYZE: Compare current vs historical (each record = 10 min interval):
                - Deviation magnitude and direction
                - Pattern: spike/drop/gradual/stuck over how many intervals
                - Correlation status with related parameters

                6) DECIDE & REPORT: 
                Report anomalies you are CONFIDENT > 80% about based on:
                - Sentinel values or physically impossible 
                - Stuck readings (>6 intervals) 
                - Large deviation (>3 std dev)
                
                FORMAT YOUR RESPONSE AS JSON:
                {
                  "anomalous_features": {
                    "feature_code": value,
                    ...
                  },
                  "trend_analysis": {
                    "feature_code": "SHORT analysis including: type of fault, magnitude, timeframe, correlation status",
                    ...
                  }
                }

                EXAMPLE:
                {
                  "anomalous_features": {
                    "wd": -9999.0,
                    "rh": 0.0,
                    "ws": 0.6
                  },
                  "trend_analysis": {
                    "wd": "Invalid sentinel value -9999.0° - sensor communication failure",
                    "rh": "Physically impossible 0% reading - sensor fault. Expected 48-56%, dropped 100% instantly. Correlation with tt appears intact but this doesn't rule out sensor malfunction.",
                    "ws": "Stuck at 0.6 m/s for 60min (6 intervals), 88% below avg. Correlation with pp broken - sensor likely stuck/failed."
                  }
                }

                CRITICAL RULES:
                - EVALUATION MODE: Do NOT use send_anomalous_status tool - just report findings
                - ALWAYS identify confirmed sensor faults only - avoid false alarms
                - ALWAYS flag sentinel values (-9999, 9999, etc.) - these are definitive sensor faults
                - ALWAYS flag physically impossible values (rh=0%, rh=100%, wd outside 0-360°)
                - ALWAYS flag stuck readings (same value >6 intervals)
                - Intact correlations do NOT prove sensor is working - a broken sensor can show correlation by coincidence
                - For wind direction (wd): only check if value is valid (0-360°); ignore all other analyses
                - Only report "Possible false alarm" when deviation is moderate AND correlations are intact AND value is physically possible
                - RESPOND IN JSON FORMAT with "anomalous_features" and "trend_analysis" keys
                """
            else:
                system_message = """You are a weather sensor Anomaly Investigation Agent to publish findings of a potential sensor malfunction. Systematically analyze sensor anomalies and publish confirmed ones.

                PROCESS:
                1) IDENTIFY: Extract top 3 anomalous features from provided data. Timestamp in received data is UTC, and local time is UTC+7.

                2) RETRIEVE: Get latest 20 records for ALL parameters using get_data_from_db(features="tt,rh,pp,ws,wd,sr,rr", limit=20)

                3) IMMEDIATE SENSOR FAULT CHECK:
                - Invalid sentinels (e.g., -9999, 9999): PUBLISH immediately
                - Physically impossible values: PUBLISH immediately
                    * rh: must be 0-100%, values at exact 0% or 100% are highly suspicious
                    * wd: must be 0-360°
                    * tt: must be within reasonable range for location (-50°C to 60°C)
                    * ws: cannot be negative
                    * pp: typical range 950-1050 hPa
                - Stuck readings: same exact value for >6 intervals (60min): PUBLISH immediately

                4) CORRELATION CHECK (secondary validation):
                For features not caught above, check correlations:
                - tt (Temperature): Should correlate negatively with rh, positively with sr
                - rh (Humidity): Should correlate negatively with tt, positively with rr. Sudden rh change should align with rr and tt change
                - ws (Wind Speed): Should correlate negatively with pp
                - pp (Pressure): Should correlate negatively with ws
                - sr (Solar Radiation): can change rapidly (300 units/10min is normal); low at night is expected
                - wd (Wind Direction): can change rapidly; do NOT use correlation; only check range (0-360°)
                - rr (Rainfall): sudden increases expected during rain, spike is normal. However, check against rh spikes

                Note: Broken correlations SUPPORT anomaly detection but intact correlations do NOT rule out sensor faults

                5) ANALYZE: Compare current vs historical (each record = 10 min interval):
                - Deviation Magnitude is possibly normal, however check against correlation status
                - Pattern: spike/drop/gradual/stuck over how many intervals
                - Correlation status with related parameters

                6) DECIDE: 
                Publish anomaly if you are CONFIDENT > 80% based on:
                - Sentinel values or physically impossible 
                - Stuck readings (>6 intervals) 
                - Large deviation (>3 std dev) 
                

                7) PUBLISH: Use send_anomalous_status with:
                - anomalous_features: {feature_code: value}
                - trend_analysis: {feature_code: "SHORT analysis including: type of fault, magnitude, timeframe, correlation status"}

                EXAMPLE:
                {
                "wd": "Invalid sentinel value -9999.0° - sensor communication failure",
                "rh": "Physically impossible 0% reading - sensor fault. Expected 48-56%, dropped 100% instantly. Correlation with tt appears intact but this doesn't rule out sensor malfunction.",
                "ws": "Stuck at 0.6 m/s for 60min (6 intervals), 88% below avg. Correlation with pp broken - sensor likely stuck/failed."
                }

                CRITICAL RULES:
                - ALWAYS publish confirmed sensor faults only - avoid false alarms
                - ALWAYS publish sentinel values (-9999, 9999, etc.) - these are definitive sensor faults
                - ALWAYS publish physically impossible values (rh=0%, rh=100%, wd outside 0-360°)
                - ALWAYS publish stuck readings (same value >6 intervals)
                - Intact correlations do NOT prove sensor is working - a broken sensor can show correlation by coincidence
                - For wind direction (wd): only check if value is valid (0-360°); ignore all other analyses
                - Only report "Possible false alarm" when deviation is moderate AND correlations are intact AND value is physically possible
                """

            agent = ChatAgent(
                system_message=system_message,
                model=ollama_model,
                tools=[*tools]  # Unpack the tools list
            )
            
            return agent
            
        except Exception as e:
            error_msg = f"Error creating agent: {e}"
            if self.thought_logger:
                self.thought_logger.log_error(error_msg)
            import traceback
            traceback.print_exc()
            return None
    
    def extract_predicted_anomalies(self, agent_response: str) -> Set[str]:
        """
        Extract anomalous features from agent's response
        Looks for JSON format ONLY - more precise extraction
        """
        predicted = set()
        
        # Try to find JSON in response with anomalous_features
        try:
            # Look for JSON patterns with anomalous_features key
            json_patterns = [
                r'\{[^{}]*"anomalous_features"[^{}]*\{[^}]+\}[^}]*\}',  # Nested JSON with anomalous_features
                r'\{[^{}]*"anomalous_features"[^{}]*:\s*\{[^}]+\}',     # Just the anomalous_features section
            ]
            
            for pattern in json_patterns:
                json_matches = re.finditer(pattern, agent_response, re.DOTALL)
                for json_match in json_matches:
                    try:
                        json_str = json_match.group(0)
                        
                        # Try to complete the JSON if it's incomplete
                        if not json_str.endswith('}'):
                            json_str += '}'
                        
                        data = json.loads(json_str)
                        
                        # ONLY use anomalous_features key - this is the authoritative source
                        if 'anomalous_features' in data:
                            anomalous_features = data['anomalous_features']
                            if isinstance(anomalous_features, dict):
                                predicted.update(anomalous_features.keys())
                                if self.thought_logger:
                                    self.thought_logger.log_step(
                                        "EXTRACTION_SUCCESS",
                                        f"Extracted anomalous features from JSON: {predicted}"
                                    )
                                return predicted  # Return immediately once found
                            
                    except json.JSONDecodeError:
                        continue
            
            # If no proper JSON found, try to extract just the anomalous_features section
            anomalous_section_match = re.search(
                r'"anomalous_features"\s*:\s*\{([^}]+)\}',
                agent_response,
                re.DOTALL
            )
            
            if anomalous_section_match:
                features_str = anomalous_section_match.group(1)
                # Extract feature names from key-value pairs
                feature_matches = re.findall(r'"([a-z]{2})":\s*[-\d.]+', features_str)
                predicted.update(feature_matches)
                if self.thought_logger:
                    self.thought_logger.log_step(
                        "EXTRACTION_PARTIAL",
                        f"Extracted from anomalous_features section: {predicted}"
                    )
                return predicted
                
        except Exception as e:
            if self.thought_logger:
                self.thought_logger.log_step("EXTRACTION_ERROR", f"JSON extraction error: {e}")
        
        # Last resort: look for explicit anomaly mentions ONLY if JSON extraction completely failed
        if not predicted:
            if self.thought_logger:
                self.thought_logger.log_step(
                    "EXTRACTION_FALLBACK",
                    "JSON extraction failed, using fallback pattern matching"
                )
            
            feature_codes = ['tt', 'rh', 'pp', 'ws', 'wd', 'sr', 'rr']
            
            # Very strict patterns - only if clearly stated as anomalous
            for code in feature_codes:
                patterns = [
                    rf'"anomalous_features"[^}}]*"{code}":\s*[-\d.]+',  # In anomalous_features JSON
                    rf'\b{code}\b[^.]*(?:is anomalous|anomaly detected|sensor fault)',  # Explicit statements
                ]
                
                for pattern in patterns:
                    if re.search(pattern, agent_response, re.IGNORECASE):
                        predicted.add(code)
                        break
        
        if self.thought_logger and predicted:
            self.thought_logger.log_step(
                "EXTRACTION_FINAL",
                f"Final extracted anomalous features: {predicted}"
            )
        elif self.thought_logger:
            self.thought_logger.log_step(
                "EXTRACTION_WARNING",
                "No anomalous features could be extracted from agent response"
            )
        
        return predicted
    
    async def analyze(self, anomaly_data, evaluate: bool = False):
        """Analyze anomaly data and optionally evaluate against ground truth."""
        if not self.agent:
            return "Error: Agent not initialized properly. Call initialize() first."
        
        try:
            # Handle both dict and string inputs
            if isinstance(anomaly_data, dict):
                anomaly_json = json.dumps(anomaly_data, indent=2)
                data_dict = anomaly_data
            elif isinstance(anomaly_data, str):
                data_dict = json.loads(anomaly_data)
                anomaly_json = anomaly_data
            else:
                return "Error: Invalid input type. Expected dict or JSON string."
            
            # Extract ground truth if in evaluation mode
            true_anomalies = set()
            all_features = set(data_dict.get('weather_data', {}).keys())
            has_ground_truth = False
            
            if evaluate:
                # Check if true_anomaly field exists
                if 'true_anomaly' in data_dict:
                    has_ground_truth = True
                    true_anomaly_raw = data_dict['true_anomaly']
                    
                    # Handle various formats for NO anomalies
                    if true_anomaly_raw in [None, "", "none", "None", "NONE", []]:
                        true_anomalies = set()  # Explicitly empty - no true anomalies
                        if self.thought_logger:
                            self.thought_logger.log_step(
                                "GROUND_TRUTH",
                                "No true anomalies (normal data case)"
                            )
                    # Handle string format
                    elif isinstance(true_anomaly_raw, str):
                        # Split by comma if multiple features, clean whitespace and quotes
                        true_anomalies = {
                            f.strip().strip('"\'') 
                            for f in true_anomaly_raw.split(',') 
                            if f.strip() and f.strip().lower() not in ['none', '']
                        }
                    # Handle list/array format
                    elif isinstance(true_anomaly_raw, list):
                        true_anomalies = {
                            str(f).strip() 
                            for f in true_anomaly_raw 
                            if f and str(f).strip().lower() not in ['none', '']
                        }
                    else:
                        true_anomalies = {str(true_anomaly_raw).strip()}
                    
                    if self.thought_logger:
                        true_str = ', '.join(sorted(true_anomalies)) if true_anomalies else 'none (normal data)'
                        self.thought_logger.log_step(
                            "GROUND_TRUTH",
                            f"True anomalies extracted: {true_str}"
                        )
                else:
                    # If true_anomaly field is missing in evaluation mode, warn
                    if self.thought_logger:
                        self.thought_logger.log_step(
                            "EVALUATION_WARNING",
                            "true_anomaly field missing - cannot evaluate this case"
                        )
            
            # Log analysis start
            if self.thought_logger:
                self.thought_logger.log_analysis_start(data_dict)
                self.thought_logger.log_reasoning(
                    "Sending query to CAMEL agent with anomaly data for systematic analysis"
                )
            
            # Create user message
            user_message = f"""Please analyze the following weather anomaly data systematically:

    {anomaly_json}

    Follow your investigation process step by step."""
            
            # Get response from agent (async)
            if self.thought_logger:
                self.thought_logger.log_step("AGENT_PROCESSING", "Waiting for agent response...")
            
            response = await self.agent.astep(BaseMessage.make_user_message(
                role_name="User",
                content=user_message
            ))
            
            # Extract response content
            if hasattr(response, 'msgs') and response.msgs:
                result = response.msgs[0].content
            elif hasattr(response, 'msg'):
                result = response.msg.content
            else:
                result = str(response)
            
            # Log tool calls if available
            if hasattr(response, 'info') and 'tool_calls' in response.info:
                for tool_call in response.info['tool_calls']:
                    if self.thought_logger:
                        self.thought_logger.log_tool_call(tool_call.tool_name, tool_call.args)
                        self.thought_logger.log_tool_result(tool_call.tool_name, str(tool_call.result))
            
            # Evaluate if requested and we have ground truth (including empty set for normal data)
            if evaluate and has_ground_truth and self.evaluator:
                predicted_anomalies = self.extract_predicted_anomalies(result)
                
                case_id = data_dict.get('sequence_number') or data_dict.get('timestamp')
                self.evaluator.add_result(
                    true_anomalies=true_anomalies,
                    predicted_anomalies=predicted_anomalies,
                    all_features=all_features,
                    case_id=str(case_id)
                )
                
                if self.thought_logger:
                    true_str = ', '.join(sorted(true_anomalies)) if true_anomalies else 'none'
                    pred_str = ', '.join(sorted(predicted_anomalies)) if predicted_anomalies else 'none'
                    match = '✓' if true_anomalies == predicted_anomalies else '✗'
                    eval_log = f"Evaluation {match} - True: [{true_str}], Predicted: [{pred_str}]"
                    self.thought_logger.log_step("EVALUATION", eval_log)
            
            # Log COMPLETE analysis result
            if self.thought_logger:
                self.thought_logger.log_analysis_end(result)
            
            return result
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON format: {e}"
            if self.thought_logger:
                self.thought_logger.log_error(error_msg)
            return f"Error: {error_msg}"
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            error_msg = f"Error during analysis: {str(e)}\n\nDetails:\n{error_details}"
            if self.thought_logger:
                self.thought_logger.log_error(error_msg)
            return error_msg
        
    async def cleanup(self):
        """Clean up resources"""
        if self.mcp_toolkit:
            await self.mcp_toolkit.disconnect()
            if self.thought_logger:
                self.thought_logger.log_step("CLEANUP", "Disconnected from MCP server")


async def run_agent(args):
    """Run the agent with given arguments"""
    try:
        # Create thought process logger (silent, file only)
        thought_logger = ThoughtProcessLogger(log_file=args.log_file)
        
        print(f"{CYAN}Initializing agent...{RESET}")
        
        agent = WeatherAnomalyAgent(
            mcp_config_path=args.mcp_config,
            model_platform=args.model_platform,
            model_type=args.model_type,
            temperature=args.temperature,
            thought_logger=thought_logger,
            evaluation_mode=args.evaluate
        )
        
        # Initialize agent (async)
        if not await agent.initialize():
            print(f"{RED}Error: Failed to initialize agent{RESET}")
            return
        
        print(f"{GREEN}✓ Agent ready{RESET}")
        if args.evaluate:
            print(f"{MAGENTA}✓ Evaluation mode enabled (MQTT publishing disabled){RESET}")
        print(f"{CYAN}Logs: {args.log_file}{RESET}\n")
        
        try:
            # Handle batch file input for evaluation
            if args.batch_file:
                try:
                    with open(args.batch_file, 'r') as f:
                        content = f.read().strip()
                    
                    # Try to parse as JSON array
                    try:
                        batch_data = json.loads(content)
                    except json.JSONDecodeError:
                        # If not valid JSON, try to split by lines and parse each
                        lines = [line.strip() for line in content.split('\n') if line.strip()]
                        batch_data = []
                        for line in lines:
                            try:
                                batch_data.append(json.loads(line))
                            except json.JSONDecodeError:
                                print(f"{YELLOW}Warning: Skipping invalid JSON line{RESET}")
                    
                    if not isinstance(batch_data, list):
                        batch_data = [batch_data]
                    
                    print(f"{YELLOW}Processing {len(batch_data)} cases...{RESET}\n")
                    
                    for idx, case in enumerate(batch_data, 1):
                        case_id = case.get('sequence_number', case.get('timestamp', idx))
                        print(f"{CYAN}[{idx}/{len(batch_data)}] Processing case {case_id}...{RESET}")
                        result = await agent.analyze(case, evaluate=args.evaluate)
                        
                        # Show brief result if not in quiet mode
                        if not args.quiet:
                            print(f"{GREEN}Result preview:{RESET} {result[:100]}...")
                        print(f"{GREEN}✓ Complete{RESET}\n")
                    
                    # Show metrics
                    if args.evaluate and agent.evaluator:
                        agent.evaluator.print_metrics_summary()
                        
                        if args.metrics_output:
                            agent.evaluator.save_metrics_to_file(args.metrics_output)
                    
                except FileNotFoundError:
                    print(f"{RED}Error: File {args.batch_file} not found{RESET}")
                except Exception as e:
                    print(f"{RED}Error processing batch file: {e}{RESET}")
                    import traceback
                    traceback.print_exc()
            
            # Handle single file input
            elif args.file:
                try:
                    with open(args.file, 'r') as f:
                        data = f.read().strip()
                    print(f"{YELLOW}Analyzing...{RESET}\n")
                    result = await agent.analyze(data, evaluate=args.evaluate)
                    print(f"\n{GREEN}╔══ Analysis Result ══╗{RESET}\n")
                    print(result)
                    print(f"\n{GREEN}╚═══════════════════════╝{RESET}\n")
                    
                    if args.evaluate and agent.evaluator:
                        agent.evaluator.print_metrics_summary()
                        
                        if args.metrics_output:
                            agent.evaluator.save_metrics_to_file(args.metrics_output)
                    
                except FileNotFoundError:
                    print(f"{RED}Error: File {args.file} not found{RESET}")
                except Exception as e:
                    print(f"{RED}Error reading file: {e}{RESET}")
            
            # Handle JSON input
            elif args.json:
                print(f"{YELLOW}Analyzing...{RESET}\n")
                result = await agent.analyze(args.json, evaluate=args.evaluate)
                print(f"\n{GREEN}╔══ Analysis Result ══╗{RESET}\n")
                print(result)
                print(f"\n{GREEN}╚═══════════════════════╝{RESET}\n")
                
                if args.evaluate and agent.evaluator:
                    agent.evaluator.print_metrics_summary()
                    
                    if args.metrics_output:
                        agent.evaluator.save_metrics_to_file(args.metrics_output)
            
            # Interactive mode
            else:
                print("=" * 60)
                print("Weather Anomaly Investigation - Interactive Mode")
                if args.evaluate:
                    print("EVALUATION MODE: Include 'true_anomaly' field in JSON")
                print("=" * 60)
                print("Commands: 'quit', 'help', 'test', 'metrics', or paste JSON")
                print("=" * 60)
                
                while True:
                    try:
                        user_input = input(f"\n{BOLD}{GREEN}> {RESET}").strip()
                        
                        if user_input.lower() in ['quit', 'exit', 'q']:
                            if args.evaluate and agent.evaluator:
                                agent.evaluator.print_metrics_summary()
                                if args.metrics_output:
                                    agent.evaluator.save_metrics_to_file(args.metrics_output)
                            print(f"\n{CYAN}Goodbye!{RESET}\n")
                            break
                        
                        if user_input.lower() == 'metrics':
                            if args.evaluate and agent.evaluator:
                                agent.evaluator.print_metrics_summary()
                            else:
                                print(f"{YELLOW}Evaluation mode not enabled. Use --evaluate flag{RESET}")
                            continue
                        
                        if user_input.lower() == 'help':
                            sample = {
                                "timestamp": "2024-08-19T07:40:00",
                                "anomaly_score": 1.445168,
                                "true_anomaly": "tt, rh",  # For evaluation
                                "weather_data": {
                                    "tt": 13.0,
                                    "rh": 0.1,
                                    "ws": 1.511
                                },
                                "top_3_contributing_features": [
                                    {"name": "tt", "score": 9.570851},
                                    {"name": "rh", "score": 0.177596},
                                    {"name": "ws", "score": 0.15328}
                                ],
                                "status": "ANOMALY",
                                "sequence_number": 1
                            }
                            print(f"\n{CYAN}Sample JSON format:{RESET}")
                            print(json.dumps(sample, indent=2))
                            print(f"\n{YELLOW}Note: 'true_anomaly' field is required for evaluation mode{RESET}")
                            print(f"\n{CYAN}For batch file, use JSON array or newline-separated JSON objects{RESET}")
                            continue
                        
                        if user_input:
                            print(f"\n{YELLOW}Analyzing...{RESET}\n")
                            result = await agent.analyze(user_input, evaluate=args.evaluate)
                            print(f"\n{GREEN}╔══ Analysis Result ══╗{RESET}\n")
                            print(result)
                            print(f"\n{GREEN}╚═══════════════════════╝{RESET}\n")
                        else:
                            print(f"{YELLOW}Please provide JSON data or type 'help'{RESET}")
                            
                    except KeyboardInterrupt:
                        if args.evaluate and agent.evaluator:
                            agent.evaluator.print_metrics_summary()
                            if args.metrics_output:
                                agent.evaluator.save_metrics_to_file(args.metrics_output)
                        print(f"\n\n{CYAN}Goodbye!{RESET}\n")
                        break
                    except Exception as e:
                        print(f"{RED}Error: {e}{RESET}")
            
            # Save summary if requested
            if args.save_summary:
                thought_logger.save_summary(args.save_summary)
                print(f"{GREEN}Summary saved to: {args.save_summary}{RESET}")
        
        finally:
            # Always cleanup
            await agent.cleanup()
                    
    except Exception as e:
        print(f"{RED}Fatal error: {e}{RESET}")
        import traceback
        traceback.print_exc()


def main():
    """Main function with interactive and non-interactive modes."""
    parser = argparse.ArgumentParser(
        description="Weather Anomaly Investigation Agent with CAMEL AI + MCP Tools + Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Batch evaluation from text file with JSON array
  python agent.py --batch_file test_cases.txt --evaluate -mo results.json
  
  # Batch evaluation from newline-separated JSON
  python agent.py --batch_file test_cases.txt --evaluate --quiet
  
  # Single case evaluation
  python agent.py --file single_case.txt --evaluate
  
  # Interactive evaluation mode
  python agent.py --evaluate
        """
    )
    parser.add_argument(
        "--json", "-j", 
        help="JSON anomaly data to analyze"
    )
    parser.add_argument(
        "--file", "-f", 
        help="Text file containing JSON anomaly data"
    )
    parser.add_argument(
        "--batch_file", "-bf",
        help="Text file containing JSON array or newline-separated JSON objects for batch evaluation"
    )
    parser.add_argument(
        "--evaluate", "-e",
        action="store_true",
        help="Enable evaluation mode (disables MQTT, requires 'true_anomaly' field)"
    )
    parser.add_argument(
        "--metrics_output", "-mo",
        default="evaluation/metrics_results.json",
        help="Output file for evaluation metrics (default: evaluation/metrics_results.json)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output during batch processing"
    )
    parser.add_argument(
        "--mcp_config", "-c",
        default="config/weather_anomaly.json",
        help="Path to MCP config file (default: config/weather_anomaly.json)"
    )
    parser.add_argument(
        "--model_platform", "-p",
        default="openai",
        choices=["gemini", "openai", "anthropic", "ollama"],
        help="Model platform (default: openai)"
    )
    parser.add_argument(
        "--model_type", "-m",
        default="gpt-4o-mini",
        help="Model type (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.0,
        help="Temperature for LLM generation"
    )
    parser.add_argument(
        "--log_file", "-l",
        default="logs/thought_process.log",
        help="Path to log file for thought process (default: logs/thought_process.log)"
    )
    parser.add_argument(
        "--save_summary", "-ss",
        help="Save thought process summary to JSON file"
    )
    
    args = parser.parse_args()
    
    # Run async function
    asyncio.run(run_agent(args))


if __name__ == "__main__":
    main()