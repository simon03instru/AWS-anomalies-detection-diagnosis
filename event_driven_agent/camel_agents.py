"""
CAMEL AI: Communication between Reporting Agent and Diagnosis Agent
Requirements: pip install camel-ai[all] openai
"""

from dotenv import load_dotenv
load_dotenv()  # Load .env file

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.types import RoleType, ModelType
from camel.configs import ChatGPTConfig
from kafka import KafkaProducer, KafkaConsumer
from kafka.admin import KafkaAdminClient, NewTopic
import json
import threading
import time
from datetime import datetime
from typing import Dict, Any
import os


class EventBus:
    """Kafka-based event bus for agent communication"""
    
    def __init__(self, bootstrap_servers='localhost:9092'):
        self.bootstrap_servers = bootstrap_servers
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None
        )
    
    def publish_event(self, topic: str, event: Dict[str, Any], key: str = None):
        """Publish an event to Kafka topic"""
        self.producer.send(topic, key=key, value=event)
        self.producer.flush()
        print(f"📤 Event published to '{topic}': {event.get('event_type', 'unknown')}")
    
    def create_topics(self, topics):
        """Create Kafka topics"""
        try:
            admin = KafkaAdminClient(bootstrap_servers=self.bootstrap_servers)
            topic_list = [NewTopic(name=topic, num_partitions=1, replication_factor=1) 
                         for topic in topics]
            admin.create_topics(new_topics=topic_list, validate_only=False)
            print(f"✅ Topics created: {topics}")
        except Exception as e:
            print(f"ℹ️  Topics may already exist: {e}")


class BaseEventDrivenAgent:
    """Base class for event-driven CAMEL agents"""
    
    def __init__(self, 
                 agent_name: str, 
                 event_bus: EventBus,
                 subscribe_topics: list = None,
                 publish_topic: str = None,
                 model_type=ModelType.GPT_4O_MINI):
        
        self.agent_name = agent_name
        self.event_bus = event_bus
        self.subscribe_topics = subscribe_topics or []
        self.publish_topic = publish_topic
        self.running = False
        self.consumer_thread = None
        
        # Initialize CAMEL agent
        self.camel_agent = None
        self.model_type = model_type
    
    def create_camel_agent(self, system_message: str):
        """Create CAMEL ChatAgent with system message"""
        sys_msg = BaseMessage.make_assistant_message(
            role_name=self.agent_name,
            content=system_message
        )
        
        # Updated for newer CAMEL AI API
        from camel.models import ModelFactory
        from camel.types import ModelPlatformType
        
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=self.model_type,
            model_config_dict={"temperature": 0.7, "max_tokens": 1500}
        )
        
        self.camel_agent = ChatAgent(
            system_message=sys_msg,
            model=model
        )
    
    def start(self):
        """Start listening to events"""
        if not self.subscribe_topics:
            print(f"⚠️  {self.agent_name} has no topics to subscribe to")
            return
        
        self.running = True
        self.consumer_thread = threading.Thread(target=self._consume_events)
        self.consumer_thread.daemon = True
        self.consumer_thread.start()
        print(f"🤖 {self.agent_name} started, listening to: {self.subscribe_topics}")
    
    def stop(self):
        """Stop the agent"""
        self.running = False
        if self.consumer_thread:
            self.consumer_thread.join(timeout=5)
    
    def _consume_events(self):
        """Consume events from Kafka"""
        consumer = KafkaConsumer(
            *self.subscribe_topics,
            bootstrap_servers=self.event_bus.bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id=f"{self.agent_name}_group",
            auto_offset_reset='latest',
            enable_auto_commit=True
        )
        
        print(f"👂 {self.agent_name} consumer ready")
        
        for message in consumer:
            if not self.running:
                break
            
            event = message.value
            print(f"\n📥 {self.agent_name} received event: {event.get('event_type')}")
            
            try:
                self.handle_event(event)
            except Exception as e:
                print(f"❌ Error handling event in {self.agent_name}: {e}")
    
    def handle_event(self, event: Dict[str, Any]):
        """Override this to handle incoming events"""
        pass
    
    def publish_event(self, event_type: str, data: Dict[str, Any], correlation_id: str = None):
        """Publish an event"""
        event = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "source": self.agent_name,
            "data": data,
            "correlation_id": correlation_id
        }
        
        if self.publish_topic:
            self.event_bus.publish_event(self.publish_topic, event, key=correlation_id)
        else:
            print(f"⚠️  {self.agent_name} has no publish topic configured")


class ReportingAgent(BaseEventDrivenAgent):
    """Agent that monitors system metrics and publishes reports as events"""
    
    def __init__(self, event_bus: EventBus):
        super().__init__(
            agent_name="ReportingAgent",
            event_bus=event_bus,
            subscribe_topics=['system-metrics'],
            publish_topic='system-reports'
        )
        
        # Create CAMEL agent with reporting capabilities
        system_message = """You are an intelligent system monitoring and reporting agent.

Your responsibilities:
1. Analyze incoming system metrics and logs
2. Identify patterns, trends, and anomalies
3. Generate comprehensive, structured reports
4. Highlight critical issues that need attention
5. Provide context about normal vs abnormal behavior

Output Format: Always respond with valid JSON containing:
- summary: Brief overview of system state
- metrics_analysis: Analysis of key metrics
- anomalies: List of detected anomalies
- severity: Overall severity level (low, medium, high, critical)
- concerns: Specific issues requiring attention

Be analytical, concise, and focus on actionable insights."""
        
        self.create_camel_agent(system_message)
        self.report_count = 0
    
    def handle_event(self, event: Dict[str, Any]):
        """Handle incoming system metrics events"""
        if event.get('event_type') == 'SYSTEM_METRICS_COLLECTED':
            self._generate_report(event)
    
    def _generate_report(self, metrics_event: Dict[str, Any]):
        """Use CAMEL agent to analyze metrics and generate report"""
        
        metrics_data = metrics_event.get('data', {})
        correlation_id = metrics_event.get('correlation_id', f"report-{self.report_count}")
        self.report_count += 1
        
        print(f"📊 Generating report for correlation_id: {correlation_id}")
        
        # Create prompt for CAMEL agent
        user_msg = BaseMessage.make_user_message(
            role_name="SystemMonitor",
            content=f"""Analyze these system metrics and generate a comprehensive report:

Server: {metrics_data.get('server_id', 'unknown')}
Timestamp: {metrics_data.get('timestamp', 'unknown')}

Metrics:
{json.dumps(metrics_data.get('metrics', {}), indent=2)}

Recent Logs:
{json.dumps(metrics_data.get('logs', []), indent=2)}

Provide your analysis as valid JSON only."""
        )
        
        # Get analysis from CAMEL agent
        response = self.camel_agent.step(user_msg)
        report_content = response.msg.content
        
        print(f"✅ Report generated by CAMEL agent")
        print(f"Preview: {report_content[:200]}...")
        
        # Try to parse as JSON, otherwise wrap it
        try:
            report_data = json.loads(report_content)
        except json.JSONDecodeError:
            report_data = {
                "raw_analysis": report_content,
                "metrics": metrics_data.get('metrics', {})
            }
        
        # Publish report event
        self.publish_event(
            event_type='SYSTEM_REPORT_GENERATED',
            data={
                'report': report_data,
                'source_metrics': metrics_data
            },
            correlation_id=correlation_id
        )


class DiagnosisAgent(BaseEventDrivenAgent):
    """Agent that diagnoses issues based on reports"""
    
    def __init__(self, event_bus: EventBus):
        super().__init__(
            agent_name="DiagnosisAgent",
            event_bus=event_bus,
            subscribe_topics=['system-reports'],
            publish_topic='diagnosis-results'
        )
        
        # Create CAMEL agent with diagnosis capabilities
        system_message = """You are an expert system diagnosis agent with deep knowledge of:
- System performance analysis
- Root cause identification
- Infrastructure troubleshooting
- Capacity planning
- Performance optimization

Your responsibilities:
1. Analyze system reports from the reporting agent
2. Diagnose root causes of issues
3. Assess severity and impact
4. Provide specific, actionable recommendations
5. Suggest preventive measures

Output Format: Always respond with valid JSON containing:
- diagnosis: Clear diagnosis of the issue
- root_cause: Identified root cause
- severity: critical/high/medium/low
- impact: Description of business impact
- recommendations: List of specific actions to take
- preventive_measures: Steps to prevent recurrence

Be technical, precise, and actionable."""
        
        self.create_camel_agent(system_message)
        self.diagnosis_count = 0
    
    def handle_event(self, event: Dict[str, Any]):
        """Handle incoming report events"""
        if event.get('event_type') == 'SYSTEM_REPORT_GENERATED':
            self._diagnose_report(event)
    
    def _diagnose_report(self, report_event: Dict[str, Any]):
        """Use CAMEL agent to diagnose issues from report"""
        
        report_data = report_event.get('data', {})
        correlation_id = report_event.get('correlation_id', f"diagnosis-{self.diagnosis_count}")
        self.diagnosis_count += 1
        
        print(f"🔍 Diagnosing report for correlation_id: {correlation_id}")
        
        # Create prompt for CAMEL agent
        user_msg = BaseMessage.make_user_message(
            role_name="ReportingAgent",
            content=f"""Analyze this system report and provide a detailed diagnosis:

Report:
{json.dumps(report_data.get('report', {}), indent=2)}

Source Metrics:
{json.dumps(report_data.get('source_metrics', {}), indent=2)}

Provide your diagnosis as valid JSON only."""
        )
        
        # Get diagnosis from CAMEL agent
        response = self.camel_agent.step(user_msg)
        diagnosis_content = response.msg.content
        
        print(f"✅ Diagnosis completed by CAMEL agent")
        print(f"Preview: {diagnosis_content[:200]}...")
        
        # Try to parse as JSON
        try:
            diagnosis_data = json.loads(diagnosis_content)
        except json.JSONDecodeError:
            diagnosis_data = {
                "raw_diagnosis": diagnosis_content
            }
        
        # Publish diagnosis event
        self.publish_event(
            event_type='DIAGNOSIS_COMPLETED',
            data={
                'diagnosis': diagnosis_data,
                'original_report': report_data
            },
            correlation_id=correlation_id
        )


class AlertingAgent(BaseEventDrivenAgent):
    """Agent that sends alerts based on diagnosis"""
    
    def __init__(self, event_bus: EventBus):
        super().__init__(
            agent_name="AlertingAgent",
            event_bus=event_bus,
            subscribe_topics=['diagnosis-results'],
            publish_topic='alerts'
        )
    
    def handle_event(self, event: Dict[str, Any]):
        """Handle diagnosis results and send alerts"""
        if event.get('event_type') == 'DIAGNOSIS_COMPLETED':
            self._send_alert(event)
    
    def _send_alert(self, diagnosis_event: Dict[str, Any]):
        """Send alert based on diagnosis severity"""
        
        diagnosis_data = diagnosis_event.get('data', {}).get('diagnosis', {})
        severity = diagnosis_data.get('severity', 'unknown')
        correlation_id = diagnosis_event.get('correlation_id')
        
        print(f"🚨 ALERT - Severity: {severity.upper()}")
        print(f"   Correlation ID: {correlation_id}")
        print(f"   Diagnosis: {diagnosis_data.get('diagnosis', 'No diagnosis available')[:100]}")
        
        if severity in ['critical', 'high']:
            print(f"   ⚠️  URGENT: Immediate action required!")
        
        # Publish alert event
        self.publish_event(
            event_type='ALERT_SENT',
            data={
                'severity': severity,
                'message': diagnosis_data.get('diagnosis', ''),
                'recommendations': diagnosis_data.get('recommendations', [])
            },
            correlation_id=correlation_id
        )


class SystemMonitor:
    """Simulates system monitoring and publishes metric events"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
    
    def collect_and_publish_metrics(self, scenario_data: Dict):
        """Collect system metrics and publish as event"""
        
        correlation_id = f"monitor-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        event = {
            "event_type": "SYSTEM_METRICS_COLLECTED",
            "timestamp": datetime.now().isoformat(),
            "source": "SystemMonitor",
            "correlation_id": correlation_id,
            "data": scenario_data
        }
        
        print(f"\n{'='*70}")
        print(f"📡 System Monitor collecting metrics...")
        print(f"   Server: {scenario_data.get('server_id')}")
        print(f"   Correlation ID: {correlation_id}")
        print(f"{'='*70}")
        
        self.event_bus.publish_event('system-metrics', event, key=correlation_id)


# ==================== DEMO SCENARIOS ====================

def main():
    """Run event-driven multi-agent system"""
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  WARNING: OPENAI_API_KEY not set!")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    print("\n" + "🤖 "*30)
    print("CAMEL AI Event-Driven Multi-Agent System")
    print("Event Bus: Apache Kafka")
    print("🤖 "*30 + "\n")
    
    # Initialize event bus
    event_bus = EventBus(bootstrap_servers='localhost:9092')
    
    # Create topics
    topics = ['system-metrics', 'system-reports', 'diagnosis-results', 'alerts']
    event_bus.create_topics(topics)
    
    print("\n" + "="*70)
    print("Initializing Agents...")
    print("="*70)
    
    # Create agents
    reporting_agent = ReportingAgent(event_bus)
    diagnosis_agent = DiagnosisAgent(event_bus)
    alerting_agent = AlertingAgent(event_bus)
    system_monitor = SystemMonitor(event_bus)
    
    # Start agents (they will listen to events)
    reporting_agent.start()
    time.sleep(1)
    diagnosis_agent.start()
    time.sleep(1)
    alerting_agent.start()
    
    # Wait for consumers to be ready
    print("\n⏳ Waiting for agents to initialize...")
    time.sleep(3)
    
    # Scenario 1: Critical CPU issue
    print("\n" + "🎬 "*30)
    print("SCENARIO 1: Critical CPU & Memory Issue")
    print("🎬 "*30)
    
    scenario_1 = {
        "server_id": "prod-web-01",
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "cpu_percent": 98.5,
            "memory_percent": 94.2,
            "disk_usage": 67.3,
            "network_in_mbps": 234.5,
            "response_time_ms": 4520,
            "error_rate": 12.3,
            "active_connections": 2847
        },
        "logs": [
            "ERROR: Out of memory exception in application",
            "CRITICAL: CPU throttling detected",
            "ERROR: Request timeout - 5000ms exceeded",
            "WARNING: Connection pool exhausted"
        ]
    }
    
    system_monitor.collect_and_publish_metrics(scenario_1)
    
    print("\n⏳ Processing events... (this may take 10-15 seconds)")
    time.sleep(15)
    
    # Scenario 2: Disk space warning
    print("\n" + "🎬 "*30)
    print("SCENARIO 2: Disk Space Warning")
    print("🎬 "*30)
    
    scenario_2 = {
        "server_id": "prod-db-01",
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "cpu_percent": 45.2,
            "memory_percent": 68.5,
            "disk_usage": 91.7,
            "disk_io_wait": 23.4,
            "database_connections": 450,
            "query_latency_ms": 890
        },
        "logs": [
            "WARNING: Disk usage above 90%",
            "INFO: Auto-vacuum initiated",
            "WARNING: Slow query detected: 3200ms"
        ]
    }
    
    system_monitor.collect_and_publish_metrics(scenario_2)
    
    print("\n⏳ Processing events...")
    time.sleep(15)
    
    print("\n" + "✅ "*30)
    print("Event-Driven Workflow Completed!")
    print("All agents communicated via Kafka events")
    print("✅ "*30)
    
    # Stop agents
    print("\n🛑 Stopping agents...")
    reporting_agent.stop()
    diagnosis_agent.stop()
    alerting_agent.stop()
    
    print("\n✨ Demo completed successfully!")


if __name__ == "__main__":
    main()