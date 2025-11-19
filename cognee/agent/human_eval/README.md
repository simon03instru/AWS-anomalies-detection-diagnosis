# AWS Diagnosis System Evaluation Framework

Evaluation Form : https://docs.google.com/forms/d/e/1FAIpQLSchy_KjY9p35ZUZLqacEPB8ulbdMLxn3KEODjNyjC2NBeHHpg/viewform?usp=dialog

## Overview

This evaluation framework is designed to assess the performance of an Automatic Weather Station (AWS) multi-agent diagnosis system. The system consists of two main components working together to detect and diagnose sensor anomalies:

1. **Station Agent**: Detects anomalies in weather sensor readings by analyzing real-time data streams and identifying deviations from expected patterns
2. **Diagnosis Agent**: Provides root cause analysis and recommends specific troubleshooting actions based on detected anomalies

The evaluation framework uses a structured Likert-scale approach adapted from medical diagnosis assessment methodologies to objectively measure the system's diagnostic accuracy and the practical utility of its recommendations.


## System Architecture

┌─────────────────┐
│  Weather Data   │
│   (Real-time)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Station Agent  │  ← Anomaly Detection & Trend Analysis
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Diagnosis Agent │  ← Root Cause Analysis & Recommendations
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ System Output   │  ← Combined Assessment
└─────────────────┘



## Evaluation Methodology

The evaluation is performed by domain experts (weather station technicians) who assess the complete system output against known or verified sensor faults. The framework consists of two primary evaluation metrics:

### Metric 1: Diagnosis Quality
Evaluates how accurately the system identifies the actual sensor or equipment fault.

### Metric 2: Troubleshooting Action Quality
Evaluates the effectiveness and practicality of the recommended troubleshooting steps.



## Evaluation Metrics

### Table 1: Evaluation Metric for AWS Diagnosis Quality

| Criteria | Score |
|----------|-------|
| The diagnosis correctly identified the actual sensor/equipment fault | 5 |
| The diagnosis identified a very similar fault or closely related cause | 4 |
| The diagnosis identified a related issue that could contribute to the problem | 3 |
| The diagnosis identified a tangentially related issue but unlikely to be the root cause | 2 |
| The diagnosis missed the actual fault or identified an unrelated issue | 0 |

**Scoring Guidelines:**
- **Score 5**: Perfect match - the system pinpointed the exact fault (e.g., "Young 05103-L anemometer mechanical failure")
- **Score 4**: Very close - identified the correct component and fault type but missed minor details
- **Score 3**: Partially correct - identified a contributing factor or related issue
- **Score 2**: Loosely related - mentioned something connected but not the primary cause
- **Score 0**: Missed or wrong - completely failed to identify the actual problem



### Table 2: Evaluation Metric for Recommended Troubleshooting Actions

| Criteria | Score |
|----------|-------|
| Strongly agree that the recommended actions would effectively resolve the issue | 5 |
| Agree that the recommended actions would likely resolve the issue | 4 |
| Neutral - the recommended actions may or may not help | 3 |
| Disagree that the recommended actions would resolve the issue | 2 |
| Strongly disagree that the recommended actions would help | 1 |

**Scoring Guidelines:**
- **Score 5**: Excellent recommendations - comprehensive, actionable, and will definitely solve the problem
- **Score 4**: Good recommendations - practical steps that should resolve the issue
- **Score 3**: Acceptable recommendations - some useful steps but incomplete or unclear effectiveness
- **Score 2**: Poor recommendations - unlikely to help or missing critical steps
- **Score 1**: Ineffective recommendations - wrong approach or could worsen the situation
