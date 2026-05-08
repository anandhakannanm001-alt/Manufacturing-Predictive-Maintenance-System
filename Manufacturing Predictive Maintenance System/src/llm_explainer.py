"""
LLM-powered report generation for predictive maintenance.
Generates human-readable explanations and maintenance recommendations.
"""

import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Any
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MaintenanceReportGenerator:
    """
    Generate detailed maintenance reports with technical explanations.
    Uses rule-based templates to create human-readable reports.
    """
    
    def __init__(self):
        """Initialize report generator."""
        self.failure_mode_descriptions = {
            'TWF': 'Tool Wear Failure - Caused by excessive tool wear beyond acceptable limits',
            'HDF': 'Heat Dissipation Failure - Related to temperature regulation issues',
            'PWF': 'Power Failure - Associated with abnormal power consumption or torque',
            'OSF': 'Overstrain Failure - Caused by mechanical stress and overload',
            'RNF': 'Random Failure - Unpredictable failure not directly linked to specific causes'
        }
        
        self.maintenance_actions = {
            'critical': [
                'Immediate machine shutdown recommended',
                'Inspect tool wear and replace if necessary',
                'Check temperature sensors and cooling systems',
                'Review power consumption patterns',
                'Schedule emergency maintenance team'
            ],
            'high': [
                'Schedule maintenance within 12 hours',
                'Monitor sensor readings continuously',
                'Prepare replacement parts',
                'Alert maintenance team'
            ],
            'medium': [
                'Schedule planned maintenance within 3-5 days',
                'Increase monitoring frequency',
                'Order replacement parts if needed',
                'Document sensor trends'
            ],
            'low': [
                'Continue normal operations',
                'Maintain regular monitoring schedule',
                'Log current readings for trend analysis',
                'Schedule next routine inspection'
            ]
        }
    
    def generate_single_report(self, prediction_result: Dict, 
                              sensor_data: Dict) -> str:
        """
        Generate detailed report for single machine prediction.
        
        Args:
            prediction_result: Prediction output from MaintenancePredictor
            sensor_data: Original sensor readings
            
        Returns:
            Formatted report string
        """
        report_lines = []
        
        # Header
        report_lines.append("="*70)
        report_lines.append("PREDICTIVE MAINTENANCE REPORT")
        report_lines.append("="*70)
        report_lines.append(f"Machine ID: {prediction_result['machine_id']}")
        report_lines.append(f"Report Generated: {prediction_result['timestamp']}")
        report_lines.append("")
        
        # Risk Assessment
        report_lines.append("RISK ASSESSMENT")
        report_lines.append("-" * 70)
        report_lines.append(f"Risk Level: {prediction_result['risk_level']}")
        report_lines.append(f"Failure Probability: {prediction_result['failure_probability']:.2%}")
        report_lines.append(f"Health Score: {prediction_result['health_score']:.2f}/1.00")
        report_lines.append(f"Prediction Confidence: {prediction_result['confidence']:.2%}")
        report_lines.append("")
        
        # Recommendation
        report_lines.append("RECOMMENDATION")
        report_lines.append("-" * 70)
        report_lines.append(prediction_result['recommendation'])
        report_lines.append("")
        
        # Sensor Analysis
        report_lines.append("SENSOR ANALYSIS")
        report_lines.append("-" * 70)
        report_lines.append(self._analyze_sensors(sensor_data))
        report_lines.append("")
        
        # Estimated Remaining Useful Life
        report_lines.append("REMAINING USEFUL LIFE (RUL)")
        report_lines.append("-" * 70)
        rul = prediction_result['estimated_hours_to_failure']
        if rul > 0:
            report_lines.append(f"Estimated Hours to Failure: {rul:.1f} hours")
            report_lines.append(f"Estimated Days to Failure: {rul/24:.1f} days")
        else:
            report_lines.append("IMMEDIATE FAILURE RISK - Machine should be shut down")
        report_lines.append("")
        
        # Cost-Benefit Analysis
        report_lines.append("COST-BENEFIT ANALYSIS")
        report_lines.append("-" * 70)
        cost_impact = prediction_result['cost_impact']
        report_lines.append(f"Expected Failure Cost: ${cost_impact['expected_failure_cost']:,.2f}")
        report_lines.append(f"Planned Maintenance Cost: ${cost_impact['maintenance_cost']:,.2f}")
        report_lines.append(f"Expected Cost without Action: ${cost_impact['expected_cost_no_action']:,.2f}")
        report_lines.append(f"Net Cost Savings: ${cost_impact['net_cost_savings']:,.2f}")
        report_lines.append(f"ROI: {cost_impact['roi_percentage']:.1f}%")
        report_lines.append("")
        
        # Recommended Actions
        report_lines.append("RECOMMENDED ACTIONS")
        report_lines.append("-" * 70)
        risk_level_lower = prediction_result['risk_level'].lower()
        actions = self.maintenance_actions.get(risk_level_lower, [])
        for i, action in enumerate(actions, 1):
            report_lines.append(f"{i}. {action}")
        report_lines.append("")
        
        # Footer
        report_lines.append("="*70)
        report_lines.append("END OF REPORT")
        report_lines.append("="*70)
        
        return "\n".join(report_lines)
    
    def generate_batch_report(self, batch_result: Dict) -> str:
        """
        Generate summary report for batch predictions.
        
        Args:
            batch_result: Batch prediction output from MaintenancePredictor
            
        Returns:
            Formatted batch report string
        """
        report_lines = []
        
        # Header
        report_lines.append("="*70)
        report_lines.append("BATCH PREDICTIVE MAINTENANCE REPORT")
        report_lines.append("="*70)
        report_lines.append(f"Report Generated: {batch_result['timestamp']}")
        report_lines.append("")
        
        # Summary Statistics
        report_lines.append("FLEET SUMMARY")
        report_lines.append("-" * 70)
        report_lines.append(f"Total Machines Analyzed: {batch_result['total_machines']}")
        report_lines.append(f"Average Failure Probability: {batch_result['average_failure_probability']:.2%}")
        report_lines.append("")
        
        # Risk Distribution
        report_lines.append("RISK DISTRIBUTION")
        report_lines.append("-" * 70)
        total = batch_result['total_machines']
        report_lines.append(f"Critical Risk: {batch_result['critical_count']} ({batch_result['critical_count']/total*100:.1f}%)")
        report_lines.append(f"High Risk: {batch_result['high_risk_count']} ({batch_result['high_risk_count']/total*100:.1f}%)")
        report_lines.append(f"Medium Risk: {batch_result['medium_risk_count']} ({batch_result['medium_risk_count']/total*100:.1f}%)")
        report_lines.append(f"Low Risk: {batch_result['low_risk_count']} ({batch_result['low_risk_count']/total*100:.1f}%)")
        report_lines.append("")
        
        # Individual Machine Results
        report_lines.append("INDIVIDUAL MACHINE RESULTS")
        report_lines.append("-" * 70)
        for pred in batch_result['predictions']:
            report_lines.append(f"\nMachine: {pred['machine_id']}")
            report_lines.append(f"  Risk Level: {pred['risk_level']}")
            report_lines.append(f"  Failure Probability: {pred['failure_probability']:.2%}")
            report_lines.append(f"  Recommendation: {pred['recommendation']}")
            report_lines.append(f"  Estimated RUL: {pred['estimated_hours_to_failure']:.1f} hours")
        report_lines.append("")
        
        # Fleet-Level Recommendations
        report_lines.append("FLEET-LEVEL RECOMMENDATIONS")
        report_lines.append("-" * 70)
        if batch_result['critical_count'] > 0:
            report_lines.append(f"⚠️  IMMEDIATE ACTION: {batch_result['critical_count']} machine(s) require immediate attention")
        if batch_result['high_risk_count'] > 0:
            report_lines.append(f"⚠️  URGENT: {batch_result['high_risk_count']} machine(s) need maintenance within 12 hours")
        if batch_result['medium_risk_count'] > 0:
            report_lines.append(f"📋 PLANNED: {batch_result['medium_risk_count']} machine(s) should be scheduled for maintenance")
        if batch_result['low_risk_count'] > 0:
            report_lines.append(f"✅ MONITOR: {batch_result['low_risk_count']} machine(s) operating normally")
        report_lines.append("")
        
        # Footer
        report_lines.append("="*70)
        report_lines.append("END OF REPORT")
        report_lines.append("="*70)
        
        return "\n".join(report_lines)
    
    def _analyze_sensors(self, sensor_data: Dict) -> str:
        """Analyze sensor readings and provide insights."""
        analysis_lines = []
        
        # Temperature analysis
        if 'air_temperature_k' in sensor_data and 'process_temperature_k' in sensor_data:
            air_temp = sensor_data['air_temperature_k']
            proc_temp = sensor_data['process_temperature_k']
            temp_diff = proc_temp - air_temp
            
            analysis_lines.append(f"Air Temperature: {air_temp:.1f} K")
            analysis_lines.append(f"Process Temperature: {proc_temp:.1f} K")
            analysis_lines.append(f"Temperature Difference: {temp_diff:.1f} K")
            
            if temp_diff > 15:
                analysis_lines.append("⚠️  High temperature differential detected")
            elif temp_diff > 10:
                analysis_lines.append("⚠️  Elevated temperature differential")
            else:
                analysis_lines.append("✓ Temperature differential within normal range")
        
        # Rotational speed analysis
        if 'rotational_speed_rpm' in sensor_data:
            rpm = sensor_data['rotational_speed_rpm']
            analysis_lines.append(f"Rotational Speed: {rpm} RPM")
            
            if rpm < 1300:
                analysis_lines.append("⚠️  Low rotational speed detected")
            elif rpm > 2700:
                analysis_lines.append("⚠️  High rotational speed detected")
            else:
                analysis_lines.append("✓ Rotational speed within normal range")
        
        # Torque analysis
        if 'torque_nm' in sensor_data:
            torque = sensor_data['torque_nm']
            analysis_lines.append(f"Torque: {torque:.1f} Nm")
            
            if torque < 10:
                analysis_lines.append("⚠️  Low torque detected")
            elif torque > 65:
                analysis_lines.append("⚠️  High torque detected")
            else:
                analysis_lines.append("✓ Torque within normal range")
        
        # Tool wear analysis
        if 'tool_wear_min' in sensor_data:
            wear = sensor_data['tool_wear_min']
            analysis_lines.append(f"Tool Wear: {wear} minutes")
            
            if wear > 200:
                analysis_lines.append("⚠️  CRITICAL: Tool wear exceeds safe limits")
            elif wear > 150:
                analysis_lines.append("⚠️  High tool wear - replacement recommended")
            elif wear > 100:
                analysis_lines.append("⚠️  Moderate tool wear - monitor closely")
            else:
                analysis_lines.append("✓ Tool wear within acceptable range")
        
        return "\n".join(analysis_lines)
    
    def generate_technical_explanation(self, prediction_result: Dict,
                                      feature_importance: pd.DataFrame = None) -> str:
        """
        Generate technical explanation of prediction.
        
        Args:
            prediction_result: Prediction output
            feature_importance: DataFrame with feature importance scores
            
        Returns:
            Technical explanation string
        """
        explanation_lines = []
        
        explanation_lines.append("TECHNICAL EXPLANATION")
        explanation_lines.append("-" * 70)
        explanation_lines.append("")
        
        # Prediction basis
        explanation_lines.append("Prediction Basis:")
        explanation_lines.append(f"The model predicts a {prediction_result['risk_level'].lower()} risk level")
        explanation_lines.append(f"based on a {prediction_result['failure_probability']:.2%} failure probability.")
        explanation_lines.append("")
        
        # Key contributing factors
        if feature_importance is not None and len(feature_importance) > 0:
            explanation_lines.append("Key Contributing Factors:")
            explanation_lines.append("Based on feature importance analysis:")
            
            top_features = feature_importance.head(5)
            for _, row in top_features.iterrows():
                explanation_lines.append(f"  - {row['feature']}: {row['importance']:.3f}")
            explanation_lines.append("")
        
        # Model confidence
        explanation_lines.append("Model Confidence:")
        confidence = prediction_result['confidence']
        if confidence > 0.8:
            explanation_lines.append("  HIGH: The model is highly confident in this prediction")
        elif confidence > 0.6:
            explanation_lines.append("  MODERATE: The model has reasonable confidence")
        else:
            explanation_lines.append("  LOW: The prediction has lower confidence, consider additional monitoring")
        explanation_lines.append("")
        
        # Limitations
        explanation_lines.append("Limitations:")
        explanation_lines.append("  - Predictions are based on historical data patterns")
        explanation_lines.append("  - External factors not captured by sensors may affect actual failure risk")
        explanation_lines.append("  - Regular model retraining is recommended to maintain accuracy")
        
        return "\n".join(explanation_lines)


# Example usage
if __name__ == "__main__":
    generator = MaintenanceReportGenerator()
    
    # Sample prediction result
    sample_prediction = {
        'machine_id': 'M001',
        'failure_probability': 0.65,
        'risk_level': 'High',
        'recommendation': 'URGENT - Schedule maintenance within 12 hours',
        'health_score': 0.35,
        'confidence': 0.65,
        'estimated_hours_to_failure': 150.5,
        'cost_impact': {
            'expected_failure_cost': 45000.0,
            'maintenance_cost': 2000.0,
            'expected_cost_no_action': 29250.0,
            'net_cost_savings': 27250.0,
            'roi_percentage': 1362.5
        },
        'timestamp': datetime.now().isoformat()
    }
    
    # Sample sensor data
    sample_sensors = {
        'machine_id': 'M001',
        'type_encoded': 1,
        'air_temperature_k': 298.0,
        'process_temperature_k': 308.0,
        'rotational_speed_rpm': 1500,
        'torque_nm': 40.0,
        'tool_wear_min': 180
    }
    
    # Generate report
    report = generator.generate_single_report(sample_prediction, sample_sensors)
    print(report)
