"""
Prediction interface for predictive maintenance system.
Provides single and batch prediction capabilities with risk assessment.
"""

import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MaintenancePredictor:
    """
    Predictive maintenance predictor with risk assessment and recommendations.
    """
    
    def __init__(self, model_path: str = "models/random_forest.pkl"):
        """
        Initialize predictor with trained model.
        
        Args:
            model_path: Path to trained model file
        """
        self.model = None
        self.feature_columns = None
        self.model_path = model_path
        self.load_model()
        
        # Risk level thresholds
        self.risk_thresholds = {
            'critical': 0.85,
            'high': 0.65,
            'medium': 0.45,
            'low': 0.0
        }
        
        # Cost parameters (can be customized)
        self.cost_params = {
            'expected_failure_cost': 45000.0,  # Cost of unplanned failure
            'maintenance_cost': 2000.0,      # Cost of planned maintenance
            'downtime_cost_per_hour': 125.0   # Cost of downtime per hour
        }
    
    def load_model(self) -> None:
        """Load trained model and feature columns."""
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
            
            # Try to load feature columns
            feature_columns_path = Path(self.model_path).parent / "feature_columns.json"
            if feature_columns_path.exists():
                with open(feature_columns_path, 'r') as f:
                    self.feature_columns = json.load(f)
                logger.info(f"Feature columns loaded: {len(self.feature_columns)} features")
        except FileNotFoundError:
            logger.error(f"Model file not found: {self.model_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_input(self, input_data: Dict) -> pd.DataFrame:
        """
        Preprocess single input for prediction.
        
        Args:
            input_data: Dictionary with sensor readings
            
        Returns:
            DataFrame with engineered features
        """
        from src.feature_engineering import FeatureEngineer
        
        # Create DataFrame from input
        df = pd.DataFrame([input_data])
        
        # Apply feature engineering
        engineer = FeatureEngineer()
        df_engineered = engineer.full_pipeline(df)
        
        return df_engineered
    
    def predict_single(self, input_data: Dict) -> Dict:
        """
        Make single machine prediction with risk assessment.
        
        Args:
            input_data: Dictionary with machine sensor readings
            
        Returns:
            Dictionary with prediction, risk level, and recommendations
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Preprocess input
        df_engineered = self.preprocess_input(input_data)
        
        # Get feature columns (excluding non-feature columns)
        exclude_cols = ['udi', 'product_id', 'machine_failure', 'type', 
                       'twf', 'hdf', 'pwf', 'osf', 'rnf', 'health_grade',
                       'noise_flags']
        feature_cols = [col for col in df_engineered.columns 
                       if col not in exclude_cols]
        
        # Ensure feature columns match training data
        if self.feature_columns:
            # Align features with training data
            for col in self.feature_columns:
                if col not in df_engineered.columns:
                    df_engineered[col] = 0  # Default value for missing features
            
            X = df_engineered[self.feature_columns]
        else:
            X = df_engineered[feature_cols]
        
        # Make prediction
        failure_proba = self.model.predict_proba(X)[0, 1]
        failure_pred = int(failure_proba > 0.5)
        
        # Determine risk level
        risk_level = self._classify_risk(failure_proba)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(risk_level)
        
        # Calculate health score
        health_score = 1.0 - failure_proba
        
        # Estimate remaining useful life (RUL)
        rul_hours = self._estimate_rul(failure_proba, input_data)
        
        # Calculate cost impact
        cost_impact = self._calculate_cost_impact(failure_proba, rul_hours)
        
        # Build response
        result = {
            'machine_id': input_data.get('machine_id', 'Unknown'),
            'failure_probability': float(failure_proba),
            'failure_prediction': failure_pred,
            'risk_level': risk_level,
            'recommendation': recommendation,
            'health_score': float(health_score),
            'confidence': float(0.5 + abs(failure_proba - 0.5)),  # Higher when confident
            'estimated_hours_to_failure': rul_hours,
            'cost_impact': cost_impact,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def predict_batch(self, input_list: List[Dict]) -> Dict:
        """
        Make batch predictions for multiple machines.
        
        Args:
            input_list: List of dictionaries with machine data
            
        Returns:
            Dictionary with batch results and summary
        """
        predictions = []
        risk_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        total_failure_prob = 0.0
        
        for machine_data in input_list:
            pred = self.predict_single(machine_data)
            predictions.append(pred)
            
            # Count risk levels
            risk_counts[pred['risk_level']] += 1
            total_failure_prob += pred['failure_probability']
        
        # Calculate summary statistics
        avg_failure_prob = total_failure_prob / len(predictions) if predictions else 0
        
        # Generate batch summary
        summary = self._generate_batch_summary(
            total_machines=len(predictions),
            risk_counts=risk_counts,
            avg_failure_prob=avg_failure_prob
        )
        
        return {
            'total_machines': len(predictions),
            'critical_count': risk_counts['critical'],
            'high_risk_count': risk_counts['high'],
            'medium_risk_count': risk_counts['medium'],
            'low_risk_count': risk_counts['low'],
            'average_failure_probability': avg_failure_prob,
            'predictions': predictions,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        }
    
    def _classify_risk(self, failure_proba: float) -> str:
        """Classify risk level based on failure probability."""
        if failure_proba >= self.risk_thresholds['critical']:
            return 'Critical'
        elif failure_proba >= self.risk_thresholds['high']:
            return 'High'
        elif failure_proba >= self.risk_thresholds['medium']:
            return 'Medium'
        else:
            return 'Low'
    
    def _generate_recommendation(self, risk_level: str) -> str:
        """Generate maintenance recommendation based on risk level."""
        recommendations = {
            'Critical': 'IMMEDIATE - Schedule maintenance immediately',
            'High': 'URGENT - Schedule maintenance within 12 hours',
            'Medium': 'PLANNED - Schedule maintenance within 3-5 days',
            'Low': 'MONITOR - Continue normal operations, monitor sensors'
        }
        return recommendations.get(risk_level, 'MONITOR')
    
    def _estimate_rul(self, failure_proba: float, input_data: Dict) -> float:
        """
        Estimate remaining useful life in hours.
        Simple heuristic based on failure probability and tool wear.
        """
        tool_wear = input_data.get('tool_wear_min', 0)
        max_wear = 300  # Maximum tool wear in minutes
        
        # Base RUL estimate (inverse of failure probability)
        base_rul = (1.0 - failure_proba) * 500  # Max 500 hours
        
        # Adjust based on tool wear
        wear_factor = 1.0 - (tool_wear / max_wear)
        
        rul = base_rul * wear_factor
        return max(0.0, rul)
    
    def _calculate_cost_impact(self, failure_proba: float, rul_hours: float) -> Dict:
        """Calculate cost-benefit analysis for maintenance decision."""
        expected_failure_cost = self.cost_params['expected_failure_cost']
        maintenance_cost = self.cost_params['maintenance_cost']
        
        # Expected cost if no action taken
        expected_cost_no_action = failure_proba * expected_failure_cost
        
        # Cost if maintenance is performed now
        cost_with_maintenance = maintenance_cost
        
        # Net savings from performing maintenance
        net_savings = expected_cost_no_action - cost_with_maintenance
        
        # ROI percentage
        roi = (net_savings / maintenance_cost * 100) if maintenance_cost > 0 else 0
        
        return {
            'expected_failure_cost': expected_failure_cost,
            'maintenance_cost': maintenance_cost,
            'expected_cost_no_action': expected_cost_no_action,
            'net_cost_savings': max(0.0, net_savings),
            'roi_percentage': max(0.0, roi)
        }
    
    def _generate_batch_summary(self, total_machines: int, 
                               risk_counts: Dict,
                               avg_failure_prob: float) -> str:
        """Generate summary text for batch prediction."""
        summary_lines = [
            f"Batch Analysis Summary for {total_machines} machines:",
            f"- Critical Risk: {risk_counts['critical']} ({risk_counts['critical']/total_machines*100:.1f}%)",
            f"- High Risk: {risk_counts['high']} ({risk_counts['high']/total_machines*100:.1f}%)",
            f"- Medium Risk: {risk_counts['medium']} ({risk_counts['medium']/total_machines*100:.1f}%)",
            f"- Low Risk: {risk_counts['low']} ({risk_counts['low']/total_machines*100:.1f}%)",
            f"- Average Failure Probability: {avg_failure_prob:.2%}"
        ]
        
        # Add action recommendations
        if risk_counts['critical'] > 0:
            summary_lines.append(f"\n⚠️  ACTION REQUIRED: {risk_counts['critical']} machine(s) require IMMEDIATE attention")
        elif risk_counts['high'] > 0:
            summary_lines.append(f"\n⚠️  URGENT: {risk_counts['high']} machine(s) require attention within 12 hours")
        
        return "\n".join(summary_lines)


# Example usage
if __name__ == "__main__":
    predictor = MaintenancePredictor(model_path="models/random_forest.pkl")
    
    # Single prediction example
    sample_input = {
        "machine_id": "M001",
        "type_encoded": 1,
        "air_temperature_k": 298.0,
        "process_temperature_k": 308.0,
        "rotational_speed_rpm": 1500,
        "torque_nm": 40.0,
        "tool_wear_min": 100
    }
    
    result = predictor.predict_single(sample_input)
    print("Single Prediction Result:")
    print(json.dumps(result, indent=2))
    
    # Batch prediction example
    batch_input = [
        {
            "machine_id": "M001",
            "type_encoded": 1,
            "air_temperature_k": 298.0,
            "process_temperature_k": 308.0,
            "rotational_speed_rpm": 1500,
            "torque_nm": 40.0,
            "tool_wear_min": 100
        },
        {
            "machine_id": "M002",
            "type_encoded": 0,
            "air_temperature_k": 300.0,
            "process_temperature_k": 310.0,
            "rotational_speed_rpm": 1800,
            "torque_nm": 50.0,
            "tool_wear_min": 250
        }
    ]
    
    batch_result = predictor.predict_batch(batch_input)
    print("\nBatch Prediction Result:")
    print(json.dumps(batch_result, indent=2))
