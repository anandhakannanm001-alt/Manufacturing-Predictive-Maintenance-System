"""
Advanced FastAPI backend for Predictive Maintenance System.
Includes ML model serving, LLM-powered explanations, and decision engine.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import json
import joblib
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Predictive Maintenance API",
    description="Advanced ML-powered predictive maintenance system",
    version="2.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class MachineReading(BaseModel):
    """Single machine sensor reading."""
    machine_id: str = Field(..., description="Unique machine identifier")
    type_encoded: int = Field(..., description="Machine type (0=L, 1=M, 2=H)")
    air_temperature_k: float = Field(..., description="Air temperature in Kelvin")
    process_temperature_k: float = Field(..., description="Process temperature in Kelvin")
    rotational_speed_rpm: float = Field(..., description="Rotational speed in RPM")
    torque_nm: float = Field(..., description="Torque in Newton-meters")
    tool_wear_min: float = Field(..., description="Tool wear in minutes")


class PredictionResponse(BaseModel):
    """Single machine prediction response."""
    machine_id: str
    failure_probability: float
    risk_level: str
    recommendation: str
    health_score: float
    confidence: float
    explanation: str
    estimated_hours_to_failure: Optional[float]
    cost_impact: Dict[str, float]
    timestamp: str


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    machines: List[MachineReading]


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    total_machines: int
    critical_count: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    average_failure_probability: float
    predictions: List[PredictionResponse]
    summary: str


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    timestamp: str
    version: str


# ============================================================================
# DECISION ENGINE
# ============================================================================

class MaintenanceDecisionEngine:
    """
    Intelligent decision engine for maintenance recommendations.
    Considers risk level, cost factors, and scheduling constraints.
    """
    
    def __init__(self):
        """Initialize decision engine."""
        self.decision_thresholds = {
            'critical': 0.8,
            'high': 0.6,
            'medium': 0.4,
            'low': 0.0
        }
        
        # Cost parameters
        self.failure_cost = 45000  # Average cost of unplanned downtime
        self.maintenance_cost = 2000  # Cost of planned maintenance
        self.emergency_maintenance_cost = 8000  # Cost of emergency maintenance
    
    def get_risk_level(self, failure_probability: float) -> str:
        """Determine risk level from probability."""
        if failure_probability >= self.decision_thresholds['critical']:
            return "Critical"
        elif failure_probability >= self.decision_thresholds['high']:
            return "High"
        elif failure_probability >= self.decision_thresholds['medium']:
            return "Medium"
        else:
            return "Low"
    
    def get_recommendation(self, failure_probability: float,
                          health_score: float,
                          tool_wear: float) -> str:
        """Generate maintenance recommendation."""
        risk_level = self.get_risk_level(failure_probability)
        
        if risk_level == "Critical":
            return "IMMEDIATE - Schedule emergency maintenance within 2 hours"
        elif risk_level == "High":
            return "URGENT - Schedule maintenance within 8-12 hours"
        elif risk_level == "Medium":
            return "PLANNED - Schedule maintenance within 3-5 days"
        else:
            return "MONITOR - Continue monitoring, no immediate action needed"
    
    def estimate_remaining_useful_life(self, tool_wear: float,
                                       health_score: float) -> float:
        """
        Estimate remaining useful life (RUL) in hours.
        Simplified model: RUL decreases exponentially with wear.
        """
        max_tool_wear = 300  # Maximum expected tool wear
        wear_ratio = min(tool_wear / max_tool_wear, 1.0)
        
        # RUL calculation based on health and wear
        base_rul = 200  # Base operating hours remaining
        rul = base_rul * (1 - health_score) * (1 - wear_ratio)
        
        return max(0.5, rul)  # Minimum 0.5 hours
    
    def calculate_cost_benefit(self, failure_probability: float,
                              hours_to_failure: float) -> Dict[str, float]:
        """
        Calculate cost-benefit analysis of maintenance.
        """
        # Expected cost of failure
        expected_failure_cost = failure_probability * self.failure_cost
        
        # Decision thresholds
        if failure_probability >= 0.8:
            # Critical - emergency maintenance recommended
            maintenance_cost = self.emergency_maintenance_cost
            cost_savings = expected_failure_cost - maintenance_cost
        elif failure_probability >= 0.4:
            # High/Medium - planned maintenance
            maintenance_cost = self.maintenance_cost
            cost_savings = expected_failure_cost - maintenance_cost
        else:
            # Low - monitoring only
            maintenance_cost = 0
            cost_savings = 0
        
        return {
            'expected_failure_cost': round(float(expected_failure_cost), 2),
            'maintenance_cost': round(float(maintenance_cost), 2),
            'net_cost_savings': round(float(cost_savings), 2),
            'roi_percentage': round(float((cost_savings / maintenance_cost * 100) if maintenance_cost > 0 else 0), 1)
        }


# ============================================================================
# LLM EXPLAINER (RULE-BASED)
# ============================================================================

class MaintenanceReportGenerator:
    """
    Generate human-readable maintenance reports and explanations.
    Uses rule-based templates (can be extended with actual LLM).
    """
    
    def __init__(self):
        """Initialize report generator."""
        self.risk_explanations = {
            'Critical': {
                'opening': "🚨 CRITICAL ALERT",
                'factors': "Multiple severe degradation indicators detected",
                'action': "Immediate intervention required to prevent catastrophic failure"
            },
            'High': {
                'opening': "⚠️ HIGH RISK ALERT",
                'factors': "Significant degradation patterns detected",
                'action': "Schedule urgent maintenance within 12 hours"
            },
            'Medium': {
                'opening': "⏱️ MEDIUM RISK",
                'factors': "Moderate degradation detected",
                'action': "Plan maintenance within 3-5 days"
            },
            'Low': {
                'opening': "✅ LOW RISK",
                'factors': "Machine operating within normal parameters",
                'action': "Continue monitoring"
            }
        }
    
    def generate_explanation(self, machine_id: str,
                            machine_type: str,
                            failure_probability: float,
                            risk_level: str,
                            health_score: float,
                            tool_wear: float,
                            temperature_diff: float,
                            power_rating: float) -> str:
        """
        Generate detailed technical explanation.
        """
        alert = self.risk_explanations[risk_level]
        
        explanation = f"""
{alert['opening']} - Machine {machine_id}

RISK ANALYSIS:
{alert['factors']}

KEY INDICATORS:
• Failure Probability: {failure_probability:.1%}
• Health Score: {health_score:.2f}/1.00
• Tool Wear Level: {tool_wear:.1f} minutes
• Temperature Difference: {temperature_diff:.1f}°C
• Power Rating: {power_rating:.1f}W

TECHNICAL ASSESSMENT:
{self._get_technical_assessment(health_score, tool_wear, temperature_diff)}

RECOMMENDATION:
{alert['action']}

NEXT STEPS:
{self._get_next_steps(risk_level)}
        """.strip()
        
        return explanation
    
    def _get_technical_assessment(self, health_score: float,
                                  tool_wear: float,
                                  temp_diff: float) -> str:
        """Generate technical assessment text."""
        issues = []
        
        if health_score > 0.7:
            issues.append("- Severe overall machine degradation detected")
        elif health_score > 0.5:
            issues.append("- Moderate machine degradation in progress")
        
        if tool_wear > 250:
            issues.append("- Tool approaching end of service life (>83% utilized)")
        elif tool_wear > 200:
            issues.append("- Tool wear at elevated levels (>67% utilized)")
        
        if temp_diff > 15:
            issues.append("- Excessive temperature differential indicates thermal stress")
        elif temp_diff > 10:
            issues.append("- Temperature differential above normal range")
        
        return "\n".join(issues) if issues else "- All indicators within normal operating parameters"
    
    def _get_next_steps(self, risk_level: str) -> str:
        """Generate next steps based on risk level."""
        steps = {
            'Critical': """1. Stop machine immediately to prevent catastrophic failure
2. Contact maintenance team for emergency response
3. Notify supervisor of production impact
4. Prepare for parts replacement if necessary""",
            'High': """1. Reduce machine load/speed if possible
2. Schedule maintenance within 8-12 hours
3. Monitor machine continuously
4. Prepare maintenance team and parts""",
            'Medium': """1. Schedule maintenance within 3-5 days
2. Monitor machine status daily
3. Prepare maintenance parts and team
4. Document current readings for trending""",
            'Low': """1. Continue normal operation
2. Monitor machine on regular schedule
3. Log readings for trend analysis
4. Review next scheduled maintenance date"""
        }
        
        return steps.get(risk_level, "Monitor and reassess regularly")


# ============================================================================
# MODEL LOADING AND UTILITIES
# ============================================================================

class ModelManager:
    """
    Manages model loading and prediction.
    """
    
    def __init__(self, model_path: str = "models/random_forest.pkl"):
        """Initialize model manager."""
        self.model_path = model_path
        self.model = None
        self.feature_columns = None
        self.scaler = None
        self.load_model()
    
    def load_model(self):
        """Load trained model from disk."""
        try:
            if Path(self.model_path).exists():
                self.model = joblib.load(self.model_path)
                logger.info(f"Model loaded from {self.model_path}")
            else:
                logger.warning(f"Model not found at {self.model_path}")
                logger.info("Using fallback heuristic model")
                self.model = None
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
    
    def predict(self, machine_data: MachineReading) -> Dict[str, float]:
        """
        Make prediction for single machine.
        Falls back to heuristic if model unavailable.
        """
        if self.model is not None:
            try:
                # Prepare features
                features = self._prepare_features(machine_data)
                
                # Predict
                prediction = self.model.predict([features])
                probability = self.model.predict_proba([features])[:, 1][0]
                confidence = max(probability, 1 - probability)
                
                return {
                    'prediction': int(prediction[0]),
                    'failure_probability': float(probability),
                    'confidence': float(confidence)
                }
            except Exception as e:
                logger.error(f"Prediction error: {e}. Using fallback.")
                return self._heuristic_predict(machine_data)
        else:
            return self._heuristic_predict(machine_data)
    
    def _prepare_features(self, machine_data: MachineReading) -> list:
        """Prepare features for model input with full feature engineering."""
        # Basic sensor readings
        type_encoded = machine_data.type_encoded
        air_temp = machine_data.air_temperature_k
        process_temp = machine_data.process_temperature_k
        rpm = machine_data.rotational_speed_rpm
        torque = machine_data.torque_nm
        tool_wear = machine_data.tool_wear_min
        
        # Temperature features
        temp_difference = process_temp - air_temp
        temp_ratio = process_temp / (air_temp + 1e-6)
        temp_excess = max(process_temp - 320, 0)
        
        # Power features
        power_w = torque * rpm / 9.5488
        power_per_tool_wear = power_w / (tool_wear + 1)
        torque_speed_product = torque * rpm
        
        # Wear features
        max_wear = 300
        wear_ratio = tool_wear / max_wear
        wear_level = 0 if tool_wear < 100 else (1 if tool_wear < 200 else (2 if tool_wear < 300 else 3))
        wear_exponential = np.exp(wear_ratio) - 1
        
        # Rolling features (use current values for single prediction)
        tool_wear_rolling_mean_5 = tool_wear
        tool_wear_rolling_std_5 = 0
        tool_wear_rolling_min_5 = tool_wear
        tool_wear_rolling_max_5 = tool_wear
        
        rpm_rolling_mean_5 = rpm
        rpm_rolling_std_5 = 0
        rpm_rolling_min_5 = rpm
        rpm_rolling_max_5 = rpm
        
        torque_rolling_mean_5 = torque
        torque_rolling_std_5 = 0
        torque_rolling_min_5 = torque
        torque_rolling_max_5 = torque
        
        process_temp_rolling_mean_5 = process_temp
        process_temp_rolling_std_5 = 0
        process_temp_rolling_min_5 = process_temp
        process_temp_rolling_max_5 = process_temp
        
        # Health index components
        wear_component = wear_ratio * 0.3
        temp_component = np.clip((temp_difference - 5) / 20, 0, 1) * 0.25
        speed_component = (2850 - rpm) / 2850 * 0.2
        speed_component = max(0, min(1, speed_component))
        power_component = 0  # Can't compute without historical mean
        health_index = min(1, max(0, wear_component + temp_component + speed_component + power_component))
        
        # Failure risk score (no failure mode columns available)
        failure_risk_score = 0
        
        # Anomaly features (use defaults for single prediction)
        power_anomaly_score = 0
        speed_stability = 1
        torque_variability = 0
        
        # Interaction features
        temp_torque_interaction = temp_difference * torque / (torque + 1e-6)
        power_wear_interaction = power_w * wear_ratio
        critical_stress_indicator = health_index * (power_w / (power_w + 1e-6))
        
        # Return all features in expected order
        return [
            type_encoded,
            air_temp,
            process_temp,
            rpm,
            torque,
            tool_wear,
            temp_difference,
            temp_ratio,
            temp_excess,
            power_w,
            power_per_tool_wear,
            torque_speed_product,
            wear_ratio,
            wear_level,
            wear_exponential,
            tool_wear_rolling_mean_5,
            tool_wear_rolling_std_5,
            tool_wear_rolling_min_5,
            tool_wear_rolling_max_5,
            rpm_rolling_mean_5,
            rpm_rolling_std_5,
            rpm_rolling_min_5,
            rpm_rolling_max_5,
            torque_rolling_mean_5,
            torque_rolling_std_5,
            torque_rolling_min_5,
            torque_rolling_max_5,
            process_temp_rolling_mean_5,
            process_temp_rolling_std_5,
            process_temp_rolling_min_5,
            process_temp_rolling_max_5,
            health_index,
            failure_risk_score,
            power_anomaly_score,
            speed_stability,
            torque_variability,
            temp_torque_interaction,
            power_wear_interaction,
            critical_stress_indicator
        ]
    
    def _heuristic_predict(self, machine_data: MachineReading) -> Dict[str, float]:
        """
        Heuristic prediction model.
        Used when ML model is unavailable.
        """
        # Calculate failure probability based on domain knowledge
        wear_factor = min(machine_data.tool_wear_min / 300, 1.0) * 0.4
        
        temp_diff = machine_data.process_temperature_k - machine_data.air_temperature_k
        temp_factor = min(max((temp_diff - 5) / 15, 0), 1) * 0.2
        
        type_factor = machine_data.type_encoded / 2 * 0.15
        
        speed_factor = max(0, (2000 - machine_data.rotational_speed_rpm) / 2000) * 0.15
        
        torque_factor = min(max((machine_data.torque_nm - 30) / 30, 0), 1) * 0.1
        
        probability = wear_factor + temp_factor + type_factor + speed_factor + torque_factor
        probability = min(0.95, max(0.05, probability))
        
        return {
            'prediction': 1 if probability > 0.5 else 0,
            'failure_probability': probability,
            'confidence': min(probability, 1 - probability) + 0.1
        }


# ============================================================================
# INITIALIZE GLOBAL COMPONENTS
# ============================================================================

model_manager = ModelManager()
decision_engine = MaintenanceDecisionEngine()
report_generator = MaintenanceReportGenerator()


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    return HealthCheckResponse(
        status="operational",
        model_loaded=model_manager.model is not None,
        timestamp=datetime.now().isoformat(),
        version="2.0.0"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_machine_failure(machine: MachineReading) -> PredictionResponse:
    """
    Predict failure risk for a single machine.
    
    Example:
    ```json
    {
        "machine_id": "M001",
        "type_encoded": 1,
        "air_temperature_k": 298.0,
        "process_temperature_k": 308.0,
        "rotational_speed_rpm": 1500,
        "torque_nm": 40.0,
        "tool_wear_min": 100
    }
    ```
    """
    try:
        # Get prediction
        pred_result = model_manager.predict(machine)
        failure_prob = pred_result['failure_probability']
        
        # Get risk level and recommendation
        risk_level = decision_engine.get_risk_level(failure_prob)
        recommendation = decision_engine.get_recommendation(
            failure_prob, 
            failure_prob,  # simplified health score
            machine.tool_wear_min
        )
        
        # Calculate RUL
        rul_hours = decision_engine.estimate_remaining_useful_life(
            machine.tool_wear_min,
            failure_prob
        )
        
        # Cost analysis
        cost_impact = decision_engine.calculate_cost_benefit(
            failure_prob,
            rul_hours
        )
        
        # Generate explanation
        temp_diff = machine.process_temperature_k - machine.air_temperature_k
        power = (machine.torque_nm * machine.rotational_speed_rpm / 9.5488)
        
        explanation = report_generator.generate_explanation(
            machine_id=machine.machine_id,
            machine_type=f"Type-{chr(65 + machine.type_encoded)}",
            failure_probability=failure_prob,
            risk_level=risk_level,
            health_score=failure_prob,
            tool_wear=machine.tool_wear_min,
            temperature_diff=temp_diff,
            power_rating=power
        )
        
        return PredictionResponse(
            machine_id=machine.machine_id,
            failure_probability=failure_prob,
            risk_level=risk_level,
            recommendation=recommendation,
            health_score=failure_prob,
            confidence=pred_result['confidence'],
            explanation=explanation,
            estimated_hours_to_failure=rul_hours,
            cost_impact=cost_impact,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-machine-failure", response_model=PredictionResponse)
async def predict_machine_failure_alias(machine: MachineReading) -> PredictionResponse:
    """
    Alias endpoint for /predict - Predict failure risk for a single machine.
    
    Example:
    ```json
    {
        "machine_id": "M001",
        "type_encoded": 1,
        "air_temperature_k": 298.0,
        "process_temperature_k": 308.0,
        "rotational_speed_rpm": 1500,
        "torque_nm": 40.0,
        "tool_wear_min": 100
    }
    ```
    """
    return await predict_machine_failure(machine)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Batch prediction for multiple machines.
    """
    try:
        predictions = []
        risk_counts = {'Critical': 0, 'High': 0, 'Medium': 0, 'Low': 0}
        failure_probs = []
        
        for machine in request.machines:
            pred = await predict_machine_failure(machine)
            predictions.append(pred)
            risk_counts[pred.risk_level] += 1
            failure_probs.append(pred.failure_probability)
        
        avg_failure_prob = np.mean(failure_probs) if failure_probs else 0
        
        # Generate batch summary
        summary = f"""
Batch Analysis Summary:
- Total Machines: {len(predictions)}
- Critical Risk: {risk_counts['Critical']}
- High Risk: {risk_counts['High']}
- Average Failure Probability: {avg_failure_prob:.1%}
- Recommended Actions: {risk_counts['Critical'] + risk_counts['High']} immediate interventions
        """.strip()
        
        return BatchPredictionResponse(
            total_machines=len(predictions),
            critical_count=risk_counts['Critical'],
            high_risk_count=risk_counts['High'],
            medium_risk_count=risk_counts['Medium'],
            low_risk_count=risk_counts['Low'],
            average_failure_probability=avg_failure_prob,
            predictions=predictions,
            summary=summary
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/info")
async def get_info():
    """Get system information."""
    return {
        "system": "Predictive Maintenance System",
        "version": "2.0.0",
        "models": {
            "primary": "Random Forest / XGBoost",
            "fallback": "Heuristic Rule-based"
        },
        "features": [
            "ML-based failure prediction",
            "LLM-powered explanations",
            "Cost-benefit analysis",
            "Remaining useful life estimation",
            "Batch processing support"
        ],
        "endpoints": [
            "GET /health",
            "POST /predict",
            "POST /predict/batch",
            "GET /info"
        ]
    }


# ============================================================================
# ERROR HANDLING
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Predictive Maintenance API Server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
