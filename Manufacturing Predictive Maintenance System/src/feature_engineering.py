"""
Advanced feature engineering module for predictive maintenance.
Creates health indices, rolling statistics, and failure risk indicators.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import logging
from typing import Tuple, List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Advanced feature engineering for predictive maintenance.
    Creates domain-specific features that capture machine degradation patterns.
    """
    
    def __init__(self):
        """Initialize FeatureEngineer."""
        self.scaler_config = {}
        self.feature_stats = {}
    
    def create_temperature_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temperature-derived features."""
        if 'process_temperature_k' in df.columns and 'air_temperature_k' in df.columns:
            df['temp_difference'] = (df['process_temperature_k'] - 
                                    df['air_temperature_k'])
            df['temp_ratio'] = (df['process_temperature_k'] / 
                               (df['air_temperature_k'] + 1e-6))
            df['temp_excess'] = np.maximum(
                df['process_temperature_k'] - 320, 0
            )
            logger.info("Temperature features created")
        
        return df
    
    def create_power_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create power and torque-related features.
        Power (W) = Torque (Nm) * RPM / 9.5488
        """
        if 'torque_nm' in df.columns and 'rotational_speed_rpm' in df.columns:
            # Mechanical power (Watts)
            df['power_w'] = (df['torque_nm'] * df['rotational_speed_rpm'] / 9.5488)
            
            # Power normalized to tool condition
            df['power_per_tool_wear'] = (df['power_w'] / 
                                        (df['tool_wear_min'] + 1))
            
            # Torque-speed product (indicates load)
            df['torque_speed_product'] = df['torque_nm'] * df['rotational_speed_rpm']
            
            logger.info("Power features created")
        
        return df
    
    def create_wear_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create tool wear intensity and degradation features."""
        if 'tool_wear_min' in df.columns:
            # Normalized wear (0-1)
            max_wear = df['tool_wear_min'].max()
            df['wear_ratio'] = df['tool_wear_min'] / (max_wear + 1e-6)
            
            # Wear intensity levels
            df['wear_level'] = pd.cut(df['tool_wear_min'], 
                                      bins=[0, 100, 200, 300, 400],
                                      labels=[0, 1, 2, 3],
                                      right=False).astype(float)
            
            # Exponential wear progression (assumes acceleration)
            df['wear_exponential'] = np.exp(df['wear_ratio']) - 1
            
            logger.info("Wear features created")
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, 
                               window: int = 5) -> pd.DataFrame:
        """
        Create rolling window statistics.
        Captures temporal patterns in sensor degradation.
        """
        feature_cols = ['tool_wear_min', 'rotational_speed_rpm', 
                       'torque_nm', 'process_temperature_k']
        
        # Sort by unique identifier to maintain order
        if 'udi' in df.columns:
            df = df.sort_values('udi').reset_index(drop=True)
        
        for col in feature_cols:
            if col not in df.columns:
                continue
            
            # Rolling mean
            df[f'{col}_rolling_mean_{window}'] = (
                df[col].rolling(window=window, min_periods=1).mean()
            )
            
            # Rolling std (volatility)
            df[f'{col}_rolling_std_{window}'] = (
                df[col].rolling(window=window, min_periods=1).std().fillna(0)
            )
            
            # Rolling min/max
            df[f'{col}_rolling_min_{window}'] = (
                df[col].rolling(window=window, min_periods=1).min()
            )
            df[f'{col}_rolling_max_{window}'] = (
                df[col].rolling(window=window, min_periods=1).max()
            )
        
        logger.info(f"Rolling features created (window={window})")
        return df
    
    def create_health_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create composite health index (0=good, 1=critical).
        Combines multiple degradation indicators.
        """
        if 'tool_wear_min' not in df.columns:
            return df
        
        # Component: Wear degradation
        max_wear = df['tool_wear_min'].max()
        wear_component = df['tool_wear_min'] / (max_wear + 1e-6) * 0.3
        
        # Component: Temperature stress
        if 'temp_difference' in df.columns:
            temp_component = (
                np.clip((df['temp_difference'] - 5) / 20, 0, 1) * 0.25
            )
        else:
            temp_component = 0
        
        # Component: Speed degradation (lower speed = potential issue)
        if 'rotational_speed_rpm' in df.columns:
            speed_component = (
                (2850 - df['rotational_speed_rpm']) / 2850 * 0.2
            ).clip(0, 1)
        else:
            speed_component = 0
        
        # Component: Power anomaly
        if 'power_w' in df.columns:
            power_mean = df['power_w'].mean()
            power_component = (
                np.abs(df['power_w'] - power_mean) / (power_mean + 1e-6) * 0.25
            ).clip(0, 1)
        else:
            power_component = 0
        
        # Composite health index
        df['health_index'] = (wear_component + temp_component + 
                             speed_component + power_component).clip(0, 1)
        
        # Health grade
        df['health_grade'] = pd.cut(df['health_index'],
                                    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                    labels=['Excellent', 'Good', 'Fair', 
                                           'Poor', 'Critical'],
                                    right=False)
        
        logger.info("Health index created")
        return df
    
    def create_failure_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create detailed failure risk score.
        Maps to multiple failure modes if present.
        """
        failure_cols = []
        for col in ['twf', 'hdf', 'pwf', 'osf', 'rnf']:
            if col in df.columns:
                failure_cols.append(col)
        
        if failure_cols:
            df['failure_risk_score'] = df[failure_cols].sum(axis=1)
        else:
            # Fallback if no failure mode columns
            df['failure_risk_score'] = 0
        
        logger.info("Failure risk score created")
        return df
    
    def create_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features that detect anomalies and unusual patterns.
        """
        if 'power_w' in df.columns:
            # Power variance
            power_rolling_std = df['power_w'].rolling(window=10, 
                                                      min_periods=1).std()
            df['power_anomaly_score'] = power_rolling_std / (df['power_w'].std() + 1e-6)
        
        if 'rotational_speed_rpm' in df.columns:
            # Speed stability
            speed_rolling_std = df['rotational_speed_rpm'].rolling(
                window=10, min_periods=1).std()
            df['speed_stability'] = 1 - (speed_rolling_std / 
                                        (df['rotational_speed_rpm'].std() + 1e-6))
        
        if 'torque_nm' in df.columns:
            # Torque variance
            torque_rolling_std = df['torque_nm'].rolling(window=10, 
                                                         min_periods=1).std()
            df['torque_variability'] = torque_rolling_std / (df['torque_nm'].std() + 1e-6)
        
        logger.info("Anomaly features created")
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key variables."""
        if 'temp_difference' in df.columns and 'torque_nm' in df.columns:
            df['temp_torque_interaction'] = (
                df['temp_difference'] * df['torque_nm'] / 
                (df['torque_nm'].max() + 1e-6)
            )
        
        if 'power_w' in df.columns and 'wear_ratio' in df.columns:
            df['power_wear_interaction'] = df['power_w'] * df['wear_ratio']
        
        if 'health_index' in df.columns and 'power_w' in df.columns:
            df['critical_stress_indicator'] = (
                df['health_index'] * (df['power_w'] / df['power_w'].max())
            )
        
        logger.info("Interaction features created")
        return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features if timestamp available."""
        if 'product_id' in df.columns:
            # Count records per product (operational time proxy)
            df['product_age'] = df.groupby('product_id').cumcount()
        
        if 'udi' in df.columns:
            # Sequential features
            df['udi_rank'] = pd.factorize(df['udi'])[0]
        
        logger.info("Time features created")
        return df
    
    def select_features(self, df: pd.DataFrame, 
                       target_col: str = 'machine_failure',
                       method: str = 'all') -> List[str]:
        """
        Feature selection using importance-based methods.
        
        Args:
            df: Input dataframe
            target_col: Target column name
            method: 'all' (return all), 'top_k' (top features)
        """
        if target_col not in df.columns:
            return df.columns.tolist()
        
        # Get numeric features
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features.remove(target_col) if target_col in numeric_features else None
        
        if method == 'all':
            logger.info(f"Selected {len(numeric_features)} numeric features")
            return numeric_features
        
        return numeric_features
    
    def scale_features(self, X_train: pd.DataFrame, 
                      X_test: pd.DataFrame = None,
                      method: str = 'standard') -> Tuple:
        """
        Scale features using specified method.
        
        Args:
            X_train: Training features
            X_test: Test features
            method: 'standard', 'minmax', or 'robust'
        """
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            return X_train, X_test
        
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns
        )
        
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns
            )
        
        logger.info(f"Features scaled using {method} method")
        
        return X_train_scaled, X_test_scaled, scaler
    
    def full_pipeline(self, df: pd.DataFrame, 
                     target_col: str = 'machine_failure') -> pd.DataFrame:
        """
        Execute complete feature engineering pipeline.
        """
        logger.info("="*60)
        logger.info("STARTING FEATURE ENGINEERING PIPELINE")
        logger.info("="*60)
        
        # Create basic features
        df = self.create_temperature_features(df)
        df = self.create_power_features(df)
        df = self.create_wear_features(df)
        
        # Create temporal features
        df = self.create_rolling_features(df, window=5)
        
        # Create derived indices
        df = self.create_health_index(df)
        df = self.create_failure_risk_score(df)
        df = self.create_anomaly_features(df)
        
        # Create interactions
        df = self.create_interaction_features(df)
        
        # Create time-based features
        df = self.create_time_features(df)
        
        logger.info("="*60)
        logger.info(f"PIPELINE COMPLETE - Created {df.shape[1]} total features")
        logger.info("="*60)
        
        return df


# Example usage
if __name__ == "__main__":
    engineer = FeatureEngineer()
    
    # Load cleaned data
    df = pd.read_csv("data/processed/ai4i2020_cleaned.csv")
    
    # Apply feature engineering
    df_engineered = engineer.full_pipeline(df)
    
    # Save engineered features
    df_engineered.to_csv(
        "data/processed/ai4i2020_engineered.csv", 
        index=False
    )
    logger.info("Engineered data saved to data/processed/ai4i2020_engineered.csv")
    
    # Display feature statistics
    logger.info("\nFeature Summary:")
    logger.info(df_engineered.info())
