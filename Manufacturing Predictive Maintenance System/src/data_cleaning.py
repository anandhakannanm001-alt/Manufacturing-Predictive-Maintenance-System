"""
Comprehensive data cleaning module for predictive maintenance.
Handles sensor noise, abnormal readings, and class imbalance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import logging
from typing import Tuple, List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Advanced data cleaning pipeline for predictive maintenance datasets.
    Handles sensor noise, outliers, missing values, and class imbalance.
    """
    
    def __init__(self, outlier_threshold: float = 3.0):
        """
        Initialize DataCleaner.
        
        Args:
            outlier_threshold: Z-score threshold for outlier detection (default: 3.0)
        """
        self.outlier_threshold = outlier_threshold
        self.scaler_stats = {}
        self.feature_stats = {}
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load raw CSV data."""
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df
    
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to lowercase with underscores."""
        df.columns = (df.columns
                     .str.lower()
                     .str.replace(' ', '_')
                     .str.replace('[', '')
                     .str.replace(']', '')
                     .str.replace('(', '')
                     .str.replace(')', ''))
        logger.info("Column names cleaned")
        return df
    
    def analyze_missing_values(self, df: pd.DataFrame) -> Dict:
        """Analyze missing value patterns."""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        
        report = {
            'total_missing': missing.sum(),
            'columns_with_missing': missing[missing > 0].to_dict(),
            'percentages': missing_pct[missing_pct > 0].to_dict()
        }
        
        if missing.sum() > 0:
            logger.warning(f"Missing values detected: {report}")
        else:
            logger.info("No missing values found")
            
        return report
    
    def handle_missing_values(self, df: pd.DataFrame, 
                             strategy: str = 'median') -> pd.DataFrame:
        """
        Handle missing values using specified strategy.
        
        Args:
            df: Input dataframe
            strategy: 'median', 'mean', 'forward_fill', or 'drop'
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if strategy == 'median':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif strategy == 'mean':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif strategy == 'forward_fill':
            df[numeric_cols] = df[numeric_cols].fillna(method='ffill')
        elif strategy == 'drop':
            df = df.dropna(subset=numeric_cols)
        
        logger.info(f"Missing values handled using {strategy} strategy")
        return df
    
    def detect_sensor_noise(self, df: pd.DataFrame, 
                           sensor_cols: List[str] = None) -> pd.DataFrame:
        """
        Detect and flag sensor noise patterns.
        Identifies sudden spikes, constant values, and anomalous readings.
        """
        if sensor_cols is None:
            sensor_cols = ['air_temperature_k', 'process_temperature_k', 
                          'rotational_speed_rpm', 'torque_nm', 'tool_wear_min']
        
        df['noise_flags'] = 0
        
        for col in sensor_cols:
            if col not in df.columns:
                continue
                
            # Detect sudden spikes using differencing
            if len(df) > 1:
                diff = df[col].diff().abs()
                threshold = diff.quantile(0.99)
                spikes = diff > threshold
                df.loc[spikes, 'noise_flags'] += 1
            
            # Detect constant values (dead sensor)
            const_windows = df[col].rolling(window=10).std() == 0
            df.loc[const_windows, 'noise_flags'] += 1
        
        noise_count = (df['noise_flags'] > 0).sum()
        logger.info(f"Detected noise in {noise_count} records ({noise_count/len(df)*100:.2f}%)")
        
        return df
    
    def remove_outliers_zscore(self, df: pd.DataFrame, 
                              columns: List[str] = None) -> Tuple[pd.DataFrame, int]:
        """
        Remove outliers using Z-score method.
        Returns cleaned dataframe and count of removed records.
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        initial_len = len(df)
        df_clean = df.copy()
        
        for col in columns:
            if col not in df_clean.columns or col == 'machine_failure':
                continue
                
            z_scores = np.abs(stats.zscore(df_clean[col], nan_policy='omit'))
            df_clean = df_clean[z_scores < self.outlier_threshold]
        
        removed = initial_len - len(df_clean)
        pct_removed = (removed / initial_len * 100) if initial_len > 0 else 0
        
        logger.info(f"Removed {removed} outliers ({pct_removed:.2f}%)")
        
        return df_clean, removed
    
    def remove_outliers_iqr(self, df: pd.DataFrame, 
                           columns: List[str] = None,
                           multiplier: float = 1.5) -> Tuple[pd.DataFrame, int]:
        """
        Remove outliers using Interquartile Range (IQR) method.
        More robust for non-normal distributions.
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        initial_len = len(df)
        df_clean = df.copy()
        
        for col in columns:
            if col not in df_clean.columns or col == 'machine_failure':
                continue
                
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            df_clean = df_clean[(df_clean[col] >= lower_bound) & 
                               (df_clean[col] <= upper_bound)]
        
        removed = initial_len - len(df_clean)
        pct_removed = (removed / initial_len * 100) if initial_len > 0 else 0
        
        logger.info(f"Removed {removed} outliers using IQR ({pct_removed:.2f}%)")
        
        return df_clean, removed
    
    def remove_duplicates(self, df: pd.DataFrame, 
                         subset: List[str] = None) -> Tuple[pd.DataFrame, int]:
        """Remove duplicate records."""
        initial_len = len(df)
        
        if subset:
            df = df.drop_duplicates(subset=subset, keep='first')
        else:
            df = df.drop_duplicates(keep='first')
        
        removed = initial_len - len(df)
        logger.info(f"Removed {removed} duplicate records")
        
        return df, removed
    
    def analyze_class_imbalance(self, df: pd.DataFrame, 
                               target_col: str = 'machine_failure') -> Dict:
        """Analyze class distribution and imbalance ratio."""
        if target_col not in df.columns:
            logger.warning(f"Target column '{target_col}' not found")
            return {}
        
        class_counts = df[target_col].value_counts()
        class_pct = (class_counts / len(df) * 100).round(2)
        
        report = {
            'class_distribution': class_counts.to_dict(),
            'class_percentages': class_pct.to_dict(),
            'imbalance_ratio': class_counts.max() / class_counts.min() if len(class_counts) > 1 else 1
        }
        
        logger.warning(f"Class imbalance ratio: {report['imbalance_ratio']:.2f}:1")
        logger.info(f"Class distribution:\n{class_counts}")
        
        return report
    
    def handle_class_imbalance(self, df: pd.DataFrame, 
                              target_col: str = 'machine_failure',
                              method: str = 'oversample') -> pd.DataFrame:
        """
        Handle class imbalance using oversampling or undersampling.
        
        Args:
            df: Input dataframe
            target_col: Target column name
            method: 'oversample' or 'undersample'
        """
        if target_col not in df.columns:
            return df
        
        positive_class = df[df[target_col] == 1]
        negative_class = df[df[target_col] == 0]
        
        if method == 'oversample':
            positive_class = positive_class.sample(n=len(negative_class), 
                                                   replace=True, 
                                                   random_state=42)
            df_balanced = pd.concat([negative_class, positive_class], 
                                   ignore_index=True)
        elif method == 'undersample':
            negative_class = negative_class.sample(n=len(positive_class), 
                                                  random_state=42)
            df_balanced = pd.concat([negative_class, positive_class], 
                                   ignore_index=True)
        else:
            return df
        
        logger.info(f"Applied {method} to balance classes")
        logger.info(f"New class distribution:\n{df_balanced[target_col].value_counts()}")
        
        return df_balanced.sample(frac=1).reset_index(drop=True)
    
    def validate_ranges(self, df: pd.DataFrame) -> Dict:
        """Validate sensor readings are within expected ranges."""
        validation_report = {}
        
        # Expected ranges for industrial machinery
        expected_ranges = {
            'air_temperature_k': (250, 350),
            'process_temperature_k': (250, 350),
            'rotational_speed_rpm': (1200, 2850),
            'torque_nm': (3.6, 76),
            'tool_wear_min': (0, 300)
        }
        
        for col, (min_val, max_val) in expected_ranges.items():
            if col not in df.columns:
                continue
            
            out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
            
            validation_report[col] = {
                'expected_range': (min_val, max_val),
                'out_of_range_count': out_of_range,
                'percentage': (out_of_range / len(df) * 100) if len(df) > 0 else 0,
                'actual_min': df[col].min(),
                'actual_max': df[col].max()
            }
        
        logger.info("Validation ranges checked")
        return validation_report
    
    def full_pipeline(self, filepath: str, 
                     handle_imbalance: bool = True,
                     outlier_method: str = 'iqr') -> pd.DataFrame:
        """
        Execute complete cleaning pipeline.
        
        Args:
            filepath: Path to raw data CSV
            handle_imbalance: Whether to balance classes
            outlier_method: 'zscore' or 'iqr'
        """
        logger.info("="*60)
        logger.info("STARTING DATA CLEANING PIPELINE")
        logger.info("="*60)
        
        # Load and clean
        df = self.load_data(filepath)
        df = self.clean_column_names(df)
        
        # Analyze and handle missing values
        self.analyze_missing_values(df)
        df = self.handle_missing_values(df)
        
        # Remove duplicates
        df, _ = self.remove_duplicates(df)
        
        # Detect noise
        df = self.detect_sensor_noise(df)
        
        # Validate ranges
        validation = self.validate_ranges(df)
        
        # Remove outliers
        if outlier_method == 'zscore':
            df, _ = self.remove_outliers_zscore(df)
        else:
            df, _ = self.remove_outliers_iqr(df)
        
        # Analyze class imbalance
        imbalance_report = self.analyze_class_imbalance(df)
        
        # Handle class imbalance
        if handle_imbalance and imbalance_report.get('imbalance_ratio', 1) > 1.5:
            df = self.handle_class_imbalance(df, method='oversample')
        
        logger.info("="*60)
        logger.info(f"PIPELINE COMPLETE - Final dataset: {len(df)} records")
        logger.info("="*60)
        
        return df


# Example usage
if __name__ == "__main__":
    cleaner = DataCleaner(outlier_threshold=3.0)
    
    # Process data
    df_cleaned = cleaner.full_pipeline(
        filepath="data/raw/ai4i2020.csv",
        handle_imbalance=True,
        outlier_method='iqr'
    )
    
    # Save cleaned data
    df_cleaned.to_csv("data/processed/ai4i2020_cleaned.csv", index=False)
    logger.info("Cleaned data saved to data/processed/ai4i2020_cleaned.csv")
