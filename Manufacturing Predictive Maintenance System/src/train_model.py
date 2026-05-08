"""
Advanced model training module for predictive maintenance.
Supports XGBoost, Random Forest, and Survival models with comprehensive evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix, roc_auc_score,
    f1_score, precision_score, recall_score,
    matthews_corrcoef, cohen_kappa_score
)
from sklearn.impute import SimpleImputer
import joblib
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
import warnings

warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.evaluation_results = {}

    def split_data(self, df: pd.DataFrame,
                   feature_cols: list,
                   target_col: str = 'machine_failure',
                   test_size: float = 0.2,
                   validation_size: float = 0.1) -> Tuple:

        X = df[feature_cols]
        y = df[target_col]

        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns
        )

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )

        val_size = validation_size / (1 - test_size)

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=self.random_state,
            stratify=y_temp
        )

        logger.info("Data split:")
        logger.info(f"  Train: {len(X_train)} samples")
        logger.info(f"  Validation: {len(X_val)} samples")
        logger.info(f"  Test: {len(X_test)} samples")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_random_forest(self,
                            X_train: pd.DataFrame,
                            y_train: pd.Series,
                            hyperparameter_tuning: bool = False):

        logger.info("Training Random Forest Classifier...")

        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=-1
        )

        rf.fit(X_train, y_train)

        logger.info("Random Forest trained successfully")
        return rf

    def train_gradient_boosting(self,
                                X_train: pd.DataFrame,
                                y_train: pd.Series,
                                hyperparameter_tuning: bool = False):

        logger.info("Training Gradient Boosting Classifier...")

        # Extra safety for NaN values
        X_train = X_train.fillna(X_train.mean())

        gb = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            subsample=0.9,
            min_samples_split=5,
            random_state=self.random_state
        )

        gb.fit(X_train, y_train)

        logger.info("Gradient Boosting trained successfully")
        return gb

    def train_xgboost(self,
                      X_train: pd.DataFrame,
                      y_train: pd.Series):

        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not installed")
            return None

        logger.info("Training XGBoost Classifier...")

        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=np.sum(y_train == 0) / np.sum(y_train == 1),
            random_state=self.random_state,
            n_jobs=-1
        )

        xgb_model.fit(X_train, y_train)

        logger.info("XGBoost trained successfully")
        return xgb_model

    def evaluate_model(self,
                       model: Any,
                       X_test: pd.DataFrame,
                       y_test: pd.Series,
                       model_name: str = "Model") -> Dict:

        logger.info(f"\nEvaluating {model_name}...")

        # Handle NaN in test data
        X_test = X_test.fillna(X_test.mean())

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': float((y_pred == y_test).mean()),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
            'matthews_corrcoef': float(matthews_corrcoef(y_test, y_pred)),
            'kappa': float(cohen_kappa_score(y_test, y_pred))
        }

        cm = confusion_matrix(y_test, y_pred)

        metrics['confusion_matrix'] = cm.tolist()

        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
        logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

        return metrics

    def compare_models(self,
                       models_dict: Dict,
                       X_test: pd.DataFrame,
                       y_test: pd.Series):

        logger.info("=" * 60)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 60)

        comparison_results = []

        for model_name, model in models_dict.items():

            if model is None:
                continue

            metrics = self.evaluate_model(
                model,
                X_test,
                y_test,
                model_name
            )

            metrics['model'] = model_name
            comparison_results.append(metrics)

        comparison_df = pd.DataFrame(comparison_results)
        comparison_df = comparison_df.set_index('model')

        comparison_df = comparison_df.sort_values(
            'roc_auc',
            ascending=False
        )

        logger.info("\nModel Comparison Results:")
        logger.info(
            comparison_df[
                ['accuracy', 'precision', 'recall',
                 'f1_score', 'roc_auc']
            ].to_string()
        )

        return comparison_df

    def save_model(self, model: Any, filepath: str):

        Path(filepath).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(model, filepath)

        logger.info(f"Model saved to {filepath}")

    def full_training_pipeline(self,
                               df: pd.DataFrame,
                               feature_cols: list,
                               target_col: str = 'machine_failure',
                               models_to_train: list = None,
                               save_path: str = "models"):

        if models_to_train is None:
            models_to_train = ['rf', 'gb']

        logger.info("=" * 60)
        logger.info("STARTING MODEL TRAINING PIPELINE")
        logger.info("=" * 60)

        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
            df,
            feature_cols,
            target_col
        )

        trained_models = {}

        # Random Forest
        if 'rf' in models_to_train:

            rf_model = self.train_random_forest(
                X_train,
                y_train
            )

            trained_models['Random Forest'] = rf_model

            self.save_model(
                rf_model,
                f"{save_path}/random_forest.pkl"
            )

        # Gradient Boosting
        if 'gb' in models_to_train:

            gb_model = self.train_gradient_boosting(
                X_train,
                y_train
            )

            trained_models['Gradient Boosting'] = gb_model

            self.save_model(
                gb_model,
                f"{save_path}/gradient_boosting.pkl"
            )

        # XGBoost
        if 'xgb' in models_to_train and XGBOOST_AVAILABLE:

            xgb_model = self.train_xgboost(
                X_train,
                y_train
            )

            if xgb_model:
                trained_models['XGBoost'] = xgb_model

                self.save_model(
                    xgb_model,
                    f"{save_path}/xgboost.pkl"
                )

        comparison_df = self.compare_models(
            trained_models,
            X_test,
            y_test
        )

        best_model_name = comparison_df['roc_auc'].idxmax()

        self.best_model = trained_models[best_model_name]

        logger.info("=" * 60)
        logger.info(f"BEST MODEL: {best_model_name}")
        logger.info(
            f"ROC-AUC Score: "
            f"{comparison_df.loc[best_model_name, 'roc_auc']:.4f}"
        )
        logger.info("=" * 60)

        return {
            'models': trained_models,
            'best_model': self.best_model,
            'best_model_name': best_model_name,
            'comparison': comparison_df
        }


if __name__ == "__main__":

    trainer = ModelTrainer()

    df = pd.read_csv(
        "data/processed/ai4i2020_engineered.csv"
    )

    exclude_cols = [
        'udi',
        'product_id',
        'machine_failure',
        'type',
        'twf',
        'hdf',
        'pwf',
        'osf',
        'rnf',
        'health_grade',
        'noise_flags'
    ]

    feature_cols = [
        col for col in df.columns
        if col not in exclude_cols
    ]

    results = trainer.full_training_pipeline(
        df=df,
        feature_cols=feature_cols,
        target_col='machine_failure',
        models_to_train=['rf', 'gb'],
        save_path='models'
    )

    logger.info("Training pipeline completed successfully!")