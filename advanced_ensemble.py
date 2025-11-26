# advanced_ensemble.py

import numpy as np
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import xgboost as xgb

# IMPORTANT: Ensure correct import path
try:
    from main import UPIFraudDetector
except ImportError:
    raise ImportError("Unable to import UPIFraudDetector from main.py. "
                      "Check that main.py is in the same directory.")


class AdvancedUPIFraudDetector(UPIFraudDetector):

    def __init__(self):
        super().__init__()

    def create_ensemble_model(self, X_train, y_train):
        """Create advanced ensemble model"""

        # Safety check to avoid division by zero for scale_pos_weight
        fraud_count = np.sum(y_train == 1)
        nonfraud_count = np.sum(y_train == 0)
        if fraud_count == 0:
            fraud_count = 1  # prevent zero division

        # Individual models
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )

        gb = GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        )

        xgb_clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            scale_pos_weight=nonfraud_count / fraud_count,
            eval_metric='logloss',     # REQUIRED to remove warnings/errors
            use_label_encoder=False    # prevents label encoder error
        )

        # Initialize models dictionary if not present
        if not hasattr(self, 'models'):
            self.models = {}

        # Ensemble model
        self.models['ensemble'] = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('gb', gb),
                ('xgb', xgb_clf)
            ],
            voting='soft',
            weights=[2, 1, 2]  # Higher weight to RF and XGB
        )

        # Fit ensemble
        self.models['ensemble'].fit(X_train, y_train)

    def adaptive_threshold_tuning(self, X_val, y_val):
        """Dynamically adjust risk thresholds based on validation performance"""

        if 'ensemble' not in self.models:
            raise ValueError("Ensemble model not trained. Call create_ensemble_model() first.")

        # Get prediction probabilities
        try:
            y_pred_proba = self.models['ensemble'].predict_proba(X_val)[:, 1]
        except AttributeError:
            raise AttributeError("Model does not support predict_proba(). Ensure all models support soft voting.")

        thresholds = np.arange(0.1, 0.9, 0.05)
        best_score = -1
        best_threshold = 0.5

        for threshold in thresholds:
            y_pred = (y_pred_proba > threshold).astype(int)
            score = self.business_aware_scoring(y_val, y_pred)

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold

    def business_aware_scoring(self, y_true, y_pred):
        """Scoring function that considers business costs"""

        # Avoid errors for multiclass confusion matrices
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        cost_fp = 10   # Cost for false positives
        cost_fn = 100  # Cost for false negatives

        total_cost = (fp * cost_fp) + (fn * cost_fn)

        # Maximum possible cost = if everything was predicted wrong
        max_possible_cost = len(y_true) * cost_fn

        score = 1 - (total_cost / max_possible_cost)
        return score

