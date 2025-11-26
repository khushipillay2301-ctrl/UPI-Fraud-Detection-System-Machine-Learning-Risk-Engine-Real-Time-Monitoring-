import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Optional imports used in advanced ensemble section
try:
    import xgboost as xgb
except Exception:
    xgb = None

class UPIFraudDetector:
    def __init__(self):
        self.models = {}
        self.scalers = {}  # will hold 'imputer' and 'standard'
        self.label_encoders = {}
        self.feature_importance = {}
        
    def generate_synthetic_data(self, n_samples=10000):
        """Generate synthetic UPI transaction data for demonstration"""
        np.random.seed(42)
        
        data = {
            'transaction_id': range(n_samples),
            'amount': np.random.exponential(500, n_samples),
            'time_of_day': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'user_age': np.random.randint(18, 70, n_samples),
            'user_income_bracket': np.random.choice(['Low', 'Medium', 'High'], n_samples),
            'device_type': np.random.choice(['Mobile', 'Tablet', 'Desktop'], n_samples),
            'location_city': np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata'], n_samples),
            'transaction_type': np.random.choice(['P2P', 'Merchant', 'Utility', 'Recharge'], n_samples),
            'previous_chargebacks': np.random.poisson(0.1, n_samples),
            'account_age_days': np.random.randint(1, 3650, n_samples),
            'transaction_frequency_1h': np.random.poisson(1, n_samples),
            'transaction_frequency_24h': np.random.poisson(5, n_samples),
            'avg_transaction_amount': np.random.exponential(400, n_samples),
            'is_new_device': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'is_foreign_country': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            'time_since_last_transaction': np.random.exponential(3600, n_samples)  # seconds
        }
        
        df = pd.DataFrame(data)
        
        # Create synthetic fraud patterns
        fraud_probability = (
            (df['amount'] > 2000).astype(float) * 0.3 +
            (df['is_new_device'] == 1).astype(float) * 0.2 +
            (df['is_foreign_country'] == 1).astype(float) * 0.4 +
            df['time_of_day'].between(0, 5).astype(float) * 0.1 +
            (df['transaction_frequency_1h'] > 3).astype(float) * 0.3 +
            (df['previous_chargebacks'] > 0).astype(float) * 0.5 +
            np.random.normal(0, 0.1, n_samples)
        )
        
        df['is_fraud'] = (fraud_probability > 0.5).astype(int)
        
        # Adjust fraud rate to ~5% safely
        fraud_indices = df[df['is_fraud'] == 1].index.to_numpy()
        if len(fraud_indices) > 0:
            keep_count = max(1, int(len(fraud_indices) * 0.05))
            keep_fraud = np.random.choice(fraud_indices, size=keep_count, replace=False)
            df['is_fraud'] = 0
            df.loc[keep_fraud, 'is_fraud'] = 1
        else:
            # fallback: set a small random subset as fraud
            idx = np.random.choice(df.index, size=max(1, int(0.05 * len(df))), replace=False)
            df['is_fraud'] = 0
            df.loc[idx, 'is_fraud'] = 1
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess the transaction data (works when 'is_fraud' may be missing)."""
        df_processed = df.copy()
        
        # Ensure expected columns exist (if not, fill with NaN so imputer can handle)
        expected_cats = ['user_income_bracket', 'device_type', 'location_city', 'transaction_type']
        for c in expected_cats:
            if c not in df_processed.columns:
                df_processed[c] = np.nan
        
        # Handle categorical variables using persistent label encoders.
        categorical_cols = expected_cats
        for col in categorical_cols:
            col_values = df_processed[col].astype(str).fillna('nan_placeholder')
            if col not in self.label_encoders:
                le = LabelEncoder()
                # fit on what's present
                le.fit(col_values)
                self.label_encoders[col] = le
                df_processed[col] = le.transform(col_values)
            else:
                le = self.label_encoders[col]
                # map using existing classes_; unseen -> -1
                class_to_code = {c: i for i, c in enumerate(le.classes_)}
                df_processed[col] = col_values.map(lambda x: class_to_code.get(x, -1)).astype(int)
        
        # Feature engineering
        # Make sure numeric columns exist to avoid KeyError
        for col in ['amount', 'avg_transaction_amount', 'transaction_frequency_1h', 'transaction_frequency_24h', 'time_since_last_transaction']:
            if col not in df_processed.columns:
                df_processed[col] = 0.0
        
        df_processed['amount_to_avg_ratio'] = df_processed['amount'] / (df_processed['avg_transaction_amount'] + 1)
        df_processed['hourly_transaction_velocity'] = df_processed['transaction_frequency_1h'] / 1
        df_processed['daily_transaction_velocity'] = df_processed['transaction_frequency_24h'] / 24
        df_processed['transaction_size_risk'] = (df_processed['amount'] > 2000).astype(int)
        
        # Select features for modeling
        feature_cols = [
            'amount', 'time_of_day', 'day_of_week', 'user_age', 
            'user_income_bracket', 'device_type', 'location_city',
            'transaction_type', 'previous_chargebacks', 'account_age_days',
            'transaction_frequency_1h', 'transaction_frequency_24h',
            'avg_transaction_amount', 'is_new_device', 'is_foreign_country',
            'time_since_last_transaction', 'amount_to_avg_ratio',
            'hourly_transaction_velocity', 'daily_transaction_velocity',
            'transaction_size_risk'
        ]
        
        # Ensure any missing feature columns are created (with zeros)
        for col in feature_cols:
            if col not in df_processed.columns:
                df_processed[col] = 0
        
        X = df_processed[feature_cols].copy()
        y = df_processed['is_fraud'].copy() if 'is_fraud' in df_processed.columns else None
        
        return X, y, feature_cols
    
    def train_models(self, X_train, y_train):
        """Train multiple ML models"""
        # Setup imputer and scaler if not already
        self.scalers['imputer'] = SimpleImputer(strategy='mean')
        self.scalers['standard'] = StandardScaler()
        
        # Impute missing values and fit scaler
        X_train_imputed = pd.DataFrame(
            self.scalers['imputer'].fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_train_scaled = pd.DataFrame(
            self.scalers['standard'].fit_transform(X_train_imputed),
            columns=X_train.columns,
            index=X_train.index
        )
        
        # Models
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        self.models['logistic_regression'] = LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        )
        
        # Train all models
        for name, model in self.models.items():
            if name == 'logistic_regression':
                model.fit(X_train_scaled, y_train)
            else:
                # tree-based models are robust to scaling; use imputed
                model.fit(X_train_imputed, y_train)
            
            # Store feature importance if available
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        results = {}
        # Transform test set similarly to training
        X_test_imputed = pd.DataFrame(
            self.scalers['imputer'].transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        X_test_scaled = pd.DataFrame(
            self.scalers['standard'].transform(X_test_imputed),
            columns=X_test.columns,
            index=X_test.index
        )
        
        for name, model in self.models.items():
            if name == 'logistic_regression':
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                y_pred = model.predict(X_test_scaled)
            else:
                # tree based: use imputed
                # some models might not implement predict_proba (rare) - handle that
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test_imputed)[:, 1]
                else:
                    # fall back to decision_function or predicted labels
                    if hasattr(model, 'decision_function'):
                        scores = model.decision_function(X_test_imputed)
                        # normalize into 0-1
                        y_pred_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
                    else:
                        y_pred_proba = model.predict(X_test_imputed)
                y_pred = model.predict(X_test_imputed)
            
            # If y_test is all one class (rare), roc_auc_score will error - guard it
            try:
                auc_score = roc_auc_score(y_test, y_pred_proba)
            except Exception:
                auc_score = float('nan')
            
            results[name] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'auc_score': auc_score,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            print(f"\n{name.upper()} Results:")
            print(f"AUC Score: {auc_score:.4f}" if not np.isnan(auc_score) else "AUC Score: N/A")
            print(classification_report(y_test, y_pred))
        
        return results
    
    def plot_feature_importance(self, feature_names, top_n=15):
        """Plot feature importance for tree-based models"""
        if not self.feature_importance:
            print("No feature importance available. Train tree models first.")
            return
        
        items = list(self.feature_importance.items())
        n_plots = min(len(items), 2)
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots + 4, 6))
        if n_plots == 1:
            axes = [axes]
        
        for idx in range(n_plots):
            model_name, importance = items[idx]
            feature_imp = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=True).tail(top_n)
            
            axes[idx].barh(feature_imp['feature'], feature_imp['importance'])
            axes[idx].set_title(f'Feature Importance - {model_name.title()}')
            axes[idx].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.show()


class RiskManagementSystem:
    def __init__(self, fraud_detector):
        self.fraud_detector = fraud_detector
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.7,
            'high': 0.9
        }
    
    def assess_risk(self, transaction_data):
        """Assess risk for a single transaction (accepts dict or single-row DataFrame)"""
        # Convert input to DataFrame if necessary
        if isinstance(transaction_data, dict):
            transaction_df = pd.DataFrame([transaction_data])
        elif isinstance(transaction_data, pd.Series):
            transaction_df = pd.DataFrame([transaction_data.to_dict()])
        elif isinstance(transaction_data, pd.DataFrame):
            transaction_df = transaction_data.reset_index(drop=True)
        else:
            raise ValueError("transaction_data must be dict, Series, or DataFrame")
        
        # Preprocess transaction (this will use the label_encoders already fitted during training)
        X, _, feature_cols = self.fraud_detector.preprocess_data(transaction_df)
        
        # Impute and transform using trained scalers
        X_imputed = pd.DataFrame(
            self.fraud_detector.scalers['imputer'].transform(X),
            columns=X.columns,
            index=X.index
        )
        X_scaled = pd.DataFrame(
            self.fraud_detector.scalers['standard'].transform(X_imputed),
            columns=X.columns,
            index=X.index
        )
        
        # Get probability from best model (Random Forest in this case if present)
        if 'random_forest' in self.fraud_detector.models:
            model = self.fraud_detector.models['random_forest']
            prob = model.predict_proba(X_imputed)[0, 1]
        elif 'ensemble' in self.fraud_detector.models:
            model = self.fraud_detector.models['ensemble']
            prob = model.predict_proba(X_imputed)[0, 1]
        elif 'logistic_regression' in self.fraud_detector.models:
            model = self.fraud_detector.models['logistic_regression']
            prob = model.predict_proba(X_scaled)[0, 1]
        else:
            raise RuntimeError("No trained model available to assess risk.")
        
        fraud_probability = float(prob)
        
        # Determine risk level and action
        if fraud_probability < self.risk_thresholds['low']:
            risk_level = "LOW"
            action = "APPROVE"
            message = "Transaction approved"
        elif fraud_probability < self.risk_thresholds['medium']:
            risk_level = "MEDIUM"
            action = "REVIEW"
            message = "Additional verification required"
        elif fraud_probability < self.risk_thresholds['high']:
            risk_level = "HIGH"
            action = "2FA_REQUIRED"
            message = "Two-factor authentication required"
        else:
            risk_level = "CRITICAL"
            action = "BLOCK"
            message = "Transaction blocked - high fraud risk"
        
        return {
            'fraud_probability': fraud_probability,
            'risk_level': risk_level,
            'action': action,
            'message': message
        }
    
    def real_time_monitoring_dashboard(self, transactions):
        """Simulate real-time transaction monitoring (transactions: DataFrame)"""
        results = []
        for _, transaction in transactions.iterrows():
            risk_assessment = self.assess_risk(pd.DataFrame([transaction]))
            results.append(risk_assessment)
        return pd.DataFrame(results)


# Real-time monitoring class (keeps same logic as original but made robust)
import threading
import time
from collections import deque
import json

class RealTimeUPIMonitor:
    def __init__(self, fraud_detector, risk_system):
        self.fraud_detector = fraud_detector
        self.risk_system = risk_system
        self.transaction_queue = deque()
        self.alert_queue = deque()
        self.running = False
        self.monitor_thread = None
        
    def add_transaction(self, transaction_data):
        """Add transaction to processing queue"""
        self.transaction_queue.append(transaction_data)
        
    def process_transactions(self):
        """Process transactions in real-time"""
        while self.running:
            if self.transaction_queue:
                transaction = self.transaction_queue.popleft()
                
                # Risk assessment
                risk_result = self.risk_system.assess_risk(pd.DataFrame([transaction]))
                
                # Trigger alerts for high risk
                if risk_result['risk_level'] in ['HIGH', 'CRITICAL']:
                    self.alert_queue.append({
                        'transaction': transaction,
                        'risk_assessment': risk_result,
                        'timestamp': time.time()
                    })
                    self.trigger_alert(risk_result, transaction)
                
                # Log result
                self.log_transaction(transaction, risk_result)
            
            time.sleep(0.01)  # Small delay to prevent CPU overload
    
    def trigger_alert(self, risk_result, transaction):
        """Trigger appropriate alerts based on risk level"""
        alert_message = {
            'LOW': f"Low risk transaction: {transaction.get('amount', 'N/A')}",
            'MEDIUM': f"Medium risk - review needed: {transaction.get('amount', 'N/A')}",
            'HIGH': f"HIGH RISK - 2FA required: {transaction.get('amount', 'N/A')}",
            'CRITICAL': f"CRITICAL RISK - BLOCKED: {transaction.get('amount', 'N/A')}"
        }
        
        print(f"ALERT: {alert_message[risk_result['risk_level']]}")
    
    def log_transaction(self, transaction, risk_result):
        """Log transaction and risk assessment"""
        log_entry = {
            'timestamp': time.time(),
            'transaction_id': transaction.get('transaction_id'),
            'amount': transaction.get('amount'),
            'risk_level': risk_result['risk_level'],
            'fraud_probability': risk_result['fraud_probability'],
            'action_taken': risk_result['action']
        }
        print(f"LOG: {json.dumps(log_entry)}")
    
    def start_monitoring(self):
        """Start the real-time monitoring system"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self.process_transactions)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("Real-time UPI monitoring started...")
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.running = False
        if self.monitor_thread is not None:
            self.monitor_thread.join(timeout=1)
        print("Real-time UPI monitoring stopped.")


# Advanced ensemble class (optional use of XGBoost)
class AdvancedUPIFraudDetector(UPIFraudDetector):
    def __init__(self):
        super().__init__()
        
    def create_ensemble_model(self, X_train, y_train):
        """Create advanced ensemble model (xgboost optional)"""
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        estimators = [('rf', rf), ('gb', gb)]
        
        if xgb is not None:
            xgb_clf = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=max(1.0, (len(y_train[y_train==0]) / max(1, len(y_train[y_train==1]))))
            )
            estimators.append(('xgb', xgb_clf))
        else:
            print("xgboost not available; building ensemble without it.")
        
        self.models['ensemble'] = VotingClassifier(
            estimators=estimators,
            voting='soft'
        )
        
        # Fit using imputed inputs
        X_train_imputed = pd.DataFrame(
            self.scalers['imputer'].transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        self.models['ensemble'].fit(X_train_imputed, y_train)
    
    def adaptive_threshold_tuning(self, X_val, y_val):
        """Dynamically adjust risk thresholds based on validation performance"""
        y_pred_proba = self.models['ensemble'].predict_proba(
            pd.DataFrame(self.scalers['imputer'].transform(X_val), columns=X_val.columns)
        )[:, 1]
        
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_score = -np.inf
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
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Business costs (example values)
        cost_fp = 10  # Cost of false positive (customer inconvenience)
        cost_fn = 100  # Cost of false negative (fraud loss)
        
        total_cost = (fp * cost_fp) + (fn * cost_fn)
        max_possible_cost = (len(y_true) * cost_fn)
        
        score = 1 - (total_cost / max_possible_cost)
        return score


def main():
    # Initialize fraud detector
    fraud_detector = UPIFraudDetector()
    
    # Generate synthetic data
    print("Generating synthetic UPI transaction data...")
    df = fraud_detector.generate_synthetic_data(10000)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
    
    # Preprocess data
    X, y, feature_cols = fraud_detector.preprocess_data(df)
    
    # If y has only one class (rare), adjust by forcing a tiny class presence - but synthetic generation should avoid this
    if y.nunique() < 2:
        raise RuntimeError("Generated labels have only one class. Regenerate data or check generation logic.")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train models
    print("\nTraining machine learning models...")
    fraud_detector.train_models(X_train, y_train)
    
    # Evaluate models
    print("\nEvaluating models...")
    results = fraud_detector.evaluate_models(X_test, y_test)
    
    # Plot feature importance (if available)
    fraud_detector.plot_feature_importance(feature_cols)
    
    # Initialize risk management system
    risk_system = RiskManagementSystem(fraud_detector)
    
    # Test real-time risk assessment on a few fraud transactions (if any exist)
    test_transactions = df[df['is_fraud'] == 1].head(5)
    if not test_transactions.empty:
        print("\nTesting real-time risk assessment...")
        risk_results = risk_system.real_time_monitoring_dashboard(test_transactions)
        print("\nReal-time Risk Assessment Results:")
        print(risk_results)
        
        # Display risk distribution
        plt.figure(figsize=(10, 6))
        risk_results['risk_level'].value_counts().plot(kind='bar')
        plt.title('Risk Level Distribution in Test Transactions')
        plt.xlabel('Risk Level')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("No positive fraud samples found to test real-time risk assessment.")
    
    return fraud_detector, risk_system


def demo_real_time_system(fraud_detector, risk_system, n=10):
    monitor = RealTimeUPIMonitor(fraud_detector, risk_system)
    monitor.start_monitoring()
    
    # Simulate real-time transactions
    for i in range(n):
        test_transaction = {
            'transaction_id': 10000 + i,
            'amount': float(np.random.exponential(500)),
            'time_of_day': int(np.random.randint(0, 24)),
            'day_of_week': int(np.random.randint(0, 7)),
            'user_age': int(np.random.randint(18, 70)),
            'user_income_bracket': 'Medium',
            'device_type': 'Mobile',
            'location_city': 'Mumbai',
            'transaction_type': 'P2P',
            'previous_chargebacks': int(np.random.poisson(0.1)),
            'account_age_days': int(np.random.randint(1, 3650)),
            'transaction_frequency_1h': int(np.random.poisson(1)),
            'transaction_frequency_24h': int(np.random.poisson(5)),
            'avg_transaction_amount': float(np.random.exponential(400)),
            'is_new_device': int(np.random.choice([0, 1], p=[0.9, 0.1])),
            'is_foreign_country': int(np.random.choice([0, 1], p=[0.99, 0.01])),
            'time_since_last_transaction': float(np.random.exponential(3600))
        }
        monitor.add_transaction(test_transaction)
        time.sleep(0.2)
    
    time.sleep(2)  # Allow processing to complete
    monitor.stop_monitoring()


if __name__ == "__main__":
    fd, rs = main()
    # Run a short demo of the real-time monitor
    demo_real_time_system(fd, rs, n=10)
