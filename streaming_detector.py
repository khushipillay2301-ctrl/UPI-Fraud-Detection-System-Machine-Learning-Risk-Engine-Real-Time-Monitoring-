# streaming_detector.py
import threading
import time
from collections import deque
import json
import numpy as np
import pandas as pd

# NOTE: If main() is defined in another file, import it:
# from main import main

class RealTimeUPIMonitor:
    def __init__(self, fraud_detector, risk_system):
        self.fraud_detector = fraud_detector
        self.risk_system = risk_system
        self.transaction_queue = deque()
        self.alert_queue = deque()
        self.running = False
        
    def add_transaction(self, transaction_data):
        """Add transaction to processing queue"""
        self.transaction_queue.append(transaction_data)
        
    def _extract_risk(self, risk_result):
        """Ensure risk_result is a dictionary, not a DataFrame"""
        if isinstance(risk_result, pd.DataFrame):
            risk_result = risk_result.iloc[0].to_dict()
        return risk_result
        
    def process_transactions(self):
        """Process transactions in real-time"""
        while self.running:
            if self.transaction_queue:
                transaction = self.transaction_queue.popleft()

                # Risk assessment
                risk_result = self.risk_system.assess_risk(
                    pd.DataFrame([transaction])
                )
                risk_result = self._extract_risk(risk_result)

                # Ensure default keys exist
                risk_result.setdefault('risk_level', "LOW")
                risk_result.setdefault('fraud_probability', 0.0)
                risk_result.setdefault('action', "ALLOW")

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
        
        message = alert_message.get(risk_result['risk_level'], "Unknown risk level")
        print(f"ALERT: {message}")
    
    def log_transaction(self, transaction, risk_result):
        """Log transaction and risk assessment"""
        log_entry = {
            'timestamp': time.time(),
            'transaction_id': transaction.get('transaction_id'),
            'amount': transaction.get('amount'),
            'risk_level': risk_result.get('risk_level'),
            'fraud_probability': risk_result.get('fraud_probability'),
            'action_taken': risk_result.get('action')
        }
        
        print(f"LOG: {json.dumps(log_entry)}")
    
    def start_monitoring(self):
        """Start the real-time monitoring system"""
        self.running = True
        monitor_thread = threading.Thread(target=self.process_transactions)
        monitor_thread.daemon = True
        monitor_thread.start()
        print("Real-time UPI monitoring started...")
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.running = False
        print("Real-time UPI monitoring stopped.")

# Example usage
def demo_real_time_system():
    try:
        # Replace this if main() is imported from another file
        fraud_detector, risk_system = main()
    except NameError:
        print("ERROR: main() is not defined. Import it properly.")
        return

    monitor = RealTimeUPIMonitor(fraud_detector, risk_system)
    
    monitor.start_monitoring()
    
    # Simulate real-time transactions
    for i in range(10):
        test_transaction = {
            'transaction_id': 10000 + i,
            'amount': float(np.random.exponential(500)),
            'time_of_day': int(np.random.randint(0, 24)),
            'is_new_device': int(np.random.choice([0, 1], p=[0.9, 0.1])),
            'is_foreign_country': 0,
            'transaction_frequency_1h': int(np.random.poisson(1)),
        }
        monitor.add_transaction(test_transaction)
        time.sleep(0.5)
    
    time.sleep(2)
    monitor.stop_monitoring()

if __name__ == "__main__":
    demo_real_time_system()
