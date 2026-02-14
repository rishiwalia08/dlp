"""
Decision Agent for autonomous action execution.

This module implements rule-based decision logic for IDS responses.
No LLM dependencies - pure Python logic.
"""


class DecisionAgent:
    """
    Autonomous decision agent for IDS response actions.
    
    Uses rule-based logic based on:
    - Attack type
    - Confidence
    - Risk score
    - Severity
    """
    
    def __init__(self):
        """Initialize decision agent with rule-based logic only."""
        print("\\n=== Initializing Decision Agent ===")
        print("Using rule-based decision logic")
    
    def decide_action(self, attack_type, confidence, risk_score, severity, llm_explanation=None):
        """
        Determine appropriate action based on system outputs.
        
        Rule-based decision logic:
        - BENIGN + Low risk: No action (log only)
        - Medium risk: Enhanced monitoring + alert
        - High risk: Simulated mitigation (rate limiting) + alert
        - Critical risk: Simulated block (quarantine IP) + escalate
        
        Args:
            attack_type: Predicted attack type
            confidence: Model confidence (0-1)
            risk_score: Computed risk score (0-1)
            severity: Risk severity category
            llm_explanation: Optional LLM explanation dict
            
        Returns:
            dict: Decision and action details
        """
        # Rule-based decision logic
        if attack_type.upper() in ['BENIGN', 'NORMAL'] and risk_score < 0.3:
            action = "No action - log only"
            action_type = "LOG"
            details = "Traffic appears benign with low risk score. Logging for audit trail."
        
        elif risk_score < 0.6:
            action = "Enable enhanced monitoring + alert SOC"
            action_type = "MONITOR"
            details = "Medium risk detected. Increasing monitoring granularity and alerting security operations center."
        
        elif risk_score < 0.85:
            action = "Simulated mitigation: rate limiting + alert"
            action_type = "MITIGATE"
            details = "High risk attack detected. Applying rate limiting to source and alerting incident response team."
        
        else:  # risk_score >= 0.85
            action = "Simulated block: quarantine IP + escalate"
            action_type = "BLOCK"
            details = "Critical risk attack detected. Quarantining source IP and escalating to senior security analyst."
        
        # Execute simulated action
        execution_log = self._execute_action(action_type, attack_type, risk_score)
        
        decision = {
            'action': action,
            'action_type': action_type,
            'rationale': details,
            'execution_log': execution_log,
            'attack_type': attack_type,
            'confidence': confidence,
            'risk_score': risk_score,
            'severity': severity
        }
        
        return decision
    
    def _execute_action(self, action_type, attack_type, risk_score):
        """
        Execute simulated action.
        
        Args:
            action_type: Type of action (LOG, MONITOR, MITIGATE, BLOCK)
            attack_type: Attack type
            risk_score: Risk score
            
        Returns:
            list: Execution log entries
        """
        log = []
        
        if action_type == "LOG":
            log.append(f"[LOG] Recorded {attack_type} detection with risk score {risk_score:.4f}")
            log.append("[LOG] Event logged to SIEM system")
        
        elif action_type == "MONITOR":
            log.append(f"[MONITOR] Enhanced monitoring enabled for {attack_type}")
            log.append("[MONITOR] Alert sent to SOC dashboard")
            log.append("[MONITOR] Increased packet capture granularity")
        
        elif action_type == "MITIGATE":
            log.append(f"[MITIGATE] Rate limiting applied for {attack_type}")
            log.append("[MITIGATE] Source IP added to watchlist")
            log.append("[MITIGATE] Incident response team notified")
            log.append("[MITIGATE] Traffic pattern analysis initiated")
        
        elif action_type == "BLOCK":
            log.append(f"[BLOCK] Critical {attack_type} attack blocked")
            log.append("[BLOCK] Source IP quarantined in firewall")
            log.append("[BLOCK] All connections from source terminated")
            log.append("[BLOCK] Senior security analyst alerted")
            log.append("[BLOCK] Forensic data collection initiated")
        
        return log
    
    def batch_decide(self, predictions_data):
        """
        Make decisions for multiple predictions.
        
        Args:
            predictions_data: List of dicts with prediction info
            
        Returns:
            list: List of decision dictionaries
        """
        print(f"\\nMaking decisions for {len(predictions_data)} predictions...")
        
        decisions = []
        for i, pred_data in enumerate(predictions_data):
            decision = self.decide_action(
                attack_type=pred_data['attack_type'],
                confidence=pred_data['confidence'],
                risk_score=pred_data['risk_score'],
                severity=pred_data['severity'],
                llm_explanation=pred_data.get('llm_explanation')
            )
            decisions.append(decision)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(predictions_data)} decisions")
        
        print("Decision making complete")
        return decisions
    
    def print_decision_summary(self, decision):
        """
        Print a formatted decision summary.
        
        Args:
            decision: Decision dictionary
        """
        print("\\n" + "="*60)
        print("DECISION AGENT OUTPUT")
        print("="*60)
        print(f"Attack Type: {decision['attack_type']}")
        print(f"Confidence: {decision['confidence']:.2%}")
        print(f"Risk Score: {decision['risk_score']:.4f}")
        print(f"Severity: {decision['severity']}")
        print(f"\\nAction: {decision['action']}")
        print(f"Action Type: {decision['action_type']}")
        print(f"\\nRationale: {decision['rationale']}")
        print("\\nExecution Log:")
        for log_entry in decision['execution_log']:
            print(f"  {log_entry}")
        print("="*60)


def create_decision_agent():
    """
    Create decision agent with rule-based logic.
    
    Returns:
        DecisionAgent instance
    """
    return DecisionAgent()


if __name__ == "__main__":
    # Test decision agent
    print("Testing Decision Agent...")
    
    agent = create_decision_agent()
    
    # Test cases
    test_cases = [
        {
            'attack_type': 'BENIGN',
            'confidence': 0.95,
            'risk_score': 0.15,
            'severity': 'Low'
        },
        {
            'attack_type': 'Probe',
            'confidence': 0.78,
            'risk_score': 0.45,
            'severity': 'Medium'
        },
        {
            'attack_type': 'DoS',
            'confidence': 0.92,
            'risk_score': 0.75,
            'severity': 'High'
        },
        {
            'attack_type': 'DDoS',
            'confidence': 0.96,
            'risk_score': 0.92,
            'severity': 'Critical'
        }
    ]
    
    print("\\n=== Testing Decision Logic ===")
    for test in test_cases:
        decision = agent.decide_action(**test)
        agent.print_decision_summary(decision)
