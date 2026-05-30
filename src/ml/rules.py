def generate_business_rules(row_data: dict, risk_band: str, top_features: dict) -> list:
    """
    Translates model output and feature impacts into business-readable rules.
    """
    rules = []
    rules.append(f"IF Risk Band is {risk_band}")
    
    for _, row in top_features.iterrows():
        feature = row['Feature']
        impact = row['Impact']
        
        # Translate feature names and impacts into readable conditions
        direction = "increases" if impact > 0 else "decreases"
        
        if "AMT_INCOME" in feature:
            rules.append(f"AND Income level {direction} risk")
        elif "EXT_SOURCE" in feature:
            rules.append(f"AND External credit score {direction} risk")
        elif "DAYS_EMPLOYED" in feature:
            rules.append(f"AND Employment duration {direction} risk")
        elif "DAYS_BIRTH" in feature:
            rules.append(f"AND Applicant age {direction} risk")
        else:
            rules.append(f"AND {feature} {direction} risk")
            
    rules.append(f"THEN Customer is flagged as {risk_band} RISK")
    
    return rules
