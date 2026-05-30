from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np

def evaluate_model(model, X_test, y_test):
    """
    Evaluates model performance and returns metrics.
    """
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "ROC-AUC": roc_auc_score(y_test, probs),
        "F1-Score": f1_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "Confusion Matrix": confusion_matrix(y_test, preds)
    }
    
    return metrics
