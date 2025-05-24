import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (roc_curve, auc, accuracy_score,
                             precision_score, recall_score, f1_score)
from sklearn.model_selection import train_test_split
from config import AUDIO_FEATURES, RANDOM_STATE

def evaluate_models(X_test, y_test):
    """Evaluate models and generate reports"""
    # Create reports directory if needed
    os.makedirs('reports/metrics', exist_ok=True)
    
    models = {
        'logreg': joblib.load('models/logreg.pkl'),
        'randomforest': joblib.load('models/randomforest.pkl'),
        'svm': joblib.load('models/svm.pkl')
    }
    scaler = joblib.load('models/scaler.pkl')
    X_test_scaled = scaler.transform(X_test)
    
    metrics = {}
    plt.figure(figsize=(10, 6))
    
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        
        # Handle models without predict_proba
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_scaled)[:,1]
        else:
            y_proba = model.decision_function(X_test_scaled)
            y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
        
        metrics[name] = {
            'roc_auc': auc(*roc_curve(y_test, y_proba)[:2]),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {metrics[name]["roc_auc"]:.2f})')

    # Save ROC plot
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Model ROC Curves')
    plt.legend()
    plt.savefig('reports/metrics/roc_curves.png')
    plt.close()
    
    # Save metrics
    pd.DataFrame(metrics).T.to_csv('reports/metrics/performance.csv')
    
    # Feature importance
    try:
        rf = models['randomforest']
        importance = pd.DataFrame({
            'Feature': AUDIO_FEATURES,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 4))
        sns.barplot(x='Importance', y='Feature', data=importance)
        plt.title('Random Forest Feature Importance')
        plt.savefig('reports/metrics/feature_importance.png')
        plt.close()
    except Exception as e:
        print(f"Feature importance error: {e}")

if __name__ == "__main__":
    data = joblib.load('models/processed_data.pkl')
    X, y = data[AUDIO_FEATURES], data['liked']
    _, X_test, _, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        stratify=y, 
        random_state=RANDOM_STATE
    )
    evaluate_models(X_test, y_test)