"""
evaluate.py - Complete evaluation module for credit default prediction models
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
# 1. SINGLE MODEL EVALUATION
def evaluate_model(model, X_test, y_test, name, threshold=0.3):
    """Evaluate a single model"""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > threshold).astype(int)
    
    metrics = {
        'model': name,
        'threshold': threshold,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'y_prob': y_prob,
        'y_pred': y_pred,
        'cm': confusion_matrix(y_test, y_pred)
    }
    
    tn, fp, fn, tp = metrics['cm'].ravel()
    metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = tn, fp, fn, tp
    
    return metrics


def print_evaluation(metrics):
    """Print evaluation results"""
    print(f"\n{'='*50}")
    print(f"{metrics['model']}")
    print(f"{'='*50}")
    print(f"Threshold:        {metrics['threshold']}")
    print(f"Accuracy:         {metrics['accuracy']:.4f}")
    print(f"Precision:        {metrics['precision']:.4f}")
    print(f"Recall:           {metrics['recall']:.4f}  ← MOST IMPORTANT")
    print(f"F1 Score:         {metrics['f1']:.4f}")
    print(f"ROC-AUC:          {metrics['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {metrics['tn']}  |  FP: {metrics['fp']}")
    print(f"  FN: {metrics['fn']}  |  TP: {metrics['tp']}")

# 2. COMPARE MULTIPLE MODELS

def compare_models(models_dict, X_test, y_test, threshold=0.3):
    """Compare multiple models"""
    print("\n" + "="*60)
    print(f"MODEL COMPARISON (Threshold = {threshold})")
    print("="*60)
    
    results = []
    all_metrics = {}
    
    for name, model in models_dict.items():
        metrics = evaluate_model(model, X_test, y_test, name, threshold)
        print_evaluation(metrics)
        results.append({
            'Model': name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1 Score': metrics['f1'],
            'ROC-AUC': metrics['roc_auc']
        })
        all_metrics[name] = metrics
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('Recall', ascending=False)
    
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON TABLE")
    print("="*60)
    print(df_results.to_string(index=False))
    
    return df_results, all_metrics

# 3. PLOT CONFUSION MATRICES
def plot_confusion_matrices(all_metrics, X_test, y_test, save_path='models/confusion_matrices.png'):
    """Plot confusion matrices for all models"""
    n_models = len(all_metrics)
    n_cols = 2
    n_rows = (n_models + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5*n_rows))
    
    if n_models == 1:
        axes = [axes]
    else:
        axes = np.array(axes).reshape(-1)
    
    for idx, (name, metrics) in enumerate(all_metrics.items()):
        cm = metrics['cm']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['No Default', 'Default'],
                    yticklabels=['No Default', 'Default'])
        axes[idx].set_title(f'{name}\nRecall: {metrics["recall"]:.4f}', fontsize=12)
        axes[idx].set_ylabel('Actual')
        axes[idx].set_xlabel('Predicted')
    
    for idx in range(len(all_metrics), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Confusion Matrices - All Models', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrices saved to {save_path}")
# 4. PLOT ROC CURVES
def plot_roc_curves(all_metrics, X_test, y_test, save_path='models/roc_curves.png'):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (name, metrics) in enumerate(all_metrics.items()):
        y_prob = metrics['y_prob']
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = metrics['roc_auc']
        plt.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2,
                label=f'{name} (AUC = {auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"ROC curves saved to {save_path}")

# 5. FIND BEST THRESHOLD
def find_best_threshold(model, X_test, y_test, model_name="Model"):
    """Find optimal probability threshold to maximize recall"""
    print("\n" + "="*60)
    print(f"🎯 Finding Best Threshold for {model_name}")
    print("="*60)
    
    y_prob = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.05, 0.65, 0.05)
    
    print("\nThreshold | Recall | Precision | F1 Score")
    print("-" * 45)
    
    best_recall = 0
    best_threshold = 0.3
    best_f1 = 0
    
    for t in thresholds:
        y_pred = (y_prob > t).astype(int)
        rec = recall_score(y_test, y_pred, zero_division=0)
        pre = precision_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print(f"   {t:.2f}     |  {rec:.4f}  |   {pre:.4f}   |  {f1:.4f}")
        
        if rec > best_recall:
            best_recall = rec
            best_threshold = t
        if f1 > best_f1:
            best_f1 = f1
    
    print(f"\nBest threshold for Recall: {best_threshold}")
    print(f"Achieves recall: {best_recall:.4f}")
    print(f"Best F1 Score achievable: {best_f1:.4f}")
    
    return best_threshold, best_recall

# 6. CLASSIFICATION REPORT
def print_classification_report(model, X_test, y_test, model_name):
    """Print classification report"""
    y_pred = model.predict(X_test)
    
    print(f"\n{'='*50}")
    print(f"Classification Report - {model_name}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, 
                                target_names=['No Default', 'Default'],
                                zero_division=0))
# 7. MAIN EVALUATION PIPELINE
def run_evaluation_pipeline(models_dict, X_test, y_test, threshold=0.3):
    """
    Complete evaluation pipeline
    
    Parameters:
    -----------
    models_dict: dict of {'model_name': model_object}
    X_test, y_test: test data
    threshold: default threshold for classification
    
    Returns:
    --------
    df_results: DataFrame with all metrics
    all_metrics: dict of all metrics
    best_threshold: optimal threshold for best model
    """
    print("EVALUATION PIPELINE")
    print("*"*20)
    
    # Compare all models
    df_results, all_metrics = compare_models(models_dict, X_test, y_test, threshold)
    
    # Save results
    os.makedirs('models', exist_ok=True)
    df_results.to_csv('models/evaluation_results.csv', index=False)
    print(f"\nResults saved to 'models/evaluation_results.csv'")
    
    # Plot confusion matrices
    plot_confusion_matrices(all_metrics, X_test, y_test)
    
    # Plot ROC curves
    plot_roc_curves(all_metrics, X_test, y_test)
    
    # Find best threshold for best model
    best_model_name = df_results.iloc[0]['Model']
    best_model = models_dict[best_model_name]
    best_threshold, best_recall = find_best_threshold(best_model, X_test, y_test, best_model_name)
    
    # Evaluate best model with optimal threshold
    print("\n" + "="*60)
    print(f"{best_model_name} WITH OPTIMAL THRESHOLD")
    print("="*60)
    optimal_metrics = evaluate_model(best_model, X_test, y_test, 
                                     f"{best_model_name} (Optimal)", best_threshold)
    print_evaluation(optimal_metrics)
    
    # Print classification report for best model
    print_classification_report(best_model, X_test, y_test, best_model_name)
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Best Model by Recall: {best_model_name}")
    print(f"  Recall (threshold={threshold}): {df_results.iloc[0]['Recall']:.4f}")
    print(f"  ROC-AUC: {df_results.iloc[0]['ROC-AUC']:.4f}")
    print(f"  Optimal threshold: {best_threshold} (Recall: {best_recall:.4f})")
    
    print("\nEvaluation complete!")
    
    return df_results, all_metrics, best_threshold
# 8. QUICK EVALUATION
def quick_evaluate(model, X_test, y_test, name, threshold=0.3):
    """Quick evaluation - returns metrics dict only"""
    metrics = evaluate_model(model, X_test, y_test, name, threshold)
    return {
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'roc_auc': metrics['roc_auc'],
        'confusion_matrix': metrics['cm']
    }
# 9. TEST SCRIPT (when run directly)

if __name__ == "__main__":
    print("="*60)
    print("EVALUATION MODULE")
    print("="*60)
    
    # Check if test data exists
    if os.path.exists('data/processed/X_test.pkl') and os.path.exists('data/processed/y_test.pkl'):
        print("\nTest data found! Running evaluation...")
        
        # Load data
        X_test = joblib.load('data/processed/X_test.pkl')
        y_test = joblib.load('data/processed/y_test.pkl')
        
        # Load models if they exist
        models = {}
        
        if os.path.exists('models/rf_model.pkl'):
            models['Random Forest'] = joblib.load('models/rf_model.pkl')
            print("Random Forest loaded")
        
        if os.path.exists('models/xgb_model.pkl'):
            models['XGBoost'] = joblib.load('models/xgb_model.pkl')
            print("XGBoost loaded")
        
        if os.path.exists('models/lgbm_model.pkl'):
            models['LightGBM'] = joblib.load('models/lgbm_model.pkl')
            print("LightGBM loaded")
        
        if os.path.exists('models/stacking_optuna_meta.pkl'):
            models['Stacking'] = joblib.load('models/stacking_optuna_meta.pkl')
            print("Stacking loaded")
        
        if models:
            # Run evaluation
            df_results, all_metrics, best_thresh = run_evaluation_pipeline(
                models, X_test, y_test, threshold=0.3
            )
        else:
            print("\nNo models found. Train models first using train_base_models.py")
    else:
        print("\nTest data not found. Run data preprocessing first.")
        print("\nTo use this module, import it in your script:")
        print("")
        print("    from src.evaluate import run_evaluation_pipeline")
        print("")
        print("    # After training your models")
        print("    models_dict = {")
        print("        'Random Forest': rf_model,")
        print("        'XGBoost': xgb_model,")
        print("        'LightGBM': lgb_model,")
        print("        'Stacking': stacking_model")
        print("    }")
        print("")
        print("    # Run evaluation")
        print("    df_results, all_metrics, best_thresh = run_evaluation_pipeline(")
        print("        models_dict, X_test, y_test, threshold=0.3")
        print("    )")
    
    print("\n" + "="*60)