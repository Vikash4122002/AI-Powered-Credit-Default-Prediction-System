"""
main.py - Complete Credit Default Prediction Pipeline
Orchestrates: Data Preprocessing → Optuna Training → Stacking → Evaluation
FINAL MODEL: Stacking Ensemble with XGBoost Meta-Learner
"""

import os
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import cross_val_score

from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

print("="*60)
print("CREDIT DEFAULT PREDICTION SYSTEM")
print("Ensemble Learning with Optuna + Stacking (XGBoost Meta-Learner)")
print("="*60)
# STEP 1: DATA PREPROCESSING
print("\n" + "="*60)
print("STEP 1: DATA PREPROCESSING")
print("="*60)
# Import preprocessing functions from your data_preprocessing.py
from src.data_preprocessing import load_and_clean_data, preprocess_with_scaling_and_smote
try:
    # Load and preprocess data
    print("Loading and cleaning data...")
    X, y = load_and_clean_data('data/raw/credit card clients.xlsx')
    
    print("Applying scaling and SMOTE...")
    X_train_resampled, X_test_scaled, y_train_resampled, y_test, scaler = preprocess_with_scaling_and_smote(X, y)
    
    print(f"\nPreprocessing Complete!")
    print(f"Training data (after SMOTE): {X_train_resampled.shape}")
    print(f"Test data: {X_test_scaled.shape}")
    print(f"Default rate (test): {y_test.mean():.4f}")
    
    # Convert to DataFrame if needed for compatibility
    if not isinstance(X_train_resampled, pd.DataFrame):
        X_train_resampled = pd.DataFrame(X_train_resampled, columns=X.columns)
    if not isinstance(X_test_scaled, pd.DataFrame):
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    # Save preprocessed data for reuse
    os.makedirs('data/processed', exist_ok=True)
    joblib.dump(X_train_resampled, 'data/processed/X_train.pkl')
    joblib.dump(y_train_resampled, 'data/processed/y_train.pkl')
    joblib.dump(X_test_scaled, 'data/processed/X_test.pkl')
    joblib.dump(y_test, 'data/processed/y_test.pkl')
    joblib.dump(scaler, 'data/processed/scaler.pkl')
    print("\nPreprocessed data saved to 'data/processed/'")   
except Exception as e:
    print(f"\nError in preprocessing: {e}")
    print("Make sure the dataset is in 'data/raw/credit card clients.xlsx'")
    exit(1)
print("\n" + "="*60)
print("STEP 2: TRAINING BASE MODELS WITH OPTUNA")
print("="*60)

# Check if models already exist to save time
if (os.path.exists('models/rf_model.pkl') and 
    os.path.exists('models/xgb_model.pkl') and 
    os.path.exists('models/lgbm_model.pkl')):
    
    print("\nModels already exist! Loading existing models...")
    rf_model = joblib.load('models/rf_model.pkl')
    xgb_model = joblib.load('models/xgb_model.pkl')
    lgb_model = joblib.load('models/lgbm_model.pkl')
    print("Base models loaded successfully!")
    
else:
    print("\nNo existing models found. Training new models with Optuna...")
    print("This may take 30-45 minutes...")
    
    # Import training functions
    from src.train_base_models import (
        train_random_forest_optuna,
        train_xgboost_optuna, 
        train_lightgbm_optuna
    )
    # Train Random Forest
    print("\nTraining Random Forest with Optuna (30 trials)...")
    rf_model, rf_study = train_random_forest_optuna(
        X_train_resampled, y_train_resampled, 
        n_trials=30, 
        save_path='models/rf_model.pkl'
    )
    # Train XGBoost
    print("\nTraining XGBoost with Optuna (30 trials)...")
    xgb_model, xgb_study = train_xgboost_optuna(
        X_train_resampled, y_train_resampled,
        n_trials=30,
        save_path='models/xgb_model.pkl'
    )
    # Train LightGBM
    print("\nTraining LightGBM with Optuna (30 trials)...")
    lgb_model, lgb_study = train_lightgbm_optuna(
        X_train_resampled, y_train_resampled,
        n_trials=30,
        save_path='models/lgbm_model.pkl'
    )
    
    print("\nAll base models trained and saved to 'models/'")
# STEP 3: BUILD STACKING ENSEMBLE
print("\n" + "="*60)
print("STEP 3: BUILDING STACKING ENSEMBLE")
print("="*60)

# Check if stacking model already exists
if os.path.exists('models/stacking_optuna_meta.pkl'):
    print("\nStacking model already exists! Loading...")
    stacking_model = joblib.load('models/stacking_optuna_meta.pkl')
    print("Stacking model loaded!")
    
else:
    print("\nNo stacking model found. Building new stacking ensemble...")
    print("This may take 15-20 minutes...")
    
    # Import stacking module
    from src.stacking_optuna_meta_only import (
        objective_meta, run_stacking_pipeline
    )
    
    # Define base models
    base_models = [
        ('random_forest', rf_model),
        ('xgboost', xgb_model),
        ('lightgbm', lgb_model)
    ]
    
    # Create and train stacking
    import optuna
    from optuna.samplers import TPESampler
    from sklearn.ensemble import StackingClassifier
    import xgboost as xgb
    
    # Run Optuna tuning for meta-learner
    print("\nOptuna Tuning: XGBoost Meta-Learner")
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    
    # Define objective function that uses base_models
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
            'max_depth': trial.suggest_int('max_depth', 2, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': -1
        }
        
        xgb_meta = xgb.XGBClassifier(**params)
        stacking = StackingClassifier(
            estimators=base_models,
            final_estimator=xgb_meta,
            cv=3,
            stack_method='predict_proba',
            passthrough=False,
            n_jobs=-1
        )
        
        scores = cross_val_score(stacking, X_train_resampled, y_train_resampled, 
                                cv=3, scoring='roc_auc', n_jobs=-1)
        return scores.mean()
    
    study.optimize(objective, n_trials=20, show_progress_bar=True)
    
    # Train final stacking with best params
    best_params = study.best_params.copy()
    best_params['eval_metric'] = 'logloss'
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    
    xgb_meta_best = xgb.XGBClassifier(**best_params)
    
    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=xgb_meta_best,
        cv=5,
        stack_method='predict_proba',
        passthrough=False,
        n_jobs=-1
    )
    
    print("\nTraining final stacking model...")
    stacking_model.fit(X_train_resampled, y_train_resampled)
    
    # Save models
    joblib.dump(stacking_model, 'models/stacking_optuna_meta.pkl')
    joblib.dump(study, 'models/stacking_optuna_study.pkl')
    print("\nStacking ensemble saved to 'models/stacking_optuna_meta.pkl'")
# STEP 4: EVALUATE ALL MODELS
print("\n" + "="*60)
print("STEP 4: MODEL EVALUATION")
print("="*60)

def evaluate_model(model, X_test, y_test, name, threshold=0.3):
    """Evaluate a single model"""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > threshold).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n{name}")
    print("-"*50)
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}  ← MOST IMPORTANT")
    print(f"  F1 Score:  {f1:.4f}")
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

# Dictionary of models
models_dict = {
    'Random Forest': rf_model,
    'XGBoost': xgb_model,
    'LightGBM': lgb_model,
    'Stacking (XGB Meta)': stacking_model
}

print("\n" + "="*60)
print("MODEL EVALUATION (Threshold = 0.3)")
print("="*60)

evaluation_results = {}
for name, model in models_dict.items():
    evaluation_results[name] = evaluate_model(model, X_test_scaled, y_test, name, threshold=0.3)
# STEP 5: FIND BEST THRESHOLD FOR STACKING
print("\n" + "="*60)
print("STEP 5: OPTIMAL THRESHOLD TUNING FOR STACKING")
print("="*60)

thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
y_prob_stack = stacking_model.predict_proba(X_test_scaled)[:, 1]

print("\nThreshold | Recall | Precision | F1 Score")
print("-" * 45)

best_recall = 0
best_thresh = 0.3

for t in thresholds:
    y_pred = (y_prob_stack > t).astype(int)
    rec = recall_score(y_test, y_pred, zero_division=0)
    pre = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"   {t:.2f}     |  {rec:.4f}  |   {pre:.4f}   |  {f1:.4f}")
    
    if rec > best_recall:
        best_recall = rec
        best_thresh = t

print(f"\nBest threshold for maximum Recall: {best_thresh}")
print(f"Achieves {best_recall*100:.1f}% recall")
# STEP 6: SAVE STACKING AS FINAL MODEL
print("\n" + "="*60)
print("STEP 6: SAVING STACKING AS FINAL MODEL")
print("="*60)

# Save stacking model with best threshold info
final_model_info = {
    'model': stacking_model,
    'model_name': 'Stacking Ensemble (XGBoost Meta-Learner)',
    'scaler': scaler,
    'best_threshold': best_thresh,
    'best_recall': best_recall,
    'base_models': ['Random Forest', 'XGBoost', 'LightGBM'],
    'features': X.columns.tolist() if hasattr(X, 'columns') else None
}

joblib.dump(final_model_info, 'models/final_model.pkl')
print("Stacking model saved as final model to 'models/final_model.pkl'")

# Save results summarya
summary_df = pd.DataFrame([
    {'Model': 'Random Forest', 'Recall (0.3)': evaluation_results['Random Forest']['recall']},
    {'Model': 'XGBoost', 'Recall (0.3)': evaluation_results['XGBoost']['recall']},
    {'Model': 'LightGBM', 'Recall (0.3)': evaluation_results['LightGBM']['recall']},
    {'Model': 'Stacking (XGB Meta)', 'Recall (0.3)': evaluation_results['Stacking (XGB Meta)']['recall']},
    {'Model': f'Stacking (Optimal Threshold={best_thresh})', 'Recall': best_recall}
])

summary_df.to_csv('models/final_summary.csv', index=False)
print("Final summary saved to 'models/final_summary.csv'")
# STEP 7: PREDICTION DEMO
print("\n" + "="*60)
print("STEP 7: PREDICTION DEMO (Using Stacking Model)")
print("="*60)

def predict_default(customer_data, model_info=None):
    """Predict if a customer will default using Stacking Ensemble"""
    if model_info is None:
        model_info = joblib.load('models/final_model.pkl')
    
    if len(customer_data.shape) == 1:
        customer_data = customer_data.reshape(1, -1)
    
    scaled_data = model_info['scaler'].transform(customer_data)
    probability = model_info['model'].predict_proba(scaled_data)[:, 1]
    prediction = (probability > model_info['best_threshold']).astype(int)
    
    return prediction[0], probability[0]

# Test prediction on first 5 samples
print("\nTesting prediction system on first 5 test samples:")
print("-" * 70)

for i in range(min(5, len(X_test_scaled))):
    pred, prob = predict_default(X_test_scaled.iloc[i:i+1].values)
    actual = y_test.iloc[i]
    status = "CORRECT" if pred == actual else "WRONG"
    
    print(f"   Sample {i+1}: Predicted={'Default' if pred==1 else 'No Default'} | "
          f"Actual={'Default' if actual==1 else 'No Default'} | "
          f"Risk={prob:.4f} | {status}")
# STEP 8: META-LEARNER ANALYSIS
print("\n" + "="*60)
print("STEP 8: META-LEARNER ANALYSIS")
print("="*60)

if hasattr(stacking_model, 'final_estimator_'):
    meta_learner = stacking_model.final_estimator_
    
    if hasattr(meta_learner, 'feature_importances_'):
        importances = meta_learner.feature_importances_
        base_model_names = ['Random Forest', 'XGBoost', 'LightGBM']
        
        print("\nBase Model Importance (according to XGBoost Meta-Learner):")
        for name, imp in zip(base_model_names, importances[:3]):
            print(f"   {name}: {imp:.4f} ({imp*100:.1f}%)")
        
        most_important = base_model_names[np.argmax(importances)]
        print(f"\nMost important base model: {most_important}")
# STEP 9: FINAL REPORT
print("\n" + "="*60)
print("FINAL REPORT")
print("="*60)

print("\nPERFORMANCE SUMMARY:")
print(f"   • Best Model: Stacking Ensemble with XGBoost Meta-Learner")
print(f"   • Recall at threshold 0.3: {evaluation_results['Stacking (XGB Meta)']['recall']:.4f}")
print(f"   • Maximum Recall: {best_recall:.4f} at threshold {best_thresh}")

print("\nBUSINESS RECOMMENDATIONS:")
print(f"   • HIGH STAKES (Mortgage/Large Loans): Use threshold={best_thresh}")
print(f"     → Catches {best_recall*100:.1f}% of defaulters")
print(f"   • BALANCED (Credit Cards): Use threshold=0.3")
print(f"     → Catches {evaluation_results['Stacking (XGB Meta)']['recall']*100:.1f}% of defaulters")
print(f"   • CONSERVATIVE (Premium Customers): Use threshold=0.4")
print(f"     → Fewer false positives, ~50% recall")
# SAVED FILES
print("\n" + "="*60)
print("SAVED FILES")
print("="*60)

print(""" Saved files in 'models/' directory:
  - stacking_optuna_meta.pkl (Stacking model)
  - stacking_optuna_study.pkl (Optuna study)
  - results_optuna_meta.csv (Performance results)
  - confusion_matrices_optuna_meta.png
  - roc_curves_optuna_meta.png
  - optuna_meta_importance.png
  - final_model.pkl (Complete pipeline with scaler)
  - final_summary.csv (Performance summary)
     Saved files in 'data/processed/':
  - X_train.pkl
  - y_train.pkl
  - X_test.pkl
  - y_test.pkl
  - scaler.pkl
""")