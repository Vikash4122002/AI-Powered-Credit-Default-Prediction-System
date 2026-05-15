"""
train_base_models.py
Trains RandomForest, XGBoost, LightGBM with Optuna hyperparameter tuning only
"""

import os
import joblib
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Create directory to save models
os.makedirs('models', exist_ok=True)

# Load preprocessed data
print("Loading data...")
X_train_resampled = joblib.load('data/processed/X_train.pkl')
y_train_resampled = joblib.load('data/processed/y_train.pkl')

X_test_scaled = joblib.load('data/processed/X_test.pkl')
y_test = joblib.load('data/processed/y_test.pkl')
print("Data loaded successfully")
print(f"Training data shape: {X_train_resampled.shape}\n")
print(f"Testing Shape  : {X_test_scaled.shape}")
# 1. RANDOM FOREST WITH OPTUNA
def objective_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'random_state': 42,
        'n_jobs': -1
    }
    model = RandomForestClassifier(**params)
    scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=5, scoring='roc_auc', n_jobs=-1)
    return scores.mean()
print("="*50)
print("Optuna Tuning: Random Forest")
print("="*50)
study_rf = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study_rf.optimize(objective_rf, n_trials=30, show_progress_bar=True)
print(f"\nBest params: {study_rf.best_params}")
print(f"Best CV ROC-AUC: {study_rf.best_value:.4f}")
best_params_rf = study_rf.best_params
best_params_rf['random_state'] = 42
best_params_rf['n_jobs'] = -1
rf = RandomForestClassifier(**best_params_rf)
rf.fit(X_train_resampled, y_train_resampled)
joblib.dump(rf, 'models/rf_model.pkl')
joblib.dump(study_rf, 'models/rf_study.pkl')
print("Random Forest saved\n")
# 2. XGBOOST WITH OPTUNA
def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_jobs': -1
    }
    model = xgb.XGBClassifier(**params)
    scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=5, scoring='roc_auc', n_jobs=-1)
    return scores.mean()
print("="*50)
print("Optuna Tuning: XGBoost")
print("="*50)
study_xgb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study_xgb.optimize(objective_xgb, n_trials=30, show_progress_bar=True)
print(f"\nBest params: {study_xgb.best_params}")
print(f"Best CV ROC-AUC: {study_xgb.best_value:.4f}")
best_params_xgb = study_xgb.best_params
best_params_xgb['eval_metric'] = 'logloss'
best_params_xgb['random_state'] = 42
best_params_xgb['n_jobs'] = -1
xgb_model = xgb.XGBClassifier(**best_params_xgb)
xgb_model.fit(X_train_resampled, y_train_resampled)
joblib.dump(xgb_model, 'models/xgb_model.pkl')
joblib.dump(study_xgb, 'models/xgb_study.pkl')
print("XGBoost saved\n")
# 3. LIGHTGBM WITH OPTUNA
def objective_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
        'random_state': 42,
        'verbose': -1
    }
    model = lgb.LGBMClassifier(**params)
    scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=5, scoring='roc_auc', n_jobs=-1)
    return scores.mean()
print("="*50)
print("Optuna Tuning: LightGBM")
print("="*50)
study_lgb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study_lgb.optimize(objective_lgb, n_trials=30, show_progress_bar=True)
print(f"\nBest params: {study_lgb.best_params}")
print(f"Best CV ROC-AUC: {study_lgb.best_value:.4f}")
best_params_lgb = study_lgb.best_params
best_params_lgb['random_state'] = 42
best_params_lgb['verbose'] = -1
lgbm = lgb.LGBMClassifier(**best_params_lgb)
lgbm.fit(X_train_resampled, y_train_resampled)
joblib.dump(lgbm, 'models/lgbm_model.pkl')
joblib.dump(study_lgb, 'models/lgbm_study.pkl')
print("LightGBM saved\n")

print("="*50)
print("All models trained and saved successfully!")
print("="*50)
# 4. QUICK VERIFICATION
print("="*50)
print("Quick Verification")
print("="*50)
sample = X_train_resampled[:5]
print("Random Forest predictions:", rf.predict(sample))
print("XGBoost predictions:", xgb_model.predict(sample))
print("LightGBM predictions:", lgbm.predict(sample))
# 5. EVALUATION FUNCTION
from sklearn.metrics import recall_score, roc_auc_score, precision_score, f1_score, accuracy_score
def evaluate(model, name, threshold=0.3):
    """Evaluate model with custom threshold"""
    try:
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_prob > threshold).astype(int)
        
        print(f"\n{name} (Threshold = {threshold})")
        print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred):.4f}")
        print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
        print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
        print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    except Exception as e:
        print(f"\n{name} - Error in evaluation: {e}")
# 6. EVALUATE ALL MODELS
print("\n" + "="*50)
print("pEVALUATION RESULTS")
print("="*50)
evaluate(rf, "Random Forest (Optuna)", threshold=0.3)
evaluate(xgb_model, "XGBoost (Optuna)", threshold=0.3)
evaluate(lgbm, "LightGBM (Optuna)", threshold=0.3)
print("\n" + "="*50)
print("All models trained and evaluated successfully!")
print("="*50)
# Print summary of saved files
print("\nSaved files in 'models/' directory:")
print("   - rf_model.pkl (Random Forest with Optuna)")
print("   - rf_study.pkl (Optuna study)")
print("   - xgb_model.pkl (XGBoost with Optuna)")
print("   - xgb_study.pkl (Optuna study)")
print("   - lgbm_model.pkl (LightGBM with Optuna)")
print("   - lgbm_study.pkl (Optuna study)")