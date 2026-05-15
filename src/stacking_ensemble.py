"""
stacking_optuna_meta_only.py

Stacking ensemble with:
- Random Forest
- XGBoost
- LightGBM

Meta Learner:
- Optuna Tuned XGBoost
"""

import os
import warnings
import joblib
import optuna 
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb

from optuna.samplers import TPESampler

from sklearn.ensemble import StackingClassifier

from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold
)

from sklearn.metrics import(accuracy_score,recall_score,precision_score,f1_score,roc_auc_score,confusion_matrix,roc_curve)

warnings.filterwarnings('ignore')


os.makedirs("models", exist_ok=True)


print("=" * 60)
print("LOADING DATA")
print("=" * 60)

X_train = joblib.load(
    'data/processed/X_train.pkl'
)

y_train = joblib.load(
    'data/processed/y_train.pkl'
)

X_test = joblib.load(
    'data/processed/X_test.pkl'
)

y_test = joblib.load(
    'data/processed/y_test.pkl'
)

print(f"\nTrain Shape : {X_train.shape}")
print(f"Test Shape  : {X_test.shape}")

print(f"\nDefault Rate : {y_test.mean():.4f}")


print("\nLoading Base Models...")

rf_loaded = joblib.load(
    'models/rf_model.pkl'
)

xgb_loaded = joblib.load(
    'models/xgb_model.pkl'
)

lgb_loaded = joblib.load(
    'models/lgbm_model.pkl'
)

print("Base Models Loaded Successfully")


base_models = [

    ('random_forest', rf_loaded),

    ('xgboost', xgb_loaded),

    ('lightgbm', lgb_loaded)
]


cv_strategy = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42
)


def objective_meta(trial):

    params = {

        'n_estimators': trial.suggest_int(
            'n_estimators',
            50,
            300,
            step=50
        ),

        'max_depth': trial.suggest_int(
            'max_depth',
            2,
            8
        ),

        'learning_rate': trial.suggest_float(
            'learning_rate',
            0.01,
            0.2,
            log=True
        ),

        'subsample': trial.suggest_float(
            'subsample',
            0.6,
            1.0
        ),

        'colsample_bytree': trial.suggest_float(
            'colsample_bytree',
            0.6,
            1.0
        ),

        'min_child_weight': trial.suggest_int(
            'min_child_weight',
            1,
            10
        ),

        'gamma': trial.suggest_float(
            'gamma',
            0,
            0.5
        ),

        'reg_alpha': trial.suggest_float(
            'reg_alpha',
            0,
            2
        ),

        'reg_lambda': trial.suggest_float(
            'reg_lambda',
            0,
            2
        ),

        'eval_metric': 'logloss',

        'random_state': 42,

        'n_jobs': -1
    }

    xgb_meta = xgb.XGBClassifier(**params)

    stacking = StackingClassifier(

        estimators=base_models,

        final_estimator=xgb_meta,

        cv=cv_strategy,

        stack_method='predict_proba',

        passthrough=False,

        n_jobs=-1
    )

    scores = cross_val_score(

        stacking,

        X_train,

        y_train,

        cv=cv_strategy,

        scoring='roc_auc',

        n_jobs=-1
    )

    return scores.mean()


print("\n" + "=" * 60)
print("OPTUNA TUNING : META LEARNER")
print("=" * 60)

study = optuna.create_study(

    direction='maximize',

    sampler=TPESampler(seed=42)
)

study.optimize(

    objective_meta,

    n_trials=20,

    show_progress_bar=True
)

print("\nOptuna Tuning Completed")

print(f"\nBest Parameters:")
print(study.best_params)

print(f"\nBest ROC-AUC : {study.best_value:.4f}")


best_params = study.best_params.copy()

best_params['eval_metric'] = 'logloss'

best_params['random_state'] = 42

best_params['n_jobs'] = -1


xgb_meta_best = xgb.XGBClassifier(**best_params)


print("\n" + "=" * 60)
print("TRAINING FINAL STACKING MODEL")
print("=" * 60)

stacking = StackingClassifier(

    estimators=base_models,

    final_estimator=xgb_meta_best,

    cv=cv_strategy,

    stack_method='predict_proba',

    passthrough=False,

    n_jobs=-1
)

print("\nTraining Final Stacking Model...")

stacking.fit(

    X_train,

    y_train
)

print("\nFinal Stacking Model Trained")


joblib.dump(

    stacking,

    'models/stacking_optuna_meta.pkl'
)

joblib.dump(

    study,

    'models/stacking_optuna_study.pkl'
)

print("\nModels Saved Successfully")


def evaluate_model(

        model,

        name,

        threshold=0.3
):

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    y_pred = (
        y_prob > threshold
    ).astype(int)

    acc = accuracy_score(
        y_test,
        y_pred
    )

    rec = recall_score(
        y_test,
        y_pred
    )

    pre = precision_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    tn, fp, fn, tp = cm.ravel()

    print("\n" + "=" * 60)

    print(f"{name}")

    print("=" * 60)

    print(f"\nThreshold : {threshold}")

    print(f"\nAccuracy  : {acc:.4f}")

    print(f"Recall    : {rec:.4f}")

    print(f"Precision : {pre:.4f}")

    print(f"F1 Score  : {f1:.4f}")

    print(f"ROC-AUC   : {auc:.4f}")

    print("\nConfusion Matrix")

    print(f"TN : {tn}")

    print(f"FP : {fp}")

    print(f"FN : {fn}")

    print(f"TP : {tp}")

    return y_prob, rec, pre, f1, auc, cm


models = {

    "Random Forest": rf_loaded,

    "XGBoost": xgb_loaded,

    "LightGBM": lgb_loaded,

    "Stacking (Optuna Meta)": stacking
}

results = []

predictions = {}

cms = {}

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

for name, model in models.items():

    y_prob, rec, pre, f1, auc, cm = evaluate_model(

        model,

        name,

        threshold=0.3
    )

    predictions[name] = y_prob

    cms[name] = cm

    results.append({

        "Model": name,

        "Recall": rec,

        "Precision": pre,

        "F1": f1,

        "AUC": auc
    })


df = pd.DataFrame(results)

df = df.sort_values(

    by='Recall',

    ascending=False
)

print("\n" + "=" * 60)
print("PERFORMANCE TABLE")
print("=" * 60)

print(df.to_string(index=False))

df.to_csv(

    'models/results_optuna_meta.csv',

    index=False
)

print("\nResults Saved")


print("\nCONFUSION MATRICES")

fig, axes = plt.subplots(

    2,

    2,

    figsize=(12, 10)
)

axes = axes.ravel()

for i, (name, cm) in enumerate(cms.items()):

    sns.heatmap(

        cm,

        annot=True,

        fmt='d',

        cmap='Blues',

        ax=axes[i]
    )

    axes[i].set_title(name)

    axes[i].set_xlabel('Predicted')

    axes[i].set_ylabel('Actual')

plt.tight_layout()

plt.savefig(

    'models/confusion_matrices_optuna_meta.png',

    dpi=150
)

plt.show()

print("Confusion Matrices Saved")


print("\nROC CURVES")

plt.figure(figsize=(10, 8))

for name, y_prob in predictions.items():

    fpr, tpr, _ = roc_curve(

        y_test,

        y_prob
    )

    auc = roc_auc_score(

        y_test,

        y_prob
    )

    plt.plot(

        fpr,

        tpr,

        lw=2,

        label=f'{name} (AUC={auc:.4f})'
    )

plt.plot(

    [0, 1],

    [0, 1],

    'k--'
)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curves')

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(

    'models/roc_curves_optuna_meta.png',

    dpi=150
)

plt.show()

print("ROC Curves Saved")


print("\n" + "=" * 60)
print("BEST THRESHOLD SEARCH")
print("=" * 60)

thresholds = np.arange(

    0.05,

    0.65,

    0.05
)

y_prob_stacking = predictions[
    "Stacking (Optuna Meta)"
]

best_score = 0

best_threshold = 0.3

for t in thresholds:

    y_pred = (
        y_prob_stacking > t
    ).astype(int)

    rec = recall_score(
        y_test,
        y_pred
    )

    pre = precision_score(
        y_test,
        y_pred
    )

    score = (
        0.7 * rec
        +
        0.3 * pre
    )

    print(

        f"Threshold={t:.2f} | "
        f"Recall={rec:.4f} | "
        f"Precision={pre:.4f}"
    )

    if score > best_score:

        best_score = score

        best_threshold = t

print(f"\nBest Threshold : {best_threshold}")


print("\n" + "=" * 60)
print("FINAL STACKING EVALUATION")
print("=" * 60)

evaluate_model(

    stacking,

    "Stacking (Optuna Meta)",

    threshold=best_threshold
)


print("\n" + "=" * 60)
print("META LEARNER ANALYSIS")
print("=" * 60)

meta_model = stacking.final_estimator_

if hasattr(meta_model, 'feature_importances_'):

    importances = meta_model.feature_importances_

    base_model_names = [

        'Random Forest',

        'XGBoost',

        'LightGBM'
    ]

    if len(importances) >= 3:

        print("\nBase Model Importance")

        for name, imp in zip(

                base_model_names,

                importances[:3]
        ):

            print(f"{name} : {imp:.4f}")

        plt.figure(figsize=(8, 5))

        plt.bar(

            base_model_names,

            importances[:3]
        )

        plt.title(
            'Meta Learner Feature Importance'
        )

        plt.ylabel(
            'Importance'
        )

        plt.tight_layout()

        plt.savefig(

            'models/meta_importance.png',

            dpi=150
        )

        plt.show()

        print("\nImportance Plot Saved")


best_model = df.iloc[0]

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"\nBest Model : {best_model['Model']}")

print(f"Recall     : {best_model['Recall']:.4f}")

print(f"Precision  : {best_model['Precision']:.4f}")

print(f"F1 Score   : {best_model['F1']:.4f}")

print(f"ROC-AUC    : {best_model['AUC']:.4f}")

print("\nSTACKING PIPELINE COMPLETED")


print("\nSaved Files")

print("\n1. stacking_optuna_meta.pkl")

print("2. stacking_optuna_study.pkl")

print("3. results_optuna_meta.csv")

print("4. confusion_matrices_optuna_meta.png")

print("5. roc_curves_optuna_meta.png")

print("6. meta_importance.png")