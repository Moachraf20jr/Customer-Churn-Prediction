"""
Customer Churn Prediction Pipeline
====================================
A complete end-to-end machine learning pipeline for predicting customer churn.
Includes data loading, EDA, preprocessing, model training, evaluation,
hyperparameter tuning, and prediction functions.

Author: Senior Machine Learning Engineer
Date: 2026
"""

import os
import sys
import warnings
import logging
from typing import Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

import joblib

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'results', 'plots')
METRICS_DIR = os.path.join(BASE_DIR, 'results', 'metrics')

for d in [DATA_DIR, MODELS_DIR, PLOTS_DIR, METRICS_DIR]:
    os.makedirs(d, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_data() -> pd.DataFrame:
    """
    Load customer churn dataset from CSV files.
    Combines training and testing datasets for a larger pool.
    """
    train_path = os.path.join(DATA_DIR, 'customer_churn_dataset-training-master.csv')
    test_path = os.path.join(DATA_DIR, 'customer_churn_dataset-testing-master.csv')

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Dataset files not found in {DATA_DIR}. "
            "Please ensure both training and testing CSV files are present."
        )

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    logger.info(f"Training samples: {len(df_train):,}")
    logger.info(f"Testing samples: {len(df_test):,}")
    logger.info(f"Total samples: {len(df):,}")
    logger.info(f"Features: {df.shape[1]}")

    return df


# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

def perform_eda(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive exploratory data analysis.
    Returns a dictionary of EDA results for reporting.
    """
    logger.info("=" * 60)
    logger.info("EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 60)

    results = {}

    # 2a. Dataset Overview
    logger.info("\n[2a] Dataset Overview:")
    logger.info(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    logger.info(f"  Columns: {list(df.columns)}")

    # 2b. Data Types & Info
    logger.info("\n[2b] Data Types:")
    dtype_info = pd.DataFrame({
        'Column': df.dtypes.index,
        'Data Type': df.dtypes.values,
        'Non-Null Count': df.notna().sum().values,
        'Null Count': df.isna().sum().values,
        'Null %': (df.isna().sum() / len(df) * 100).round(2).values
    })
    logger.info(f"\n{dtype_info.to_string(index=False)}")

    # 2c. Missing Values
    logger.info("\n[2c] Missing Values:")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Missing %': missing_pct.values
        })
        logger.info(f"\n{missing_df.to_string(index=False)}")
    else:
        logger.info("  No missing values found.")

    # 2d. Duplicate Records
    logger.info("\n[2d] Duplicate Records:")
    dup_count = df.duplicated().sum()
    logger.info(f"  Duplicate rows: {dup_count:,} ({dup_count/len(df)*100:.2f}%)")
    results['duplicates'] = int(dup_count)

    # 2e. Statistical Summary
    logger.info("\n[2e] Statistical Summary (Numerical Features):")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'CustomerID' in num_cols:
        num_cols.remove('CustomerID')
    if 'Churn' in num_cols:
        num_cols.remove('Churn')
    logger.info(f"\n{df[num_cols].describe().to_string()}")

    # 2f. Class Distribution
    logger.info("\n[2f] Class Distribution (Churn):")
    churn_counts = df['Churn'].value_counts()
    churn_pcts = df['Churn'].value_counts(normalize=True) * 100
    for actual_label in churn_counts.index:
        count = churn_counts[actual_label]
        pct = churn_pcts[actual_label]
        label_str = 'Churned' if actual_label == 1 else 'Not Churned'
        logger.info(f"  {label_str}: {int(count):,} ({pct:.2f}%)")

    churned_count = int(churn_counts.get(1.0, churn_counts.get(1, 0)))
    not_churned_count = int(churn_counts.get(0.0, churn_counts.get(0, 0)))
    churn_rate = (churned_count / len(df.dropna(subset=['Churn'])) * 100)

    results['class_distribution'] = {
        'not_churned': not_churned_count,
        'churned': churned_count,
        'churn_rate': churn_rate
    }

    results['numerical_columns'] = num_cols
    results['categorical_columns'] = df.select_dtypes(include=['object', 'str']).columns.tolist()
    results['data_shape'] = df.shape

    # --- Save EDA Visualizations ---

    # 1. Churn Distribution (Pie Chart)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Customer Churn - Exploratory Data Analysis', fontsize=16, fontweight='bold')

    colors = ['#e74c3c' if idx == 1 else '#2ecc71' for idx in churn_counts.index]
    pie_labels = ['Churned' if idx == 1 else 'Not Churned' for idx in churn_counts.index]
    wedges, texts, autotexts = axes[0, 0].pie(
        churn_counts.values, labels=pie_labels,
        autopct='%1.1f%%', colors=colors, startangle=90, explode=(0, 0.05)
    )
    axes[0, 0].set_title('Churn Distribution', fontsize=13, fontweight='bold')

    # 2. Age Distribution by Churn
    for churn_val, color, label in [(0, '#2ecc71', 'Not Churned'), (1, '#e74c3c', 'Churned')]:
        subset = df[df['Churn'] == churn_val]['Age']
        axes[0, 1].hist(subset, bins=30, alpha=0.6, color=color, label=label)
    axes[0, 1].set_title('Age Distribution by Churn', fontweight='bold')
    axes[0, 1].set_xlabel('Age')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()

    # 3. Tenure Distribution by Churn
    for churn_val, color, label in [(0, '#2ecc71', 'Not Churned'), (1, '#e74c3c', 'Churned')]:
        subset = df[df['Churn'] == churn_val]['Tenure']
        axes[0, 2].hist(subset, bins=30, alpha=0.6, color=color, label=label)
    axes[0, 2].set_title('Tenure Distribution by Churn', fontweight='bold')
    axes[0, 2].set_xlabel('Tenure (months)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].legend()

    # 4. Support Calls vs Churn
    support_counts = df.groupby('Support Calls')['Churn'].mean() * 100
    axes[1, 0].bar(support_counts.index, support_counts.values, color='#3498db')
    axes[1, 0].set_title('Churn Rate by Support Calls', fontweight='bold')
    axes[1, 0].set_xlabel('Number of Support Calls')
    axes[1, 0].set_ylabel('Churn Rate (%)')

    # 5. Payment Delay vs Churn
    for churn_val, color, label in [(0, '#2ecc71', 'Not Churned'), (1, '#e74c3c', 'Churned')]:
        subset = df[df['Churn'] == churn_val]['Payment Delay']
        axes[1, 1].hist(subset, bins=30, alpha=0.6, color=color, label=label)
    axes[1, 1].set_title('Payment Delay Distribution by Churn', fontweight='bold')
    axes[1, 1].set_xlabel('Payment Delay (days)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()

    # 6. Contract Length vs Churn
    contract_churn = df.groupby('Contract Length')['Churn'].mean() * 100
    contract_order = ['Monthly', 'Quarterly', 'Annual']
    contract_churn = contract_churn.reindex(
        [c for c in contract_order if c in contract_churn.index]
    )
    axes[1, 2].bar(contract_churn.index, contract_churn.values, color=['#e74c3c', '#f39c12', '#2ecc71'])
    axes[1, 2].set_title('Churn Rate by Contract Length', fontweight='bold')
    axes[1, 2].set_xlabel('Contract Length')
    axes[1, 2].set_ylabel('Churn Rate (%)')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(PLOTS_DIR, 'eda_overview.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  [SAVED] eda_overview.png")

    # Correlation Heatmap
    logger.info("\n[2g] Correlation Analysis:")
    corr_cols = num_cols + ['Churn'] if 'Churn' not in num_cols else num_cols
    corr_matrix = df[corr_cols].corr()

    plt.figure(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt='.2f',
        cmap='RdBu_r', center=0, square=True,
        linewidths=0.5, cbar_kws={'shrink': 0.8}
    )
    plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  [SAVED] correlation_heatmap.png")

    # Sort correlations with Churn
    churn_corr = corr_matrix['Churn'].drop('Churn').sort_values(key=abs, ascending=False)
    logger.info("  Features most correlated with Churn:")
    for feat, corr_val in churn_corr.items():
        logger.info(f"    {feat}: {corr_val:+.4f}")
    results['churn_correlations'] = churn_corr.to_dict()

    # Bar plot of correlations with Churn
    plt.figure(figsize=(10, 6))
    colors_bar = ['#e74c3c' if v < 0 else '#2ecc71' for v in churn_corr.values]
    plt.barh(churn_corr.index, churn_corr.values, color=colors_bar)
    plt.xlabel('Correlation with Churn')
    plt.title('Feature Correlation with Churn', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'churn_correlation_bar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  [SAVED] churn_correlation_bar.png")

    # Categorical features vs Churn
    cat_cols = df.select_dtypes(include=['object', 'str']).columns.tolist()
    for col in cat_cols:
        if col not in ['CustomerID']:
            plt.figure(figsize=(10, 6))
            cross = pd.crosstab(df[col], df['Churn'], normalize='index') * 100
            cross.plot(kind='bar', stacked=True, color=['#2ecc71', '#e74c3c'], ax=plt.gca())
            plt.title(f'Churn Distribution by {col}', fontsize=13, fontweight='bold')
            plt.xlabel(col)
            plt.ylabel('Percentage (%)')
            plt.legend(['Not Churned', 'Churned'], loc='upper right')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f'{col}_vs_churn.png'), dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"  [SAVED] {col}_vs_churn.png")

    return results


# =============================================================================
# 3. DATA PREPROCESSING
# =============================================================================

def preprocess_data(df: pd.DataFrame, is_training: bool = True,
                    encoders: Optional[Dict] = None,
                    scaler: Optional[StandardScaler] = None
                    ) -> Tuple[pd.DataFrame, pd.Series, Dict, StandardScaler]:
    """
    Clean and preprocess the data:
    - Handle missing values
    - Remove irrelevant columns
    - Encode categorical variables
    - Scale numerical features
    - Feature engineering
    """
    logger.info("=" * 60)
    logger.info("DATA PREPROCESSING")
    logger.info("=" * 60)

    data = df.copy()

    # 3a. Remove CustomerID (irrelevant for modeling)
    if 'CustomerID' in data.columns:
        data = data.drop(columns=['CustomerID'])
        logger.info("[3a] Removed 'CustomerID' column (irrelevant for modeling).")

    # 3b. Handle Missing Values
    logger.info("\n[3b] Handling Missing Values:")
    missing_before = data.isna().sum().sum()
    logger.info(f"  Total missing values before: {missing_before}")

    # Numerical columns: fill with median
    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if 'Churn' in num_cols:
        num_cols.remove('Churn')
    for col in num_cols:
        if data[col].isna().sum() > 0:
            median_val = data[col].median()
            data[col] = data[col].fillna(median_val)
            logger.info(f"  Filled '{col}' missing values with median ({median_val:.2f})")

    # Categorical columns: fill with mode
    cat_cols = data.select_dtypes(include=['object', 'str']).columns.tolist()
    for col in cat_cols:
        if data[col].isna().sum() > 0:
            mode_val = data[col].mode()[0]
            data[col] = data[col].fillna(mode_val)
            logger.info(f"  Filled '{col}' missing values with mode ({mode_val})")

    # Drop rows with NaN in target variable
    if 'Churn' in data.columns and data['Churn'].isna().any():
        nan_churn_count = data['Churn'].isna().sum()
        data = data.dropna(subset=['Churn'])
        logger.info(f"  Dropped {nan_churn_count} row(s) with NaN target.")

    missing_after = data.isna().sum().sum()
    logger.info(f"  Total missing values after: {missing_after}")

    # 3c. Remove Duplicates
    dup_before = data.duplicated().sum()
    if dup_before > 0:
        data = data.drop_duplicates(keep='first')
        logger.info(f"\n[3c] Removed {dup_before:,} duplicate rows.")

    # 3d. Feature Engineering
    logger.info("\n[3d] Feature Engineering:")
    # Usage-to-Support ratio: how often they use vs call support
    if 'Usage Frequency' in data.columns and 'Support Calls' in data.columns:
        data['Usage_Support_Ratio'] = data['Usage Frequency'] / (
            data['Support Calls'] + 1
        )
        logger.info("  Created 'Usage_Support_Ratio' (Usage Freq / Support Calls)")

    # Average spend per month of tenure
    if 'Total Spend' in data.columns and 'Tenure' in data.columns:
        data['Spend_Per_Month'] = data['Total Spend'] / (data['Tenure'] + 1)
        logger.info("  Created 'Spend_Per_Month' (Total Spend / Tenure)")

    # Interaction gap (if Last Interaction is available)
    if 'Last Interaction' in data.columns:
        data['Days_Since_Interaction'] = data['Last Interaction']
        logger.info("  Kept 'Days_Since_Interaction' for modeling.")

    # 3e. Encode Categorical Variables
    logger.info("\n[3e] Encoding Categorical Variables:")
    if encoders is None:
        encoders = {}
        is_training = True
    else:
        is_training = False

    cat_cols = data.select_dtypes(include=['object', 'str']).columns.tolist()
    for col in cat_cols:
        if col != 'Churn':
            if is_training:
                encoders[col] = LabelEncoder()
                data[col] = encoders[col].fit_transform(data[col])
            else:
                data[col] = encoders[col].transform(data[col])
            logger.info(f"  Label encoded '{col}' -> {len(encoders[col].classes_)} classes")

    # 3f. Scale Numerical Features
    logger.info("\n[3f] Scaling Numerical Features:")
    scale_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if 'Churn' in scale_cols:
        scale_cols.remove('Churn')

    if scaler is None:
        scaler = StandardScaler()
        data[scale_cols] = scaler.fit_transform(data[scale_cols])
        logger.info(f"  Fitted StandardScaler on {len(scale_cols)} features")
    else:
        data[scale_cols] = scaler.transform(data[scale_cols])
        logger.info(f"  Transformed {len(scale_cols)} features using existing scaler")

    # Separate features and target
    X = data.drop(columns=['Churn'])
    y = data['Churn']

    logger.info(f"\n  Final feature set: {X.shape[1]} features")
    logger.info(f"  Final samples: {len(X):,}")
    logger.info(f"  Features: {list(X.columns)}")

    return X, y, encoders, scaler


# =============================================================================
# 4. MODEL TRAINING & EVALUATION
# =============================================================================

def train_and_evaluate(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series, y_test: pd.Series
) -> Tuple[Dict[str, Dict], Any]:
    """
    Train multiple classification models and compare their performance.
    Models: Logistic Regression, Random Forest, XGBoost, LightGBM
    """
    logger.info("=" * 60)
    logger.info("MODEL TRAINING & EVALUATION")
    logger.info("=" * 60)

    # Use subset of data for training to manage memory
    X_train_sub = X_train
    y_train_sub = y_train
    if len(X_train) > 100000:
        from sklearn.utils import resample
        X_train_sub, y_train_sub = resample(
            X_train, y_train, n_samples=100000,
            random_state=RANDOM_STATE, stratify=y_train
        )
        logger.info(f"  Subsampled training data to 100,000 for faster training.")

    models = {
        'Logistic Regression': LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000, n_jobs=1
        ),
        'Random Forest': RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=1,
            n_estimators=50, max_depth=15,
            min_samples_leaf=10, max_samples=0.5
        ),
    }

    # XGBoost
    try:
        import xgboost as xgb
        models['XGBoost'] = xgb.XGBClassifier(
            random_state=RANDOM_STATE, use_label_encoder=False,
            eval_metric='logloss', verbosity=0
        )
        logger.info("  XGBoost loaded successfully.")
    except Exception as e:
        logger.warning(f"  XGBoost not available ({e}). Skipping.")

    # LightGBM
    try:
        import lightgbm as lgb
        models['LightGBM'] = lgb.LGBMClassifier(
            random_state=RANDOM_STATE, verbose=-1
        )
        logger.info("  LightGBM loaded successfully.")
    except Exception as e:
        logger.warning(f"  LightGBM not available ({e}). Skipping.")

    results = {}
    best_model = None
    best_score = -1
    best_name = None

    # Plot setup for ROC curves
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.5)

    for name, model in models.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Training: {name}")
        logger.info(f"{'='*50}")

        # Train (use subsample for tree-based models to manage memory)
        if name in ['Random Forest', 'XGBoost', 'LightGBM']:
            model.fit(X_train_sub, y_train_sub)
        else:
            model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)

        results[name] = {
            'model': model,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }

        logger.info(f"  Accuracy:  {acc:.4f}")
        logger.info(f"  Precision: {prec:.4f}")
        logger.info(f"  Recall:    {rec:.4f}")
        logger.info(f"  F1-Score:  {f1:.4f}")
        logger.info(f"  ROC-AUC:   {roc_auc:.4f}")
        logger.info(f"\n  Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")

        # Confusion Matrix plot
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Churned', 'Churned'],
            yticklabels=['Not Churned', 'Churned'],
            ax=ax_cm
        )
        ax_cm.set_title(f'Confusion Matrix - {name}', fontsize=13, fontweight='bold')
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'confusion_matrix_{name.replace(" ", "_")}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)

        # Track best model by F1-score
        if f1 > best_score:
            best_score = f1
            best_model = model
            best_name = name

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'roc_curves_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"\n  [SAVED] roc_curves_comparison.png")

    # --- Model Comparison Bar Chart ---
    metrics_df = pd.DataFrame({
        name: {
            m: results[name][m]
            for m in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        }
        for name in results
    }).T

    metrics_df.plot(kind='bar', figsize=(14, 7), rot=0, colormap='viridis')
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend(loc='lower right', title='Metrics')
    plt.grid(axis='y', alpha=0.3)
    for i, name in enumerate(metrics_df.index):
        for j, metric in enumerate(metrics_df.columns):
            val = metrics_df.loc[name, metric]
            plt.text(i + j * 0.15 - 0.2, val + 0.01, f'{val:.3f}', fontsize=8, ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  [SAVED] model_comparison.png")

    # Save metrics to CSV
    metrics_df.to_csv(os.path.join(METRICS_DIR, 'model_metrics.csv'))
    logger.info(f"\n  [SAVED] metrics to model_metrics.csv")

    logger.info(f"\n{'='*50}")
    logger.info(f"BEST MODEL: {best_name} (F1 = {best_score:.4f})")
    logger.info(f"{'='*50}")

    return results, best_name


# =============================================================================
# 5. HYPERPARAMETER TUNING
# =============================================================================

def hyperparameter_tuning(
    best_model_name: str,
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[Any, Dict]:
    """
    Perform hyperparameter tuning on the best performing model using GridSearchCV.
    """
    logger.info("=" * 60)
    logger.info(f"HYPERPARAMETER TUNING: {best_model_name}")
    logger.info("=" * 60)

    # Subsample for tuning to manage memory
    tune_sample_size = min(50000, len(X_train))
    X_tune, y_tune = X_train, y_train
    if len(X_train) > tune_sample_size:
        from sklearn.utils import resample
        X_tune, y_tune = resample(
            X_train, y_train, n_samples=tune_sample_size,
            random_state=RANDOM_STATE, stratify=y_train
        )
        logger.info(f"  Using {tune_sample_size} samples for tuning.")

    param_grids = {
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['liblinear'],
            'class_weight': [None, 'balanced']
        },
        'Random Forest': {
            'n_estimators': [50, 100],
            'max_depth': [10, 20],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0]
        },
        'LightGBM': {
            'n_estimators': [100, 200],
            'max_depth': [-1, 10],
            'learning_rate': [0.01, 0.1],
            'num_leaves': [31, 50]
        }
    }

    if best_model_name not in param_grids:
        logger.warning(f"No param grid found for {best_model_name}. Skipping tuning.")
        return None, {}

    # Create base model
    model_map = {
        'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE),
        'XGBoost': None,
        'LightGBM': None
    }

    if best_model_name == 'XGBoost':
        try:
            import xgboost as xgb
            base_model = xgb.XGBClassifier(
                random_state=RANDOM_STATE, use_label_encoder=False,
                eval_metric='logloss', verbosity=0
            )
        except Exception as e:
            logger.warning(f"  XGBoost unavailable ({e}). Using base model.")
            base_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    elif best_model_name == 'LightGBM':
        try:
            import lightgbm as lgb
            base_model = lgb.LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)
        except Exception as e:
            logger.warning(f"  LightGBM unavailable ({e}). Using base model.")
            base_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    else:
        base_model = model_map[best_model_name]

    param_grid = param_grids[best_model_name]

    logger.info(f"  Parameter grid: {param_grid}")
    logger.info(f"  Total combinations to search...")

    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=1,
        verbose=1,
        return_train_score=True
    )

    grid_search.fit(X_tune, y_tune)

    logger.info(f"\n  Best Parameters: {grid_search.best_params_}")
    logger.info(f"  Best CV F1-Score: {grid_search.best_score_:.4f}")

    # Evaluate on test set
    best_estimator = grid_search.best_estimator_
    y_pred = best_estimator.predict(X_test)
    y_proba = best_estimator.predict_proba(X_test)[:, 1]

    test_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }

    logger.info(f"\n  Test Set Performance after Tuning:")
    for metric, value in test_metrics.items():
        logger.info(f"    {metric}: {value:.4f}")

    # Confusion Matrix for tuned model
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Greens',
        xticklabels=['Not Churned', 'Churned'],
        yticklabels=['Not Churned', 'Churned'],
        ax=ax_cm
    )
    ax_cm.set_title(f'Confusion Matrix - {best_model_name} (Tuned)', fontsize=13, fontweight='bold')
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'confusion_matrix_{best_model_name.replace(" ", "_")}_tuned.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  [SAVED] confusion_matrix_{best_model_name}_tuned.png")

    return best_estimator, grid_search.best_params_


# =============================================================================
# 6. FEATURE IMPORTANCE ANALYSIS
# =============================================================================

def analyze_feature_importance(
    model: Any, feature_names: list, model_name: str
) -> pd.DataFrame:
    """
    Extract and visualize feature importance from the trained model.
    Handles tree-based models, linear models, and XGBoost/LightGBM.
    """
    logger.info("=" * 60)
    logger.info("FEATURE IMPORTANCE ANALYSIS")
    logger.info("=" * 60)

    importance_df = None

    # Tree-based models (Random Forest, XGBoost, LightGBM)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        logger.info("  Using model's feature_importances_ attribute.")

    # Linear models (Logistic Regression)
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        logger.info("  Using model's coefficients (absolute values).")

    if importance_df is not None and len(importance_df) > 0:
        # Plot
        top_n = min(15, len(importance_df))
        plt.figure(figsize=(10, 7))
        top_features = importance_df.head(top_n)
        colors_imp = plt.cm.viridis(np.linspace(0.8, 0.2, top_n))
        plt.barh(range(top_n), top_features['Importance'].values, color=colors_imp)
        plt.yticks(range(top_n), top_features['Feature'].values)
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Feature Importances - {model_name}', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'feature_importance_{model_name.replace(" ", "_")}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  [SAVED] feature_importance_{model_name.replace(' ', '_')}.png")

        logger.info(f"\n  Top 5 Most Important Features:")
        for i, row in importance_df.head(5).iterrows():
            logger.info(f"    {i+1}. {row['Feature']}: {row['Importance']:.4f}")

        importance_df.to_csv(
            os.path.join(METRICS_DIR, f'feature_importance_{model_name.replace(" ", "_")}.csv'),
            index=False
        )
        logger.info(f"  [SAVED] feature importance CSV")

    return importance_df


# =============================================================================
# 7. SAVE FINAL MODEL
# =============================================================================

def save_model(model: Any, scaler: StandardScaler, encoders: Dict,
               feature_names: list, model_name: str, metrics: Dict):
    """
    Save the trained model, scaler, encoders, and metadata using joblib.
    """
    logger.info("=" * 60)
    logger.info("SAVING MODEL & ARTIFACTS")
    logger.info("=" * 60)

    # Save model (with compression to save disk space)
    model_path = os.path.join(MODELS_DIR, f'{model_name.replace(" ", "_").lower()}_model.pkl')
    joblib.dump(model, model_path, compress=('zlib', 3))
    logger.info(f"  Model saved to: {model_path}")

    # Save scaler
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    logger.info(f"  Scaler saved to: {scaler_path}")

    # Save encoders
    encoders_path = os.path.join(MODELS_DIR, 'encoders.pkl')
    joblib.dump(encoders, encoders_path)
    logger.info(f"  Encoders saved to: {encoders_path}")

    # Save metadata
    metadata = {
        'model_name': model_name,
        'feature_names': feature_names,
        'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in metrics.items()},
        'random_state': RANDOM_STATE
    }
    metadata_path = os.path.join(MODELS_DIR, 'metadata.pkl')
    joblib.dump(metadata, metadata_path)
    logger.info(f"  Metadata saved to: {metadata_path}")

    logger.info(f"\n  All artifacts saved in: {MODELS_DIR}/")


# =============================================================================
# 8. PREDICTION FUNCTION
# =============================================================================

def load_model_artifacts(models_dir: str = None) -> Dict:
    """
    Load all saved model artifacts for inference.
    Returns a dictionary with model, scaler, encoders, and metadata.
    """
    if models_dir is None:
        models_dir = MODELS_DIR

    artifacts = {}

    # Find the model file
    model_files = [f for f in os.listdir(models_dir) if f.endswith('_model.pkl')]
    if not model_files:
        raise FileNotFoundError(f"No model file (*_model.pkl) found in {models_dir}")

    model_path = os.path.join(models_dir, model_files[0])
    artifacts['model'] = joblib.load(model_path)

    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    if os.path.exists(scaler_path):
        artifacts['scaler'] = joblib.load(scaler_path)

    encoders_path = os.path.join(models_dir, 'encoders.pkl')
    if os.path.exists(encoders_path):
        artifacts['encoders'] = joblib.load(encoders_path)

    metadata_path = os.path.join(models_dir, 'metadata.pkl')
    if os.path.exists(metadata_path):
        artifacts['metadata'] = joblib.load(metadata_path)

    logger.info(f"Loaded model artifacts from {models_dir}")
    logger.info(f"  Model: {os.path.basename(model_path)}")

    return artifacts


def predict_churn(customer_data: pd.DataFrame, artifacts: Dict) -> pd.DataFrame:
    """
    Predict churn for new customer data.

    Parameters
    ----------
    customer_data : pd.DataFrame
        DataFrame containing customer features (must match training columns).
    artifacts : dict
        Dictionary containing the trained model, scaler, encoders.

    Returns
    -------
    pd.DataFrame
        Original data with 'Churn_Prediction' and 'Churn_Probability' columns.
    """
    data = customer_data.copy()

    # Retrieve artifacts
    model = artifacts.get('model')
    scaler = artifacts.get('scaler')
    encoders = artifacts.get('encoders', {})
    metadata = artifacts.get('metadata', {})

    expected_features = metadata.get('feature_names', [])

    # Encode categorical features
    cat_cols = data.select_dtypes(include=['object', 'str']).columns.tolist()
    for col in cat_cols:
        if col in encoders:
            try:
                data[col] = encoders[col].transform(data[col])
            except ValueError as e:
                logger.warning(f"  Encoding warning for '{col}': {e}")
                unseen_mask = ~data[col].isin(encoders[col].classes_)
                if unseen_mask.any():
                    data.loc[unseen_mask, col] = encoders[col].classes_[0]
                data[col] = encoders[col].transform(data[col])

    # Drop CustomerID if present
    if 'CustomerID' in data.columns:
        data = data.drop(columns=['CustomerID'])

    # Add engineered features if needed
    if 'Usage Frequency' in data.columns and 'Support Calls' in data.columns:
        data['Usage_Support_Ratio'] = data['Usage Frequency'] / (data['Support Calls'] + 1)

    if 'Total Spend' in data.columns and 'Tenure' in data.columns:
        data['Spend_Per_Month'] = data['Total Spend'] / (data['Tenure'] + 1)

    # Ensure all expected features exist
    for col in expected_features:
        if col not in data.columns:
            data[col] = 0

    # Reorder columns to match training
    data = data[expected_features]

    # Scale (preserve DataFrame for feature names)
    if scaler:
        data_scaled = pd.DataFrame(
            scaler.transform(data),
            columns=data.columns,
            index=data.index
        )
    else:
        data_scaled = data

    # Predict
    predictions = model.predict(data_scaled)
    probabilities = model.predict_proba(data_scaled)[:, 1]

    # Prepare results
    results = customer_data.copy()
    results['Churn_Prediction'] = predictions
    results['Churn_Probability'] = np.round(probabilities, 4)
    results['Risk_Level'] = [
        'High' if p >= 0.7 else ('Medium' if p >= 0.3 else 'Low')
        for p in probabilities
    ]

    return results


# =============================================================================
# 9. MAIN PIPELINE
# =============================================================================

def main():
    """
    Execute the complete customer churn prediction pipeline:
    1. Load data
    2. EDA
    3. Preprocessing
    4. Train/Test split
    5. Model training & comparison
    6. Hyperparameter tuning
    7. Feature importance
    8. Save model
    9. Prediction example
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("CUSTOMER CHURN PREDICTION PIPELINE")
    logger.info("=" * 60)

    # Step 1: Load Data
    logger.info("\n>>> STEP 1: LOADING DATA")
    df = load_data()
    logger.info(f"  Data loaded: {df.shape}")

    # Step 2: EDA
    logger.info("\n>>> STEP 2: EXPLORATORY DATA ANALYSIS")
    eda_results = perform_eda(df)

    # Step 3: Preprocessing
    logger.info("\n>>> STEP 3: DATA PREPROCESSING")
    X, y, encoders, scaler = preprocess_data(df, is_training=True)
    logger.info(f"  Preprocessed data: X={X.shape}, y={y.shape}")

    # Step 4: Train/Test Split
    logger.info("\n>>> STEP 4: TRAIN-TEST SPLIT")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"  Training set: {X_train.shape[0]:,} samples")
    logger.info(f"  Testing set:  {X_test.shape[0]:,} samples")

    # Step 5: Train & Compare Models
    logger.info("\n>>> STEP 5: MODEL TRAINING & COMPARISON")
    results, best_name = train_and_evaluate(X_train, X_test, y_train, y_test)

    best_model_obj = results[best_name]['model']

    # Step 6: Hyperparameter Tuning
    logger.info("\n>>> STEP 6: HYPERPARAMETER TUNING")
    tuned_model, best_params = hyperparameter_tuning(
        best_name, X_train, y_train, X_test, y_test
    )
    final_model = tuned_model if tuned_model is not None else best_model_obj

    # Step 7: Feature Importance
    logger.info("\n>>> STEP 7: FEATURE IMPORTANCE ANALYSIS")
    feature_names = X.columns.tolist()
    importance_df = analyze_feature_importance(final_model, feature_names, best_name)

    # Step 8: Save Model
    logger.info("\n>>> STEP 8: SAVING MODEL")
    final_metrics = {
        'accuracy': accuracy_score(y_test, final_model.predict(X_test)),
        'precision': precision_score(y_test, final_model.predict(X_test), zero_division=0),
        'recall': recall_score(y_test, final_model.predict(X_test), zero_division=0),
        'f1_score': f1_score(y_test, final_model.predict(X_test), zero_division=0),
        'roc_auc': roc_auc_score(y_test, final_model.predict_proba(X_test)[:, 1])
    }
    save_model(final_model, scaler, encoders, feature_names, best_name, final_metrics)

    # Step 9: Prediction Demo
    logger.info("\n>>> STEP 9: PREDICTION DEMO")
    logger.info("  Loading model artifacts for inference demo...")
    artifacts = load_model_artifacts()

    # Create sample data
    sample_data = X_test.head(5).copy()
    sample_data_with_ids = pd.DataFrame(
        scaler.inverse_transform(sample_data),
        columns=sample_data.columns
    )
    predictions = predict_churn(sample_data_with_ids, artifacts)
    logger.info("\n  Sample Predictions:")
    for i, row in predictions.iterrows():
        logger.info(
            f"    Customer {i}: Prediction={'Churn' if row['Churn_Prediction'] == 1 else 'No Churn'} "
            f"(Probability: {row['Churn_Probability']:.2%}, Risk: {row['Risk_Level']})"
        )

    # Final Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE - SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Dataset: {eda_results['data_shape'][0]:,} samples, {eda_results['data_shape'][1]} features")
    logger.info(f"  Churn Rate: {eda_results['class_distribution']['churn_rate']:.2f}%")
    logger.info(f"  Best Model: {best_name}")
    logger.info(f"  Test F1-Score: {final_metrics['f1_score']:.4f}")
    logger.info(f"  Test ROC-AUC: {final_metrics['roc_auc']:.4f}")
    if best_params:
        logger.info(f"  Best Hyperparameters: {best_params}")
    if importance_df is not None and len(importance_df) > 0:
        logger.info(f"  Top Feature: {importance_df.iloc[0]['Feature']} ({importance_df.iloc[0]['Importance']:.4f})")
    logger.info(f"  All plots saved in: {PLOTS_DIR}/")
    logger.info(f"  All metrics saved in: {METRICS_DIR}/")
    logger.info(f"  Model artifacts saved in: {MODELS_DIR}/")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
