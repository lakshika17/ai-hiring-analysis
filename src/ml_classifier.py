"""
Phase 5: Machine Learning Classifier for Hire/No-Hire Predictions
Implements Random Forest classifier with fairness analysis.

This module demonstrates how AI/ML can be applied to hiring decisions
and analyzes potential bias in the model's predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


def generate_hire_labels(df: pd.DataFrame, 
                         method: str = 'Balanced',
                         hire_rate: float = 0.3) -> pd.Series:
    """
    Generate hire/no-hire labels based on MAUT scores.
    
    Args:
        df: DataFrame with MAUT scores
        method: 'Efficiency', 'Equity', or 'Balanced' - which frame to use
        hire_rate: Proportion of candidates to label as "hire"
    
    Returns:
        Series with binary labels (1=hire, 0=no-hire)
    """
    # Capitalize first letter to match column names
    method_cap = method.capitalize()
    score_col = f'Score_{method_cap}'
    
    if score_col not in df.columns:
        raise ValueError(f"Score column {score_col} not found. Run apply_all_frames first.")
    
    # Top X% get hired based on their MAUT score
    threshold = df[score_col].quantile(1 - hire_rate)
    labels = (df[score_col] >= threshold).astype(int)
    
    return labels


def prepare_ml_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare feature matrix for ML model.
    
    Returns:
        Tuple of (feature_dataframe, feature_names)
    """
    # Use normalized utility scores as features
    feature_cols = ['U_x_Skills', 'U_x_Cultural', 'U_x_Ethics', 'U_x_Integrity']
    
    # Optionally include demographic features (to analyze bias)
    # Note: In real hiring, using these directly would be illegal discrimination
    # We include them here to DETECT if the model learns biased patterns
    
    features = df[feature_cols].copy()
    
    return features, feature_cols


def train_random_forest(X: pd.DataFrame, y: pd.Series,
                        test_size: float = 0.3,
                        random_state: int = 42) -> Dict:
    """
    Train a Random Forest classifier for hire/no-hire prediction.
    
    Args:
        X: Feature matrix
        y: Binary labels (1=hire, 0=no-hire)
        test_size: Proportion for test split
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with model, predictions, and metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Train Random Forest with improved parameters for larger dataset
    model = RandomForestClassifier(
        n_estimators=200,          # Increased from 100 for better ensemble
        max_depth=8,               # Increased from 5 for more complex patterns
        min_samples_split=10,      # Increased from 5 to prevent overfitting
        min_samples_leaf=4,        # Increased from 2 for smoother predictions
        random_state=random_state,
        class_weight='balanced',   # Handle imbalanced classes
        n_jobs=-1                  # Parallel processing for speed
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Cross-validation score with more folds for reliable estimates
    cv_scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'y_test_proba': y_test_proba,
        'cv_scores': cv_scores,
        'feature_importance': feature_importance,
    }


def compute_model_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Compute classification performance metrics.
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
    }


def compute_fairness_metrics(df: pd.DataFrame, 
                              predictions: np.ndarray,
                              sensitive_col: str = 'x_Minority') -> Dict:
    """
    Compute fairness metrics for ML predictions.
    
    Analyzes whether the model exhibits bias against protected groups.
    
    Metrics:
    - Demographic Parity: P(Hire | Minority) vs P(Hire | Non-Minority)
    - Disparate Impact Ratio: Should be >= 0.8 (4/5ths rule)
    - Equal Opportunity: True positive rates across groups
    """
    df_analysis = df.copy()
    df_analysis['ML_Prediction'] = predictions
    
    # Group statistics
    minority = df_analysis[df_analysis[sensitive_col] == 1]
    non_minority = df_analysis[df_analysis[sensitive_col] == 0]
    
    # Selection rates
    rate_minority = minority['ML_Prediction'].mean()
    rate_non_minority = non_minority['ML_Prediction'].mean()
    
    # Disparate Impact Ratio
    disparate_impact = rate_minority / rate_non_minority if rate_non_minority > 0 else 0
    
    # Statistical parity difference
    statistical_parity_diff = rate_minority - rate_non_minority
    
    return {
        'group_sizes': {
            'minority': len(minority),
            'non_minority': len(non_minority)
        },
        'selection_rates': {
            'minority': round(rate_minority, 3),
            'non_minority': round(rate_non_minority, 3)
        },
        'disparate_impact_ratio': round(disparate_impact, 3),
        'statistical_parity_difference': round(statistical_parity_diff, 3),
        'passes_4_5ths_rule': disparate_impact >= 0.8,
        'bias_detected': disparate_impact < 0.8 or disparate_impact > 1.25,
    }


def compare_ml_vs_maut(df: pd.DataFrame, 
                        ml_predictions: np.ndarray,
                        frame: str = 'Balanced') -> Dict:
    """
    Compare ML predictions with MAUT-based recommendations.
    
    This helps understand how the ML model differs from the
    explicit decision frame approach.
    """
    # Capitalize first letter to match column names
    frame_cap = frame.capitalize()
    score_col = f'Score_{frame_cap}'
    rank_col = f'Rank_{frame_cap}'
    
    # Get MAUT top 30% (same as hire_rate)
    threshold = df[score_col].quantile(0.7)
    maut_hire = (df[score_col] >= threshold).astype(int)
    
    # Compare predictions
    agreement = (ml_predictions == maut_hire).mean()
    
    # Cases where they disagree
    disagree_mask = ml_predictions != maut_hire
    ml_hire_maut_reject = ((ml_predictions == 1) & (maut_hire == 0)).sum()
    ml_reject_maut_hire = ((ml_predictions == 0) & (maut_hire == 1)).sum()
    
    # Analyze disagreement by demographics
    disagree_df = df[disagree_mask]
    
    return {
        'agreement_rate': round(agreement, 3),
        'total_disagreements': disagree_mask.sum(),
        'ml_hire_maut_reject': ml_hire_maut_reject,
        'ml_reject_maut_hire': ml_reject_maut_hire,
        'disagreement_minority_rate': disagree_df['x_Minority'].mean() if len(disagree_df) > 0 else 0,
    }


def run_ml_analysis(df: pd.DataFrame, 
                    label_method: str = 'balanced',
                    hire_rate: float = 0.3) -> Dict:
    """
    Run complete ML analysis pipeline.
    
    Args:
        df: DataFrame with normalized scores (after apply_all_frames)
        label_method: Which MAUT frame to use for generating labels
        hire_rate: Proportion of candidates to label as "hire"
    
    Returns:
        Complete analysis results including model, metrics, and fairness
    """
    # Generate labels
    y = generate_hire_labels(df, method=label_method, hire_rate=hire_rate)
    
    # Prepare features
    X, feature_names = prepare_ml_features(df)
    
    # Train model
    training_results = train_random_forest(X, y)
    
    # Get full predictions for fairness analysis
    full_predictions = training_results['model'].predict(X)
    full_probabilities = training_results['model'].predict_proba(X)[:, 1]
    
    # Compute metrics
    test_metrics = compute_model_metrics(
        training_results['y_test'], 
        training_results['y_test_pred']
    )
    
    # Compute fairness metrics
    fairness_minority = compute_fairness_metrics(df, full_predictions, 'x_Minority')
    fairness_gender = compute_fairness_metrics(df, full_predictions, 'x_Gender')
    
    # Compare with MAUT
    maut_comparison = compare_ml_vs_maut(df, full_predictions, label_method)
    
    return {
        'model': training_results['model'],
        'feature_importance': training_results['feature_importance'],
        'cv_scores': training_results['cv_scores'],
        'test_metrics': test_metrics,
        'fairness': {
            'minority': fairness_minority,
            'gender': fairness_gender,
        },
        'maut_comparison': maut_comparison,
        'predictions': full_predictions,
        'probabilities': full_probabilities,
        'labels': y,
    }


def format_ml_report(results: Dict) -> str:
    """
    Format ML analysis results as a readable report.
    """
    report = []
    report.append("=" * 60)
    report.append("MACHINE LEARNING CLASSIFIER ANALYSIS")
    report.append("=" * 60)
    
    # Model Performance
    report.append("\n📊 MODEL PERFORMANCE:")
    report.append(f"  Cross-Validation Accuracy: {results['cv_scores'].mean():.1%} (±{results['cv_scores'].std():.1%})")
    report.append(f"  Test Accuracy: {results['test_metrics']['accuracy']:.1%}")
    report.append(f"  Precision: {results['test_metrics']['precision']:.1%}")
    report.append(f"  Recall: {results['test_metrics']['recall']:.1%}")
    report.append(f"  F1 Score: {results['test_metrics']['f1']:.1%}")
    
    # Feature Importance
    report.append("\n🎯 FEATURE IMPORTANCE:")
    for _, row in results['feature_importance'].iterrows():
        bar = "█" * int(row['Importance'] * 20)
        report.append(f"  {row['Feature'].replace('U_x_', ''):<12} {bar} {row['Importance']:.3f}")
    
    # Fairness Analysis
    report.append("\n⚖️ FAIRNESS ANALYSIS (Minority):")
    fm = results['fairness']['minority']
    report.append(f"  Selection Rate (Minority): {fm['selection_rates']['minority']:.1%}")
    report.append(f"  Selection Rate (Non-Minority): {fm['selection_rates']['non_minority']:.1%}")
    report.append(f"  Disparate Impact Ratio: {fm['disparate_impact_ratio']:.3f}")
    report.append(f"  Passes 4/5ths Rule: {'✓ Yes' if fm['passes_4_5ths_rule'] else '✗ No - BIAS DETECTED'}")
    
    report.append("\n⚖️ FAIRNESS ANALYSIS (Gender):")
    fg = results['fairness']['gender']
    report.append(f"  Selection Rate (Female): {fg['selection_rates']['minority']:.1%}")
    report.append(f"  Selection Rate (Male): {fg['selection_rates']['non_minority']:.1%}")
    report.append(f"  Disparate Impact Ratio: {fg['disparate_impact_ratio']:.3f}")
    report.append(f"  Passes 4/5ths Rule: {'✓ Yes' if fg['passes_4_5ths_rule'] else '✗ No - BIAS DETECTED'}")
    
    # MAUT Comparison
    report.append("\n🔄 ML vs MAUT COMPARISON:")
    mc = results['maut_comparison']
    report.append(f"  Agreement Rate: {mc['agreement_rate']:.1%}")
    report.append(f"  Total Disagreements: {mc['total_disagreements']}")
    report.append(f"  ML hires but MAUT rejects: {mc['ml_hire_maut_reject']}")
    report.append(f"  ML rejects but MAUT hires: {mc['ml_reject_maut_hire']}")
    
    report.append("\n" + "=" * 60)
    
    return "\n".join(report)


if __name__ == "__main__":
    from data_generator import generate_mock_candidates, apply_data_integrity_mitigation
    from maut_scorer import apply_all_frames
    
    # Generate and prepare data
    df = generate_mock_candidates(150)
    df = apply_data_integrity_mitigation(df)
    df = apply_all_frames(df)
    
    # Run ML analysis
    results = run_ml_analysis(df, label_method='balanced', hire_rate=0.3)
    
    # Print report
    print(format_ml_report(results))
