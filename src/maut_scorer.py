"""
Phase 2: Multi-Criteria Assessment using MAUT (Multiattribute Utility Theory)
Implements normalization, decision frames, and weighted aggregation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


# ============================================================================
# DECISION FRAMES: Weight configurations for different priorities
# ============================================================================

DECISION_FRAMES = {
    'Efficiency': {
        'description': 'Maximizes skills and experience (normative approach)',
        'weights': {
            'x_Skills': 0.45,
            'x_Cultural': 0.25,
            'x_Ethics': 0.15,
            'x_Equity': 0.05,  # Combined Gender + Minority consideration
            'x_Integrity': 0.10,
        }
    },
    'Equity': {
        'description': 'Prioritizes demographic diversity and fairness (constructive approach)',
        'weights': {
            'x_Skills': 0.15,
            'x_Cultural': 0.20,
            'x_Ethics': 0.15,
            'x_Equity': 0.40,  # Combined Gender + Minority consideration
            'x_Integrity': 0.10,
        }
    },
    'Balanced': {
        'description': 'Seeks compromise between efficiency and equity objectives',
        'weights': {
            'x_Skills': 0.25,
            'x_Cultural': 0.20,
            'x_Ethics': 0.15,
            'x_Equity': 0.25,
            'x_Integrity': 0.15,
        }
    }
}


def normalize_min_max(series: pd.Series, maximize: bool = True) -> pd.Series:
    """
    Normalize a series to [0, 1] using min-max normalization.
    
    Formula: U_i(x) = (x - x_min) / (x_max - x_min)
    
    Args:
        series: Raw score series
        maximize: If True, higher values = higher utility. If False, reverse.
    """
    x_min, x_max = series.min(), series.max()
    
    if x_max == x_min:
        return pd.Series(0.5, index=series.index)
    
    normalized = (series - x_min) / (x_max - x_min)
    
    if not maximize:
        normalized = 1 - normalized
    
    return normalized


def compute_equity_score(df: pd.DataFrame) -> pd.Series:
    """
    Compute a combined equity score based on Gender and Minority status.
    
    This score represents the contribution to organizational diversity.
    Higher scores for underrepresented groups.
    """
    # Equity score: weighted combination of gender and minority representation
    # Female (x_Gender=1) and Minority (x_Minority=1) get higher equity scores
    equity_base = df['x_Gender'] * 0.5 + df['x_Minority'] * 0.5
    
    # Add small random component to differentiate within groups
    np.random.seed(42)
    noise = np.random.uniform(0, 0.1, len(df))
    
    equity_score = (equity_base + noise).clip(0, 1)
    return equity_score


def normalize_all_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize all candidate attributes to utility scores in [0, 1].
    Returns a new DataFrame with normalized utility columns.
    """
    df_norm = df.copy()
    
    # Normalize continuous attributes (higher is better)
    for col in ['x_Skills', 'x_Cultural', 'x_Ethics', 'x_Integrity']:
        df_norm[f'U_{col}'] = normalize_min_max(df[col], maximize=True)
    
    # Compute and add equity score
    df_norm['x_Equity'] = compute_equity_score(df)
    df_norm['U_x_Equity'] = df_norm['x_Equity']  # Already in [0, 1]
    
    return df_norm


def compute_maut_score(df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """
    Compute the MAUT aggregate utility score.
    
    Formula: U(Candidate) = Σ k_i * U_i(x)
    
    Args:
        df: DataFrame with normalized utility columns (U_x_*)
        weights: Dictionary of weights for each attribute
    """
    total_score = pd.Series(0.0, index=df.index)
    
    for attr, weight in weights.items():
        utility_col = f'U_{attr}'
        if utility_col in df.columns:
            total_score += weight * df[utility_col]
    
    return total_score


def apply_decision_frame(df: pd.DataFrame, frame_name: str) -> pd.DataFrame:
    """
    Apply a specific decision frame to compute candidate scores.
    
    Returns DataFrame with the frame's utility score and ranking.
    """
    if frame_name not in DECISION_FRAMES:
        raise ValueError(f"Unknown frame: {frame_name}. Available: {list(DECISION_FRAMES.keys())}")
    
    frame = DECISION_FRAMES[frame_name]
    weights = frame['weights']
    
    df = df.copy()
    score_col = f'Score_{frame_name}'
    rank_col = f'Rank_{frame_name}'
    
    df[score_col] = compute_maut_score(df, weights)
    df[rank_col] = df[score_col].rank(ascending=False, method='min').astype(int)
    
    return df


def apply_all_frames(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all decision frames and compute scores for each.
    Returns DataFrame with scores and rankings for all frames.
    """
    # First normalize all attributes
    df = normalize_all_attributes(df)
    
    # Apply each decision frame
    for frame_name in DECISION_FRAMES.keys():
        df = apply_decision_frame(df, frame_name)
    
    return df


def compute_demographic_parity(df: pd.DataFrame, score_col: str, 
                                threshold_percentile: float = 70) -> Dict:
    """
    Compute demographic parity ratio for a given score column.
    
    Demographic Parity Ratio = P(Selected | Minority) / P(Selected | Non-Minority)
    
    A ratio of 1.0 indicates perfect parity.
    Ratio < 0.8 often indicates potential disparate impact.
    """
    threshold = np.percentile(df[score_col], threshold_percentile)
    df_selected = df[df[score_col] >= threshold]
    
    # Selection rates by group
    total_minority = df['x_Minority'].sum()
    total_non_minority = len(df) - total_minority
    
    selected_minority = df_selected['x_Minority'].sum()
    selected_non_minority = len(df_selected) - selected_minority
    
    rate_minority = selected_minority / total_minority if total_minority > 0 else 0
    rate_non_minority = selected_non_minority / total_non_minority if total_non_minority > 0 else 0
    
    parity_ratio = rate_minority / rate_non_minority if rate_non_minority > 0 else 0
    
    return {
        'threshold': threshold,
        'selected_count': len(df_selected),
        'minority_selected': selected_minority,
        'non_minority_selected': selected_non_minority,
        'rate_minority': round(rate_minority, 3),
        'rate_non_minority': round(rate_non_minority, 3),
        'demographic_parity_ratio': round(parity_ratio, 3),
        'fair': parity_ratio >= 0.8 and parity_ratio <= 1.25,
    }


def get_top_candidates(df: pd.DataFrame, frame_name: str, n: int = 10) -> pd.DataFrame:
    """Get top N candidates for a specific decision frame."""
    rank_col = f'Rank_{frame_name}'
    return df[df[rank_col] <= n].sort_values(rank_col)


def compare_frames(df: pd.DataFrame, top_n: int = 10) -> Dict:
    """
    Compare how top candidates differ across decision frames.
    """
    results = {}
    
    for frame_name in DECISION_FRAMES.keys():
        top_candidates = get_top_candidates(df, frame_name, top_n)
        score_col = f'Score_{frame_name}'
        
        results[frame_name] = {
            'top_candidates': top_candidates['Candidate_ID'].tolist(),
            'avg_score': top_candidates[score_col].mean().round(3),
            'minority_count': top_candidates['x_Minority'].sum(),
            'female_count': top_candidates['x_Gender'].sum(),
            'demographic_parity': compute_demographic_parity(df, score_col),
        }
    
    # Calculate overlap between frames
    efficiency_set = set(results['Efficiency']['top_candidates'])
    equity_set = set(results['Equity']['top_candidates'])
    balanced_set = set(results['Balanced']['top_candidates'])
    
    results['overlap'] = {
        'efficiency_equity': len(efficiency_set & equity_set),
        'efficiency_balanced': len(efficiency_set & balanced_set),
        'equity_balanced': len(equity_set & balanced_set),
        'all_three': len(efficiency_set & equity_set & balanced_set),
    }
    
    return results


if __name__ == "__main__":
    from data_generator import generate_mock_candidates, apply_data_integrity_mitigation
    
    # Generate test data
    df = generate_mock_candidates(100)
    df = apply_data_integrity_mitigation(df)
    
    # Apply all decision frames
    df = apply_all_frames(df)
    
    print("=" * 60)
    print("MAUT SCORING RESULTS")
    print("=" * 60)
    
    for frame_name in DECISION_FRAMES.keys():
        print(f"\n--- {frame_name} Frame ---")
        print(f"Description: {DECISION_FRAMES[frame_name]['description']}")
        print(f"Weights: {DECISION_FRAMES[frame_name]['weights']}")
        
        top_10 = get_top_candidates(df, frame_name, 10)
        print(f"\nTop 10 Candidates:")
        print(top_10[['Candidate_ID', f'Score_{frame_name}', 'x_Skills', 'x_Gender', 'x_Minority']].to_string())
    
    print("\n" + "=" * 60)
    print("FRAME COMPARISON")
    print("=" * 60)
    comparison = compare_frames(df, 10)
    
    for frame_name in DECISION_FRAMES.keys():
        print(f"\n{frame_name}:")
        print(f"  Minority in Top 10: {comparison[frame_name]['minority_count']}")
        print(f"  Female in Top 10: {comparison[frame_name]['female_count']}")
        print(f"  Demographic Parity Ratio: {comparison[frame_name]['demographic_parity']['demographic_parity_ratio']}")
    
    print(f"\nOverlap Analysis:")
    print(f"  Efficiency ∩ Equity: {comparison['overlap']['efficiency_equity']} candidates")
    print(f"  All Three Frames: {comparison['overlap']['all_three']} candidates")
