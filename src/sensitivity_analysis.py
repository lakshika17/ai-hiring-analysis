"""
Phase 3: Global Sensitivity Analysis (GSA) using Variance-based Sobol Method
Implements DOE with Sobol sequences and calculates sensitivity indices.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from SALib.sample import saltelli
from SALib.analyze import sobol


def define_weight_problem() -> Dict:
    """
    Define the GSA problem for weight sensitivity analysis.
    
    We analyze how uncertainty in the decision weights (k_i) affects
    the ranking outcomes. Each weight can vary within a defined range.
    """
    problem = {
        'num_vars': 5,
        'names': ['k_Skills', 'k_Cultural', 'k_Ethics', 'k_Equity', 'k_Integrity'],
        'bounds': [
            [0.1, 0.5],   # k_Skills: 10% to 50%
            [0.1, 0.4],   # k_Cultural: 10% to 40%
            [0.05, 0.3],  # k_Ethics: 5% to 30%
            [0.05, 0.5],  # k_Equity: 5% to 50%
            [0.05, 0.2],  # k_Integrity: 5% to 20%
        ]
    }
    return problem


def generate_sobol_samples(problem: Dict, n_samples: int = 1024) -> np.ndarray:
    """
    Generate Sobol sequence samples for the weight space.
    
    Uses Saltelli's extension for efficient computation of both
    first-order and total-order sensitivity indices.
    
    Total evaluations = N * (2D + 2) where D = number of variables
    For 5 variables and N=1024: 1024 * 12 = 12,288 samples
    """
    samples = saltelli.sample(problem, n_samples, calc_second_order=False)
    return samples


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Normalize weight samples so they sum to 1."""
    return weights / weights.sum(axis=1, keepdims=True)


def evaluate_model(df_normalized: pd.DataFrame, weight_samples: np.ndarray) -> np.ndarray:
    """
    Evaluate the hiring model for each weight configuration.
    
    For each weight sample, compute the candidate scores and return
    a metric that captures the ranking behavior (e.g., variance in
    minority representation among top candidates).
    
    Returns array of output values for sensitivity analysis.
    """
    n_samples = len(weight_samples)
    outputs = np.zeros(n_samples)
    
    # Define attribute mapping
    attr_cols = ['U_x_Skills', 'U_x_Cultural', 'U_x_Ethics', 'U_x_Equity', 'U_x_Integrity']
    
    for i, raw_weights in enumerate(weight_samples):
        # Normalize weights to sum to 1
        weights = raw_weights / raw_weights.sum()
        
        # Compute weighted score for each candidate
        scores = np.zeros(len(df_normalized))
        for j, col in enumerate(attr_cols):
            if col in df_normalized.columns:
                scores += weights[j] * df_normalized[col].values
        
        # Output metric: proportion of minorities in top 30%
        threshold = np.percentile(scores, 70)
        selected = scores >= threshold
        minority_mask = df_normalized['x_Minority'].values == 1
        
        if selected.sum() > 0:
            minority_rate = (selected & minority_mask).sum() / selected.sum()
        else:
            minority_rate = 0
        
        outputs[i] = minority_rate
    
    return outputs


def evaluate_model_ranking_variance(df_normalized: pd.DataFrame, 
                                     weight_samples: np.ndarray) -> np.ndarray:
    """
    Alternative output: Measure how much rankings change with weight variations.
    
    Returns the average rank of minority candidates (lower = better for minorities).
    """
    n_samples = len(weight_samples)
    outputs = np.zeros(n_samples)
    
    attr_cols = ['U_x_Skills', 'U_x_Cultural', 'U_x_Ethics', 'U_x_Equity', 'U_x_Integrity']
    
    for i, raw_weights in enumerate(weight_samples):
        weights = raw_weights / raw_weights.sum()
        
        scores = np.zeros(len(df_normalized))
        for j, col in enumerate(attr_cols):
            if col in df_normalized.columns:
                scores += weights[j] * df_normalized[col].values
        
        # Compute rankings
        rankings = pd.Series(scores).rank(ascending=False).values
        
        # Output: average rank of minority candidates
        minority_mask = df_normalized['x_Minority'].values == 1
        avg_minority_rank = rankings[minority_mask].mean()
        
        outputs[i] = avg_minority_rank
    
    return outputs


def run_sobol_analysis(df_normalized: pd.DataFrame, 
                       n_samples: int = 1024,
                       output_type: str = 'minority_rate') -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Run complete Sobol sensitivity analysis.
    
    Args:
        df_normalized: DataFrame with normalized utility columns
        n_samples: Number of base samples (total = n_samples * 12 for 5 vars)
        output_type: 'minority_rate' or 'ranking_variance'
    
    Returns:
        Tuple of (sensitivity_indices, samples, outputs)
    """
    # Define the problem
    problem = define_weight_problem()
    
    # Generate Sobol samples
    samples = generate_sobol_samples(problem, n_samples)
    
    # Evaluate model
    if output_type == 'minority_rate':
        outputs = evaluate_model(df_normalized, samples)
    else:
        outputs = evaluate_model_ranking_variance(df_normalized, samples)
    
    # Compute Sobol indices
    Si = sobol.analyze(problem, outputs, calc_second_order=False)
    
    return Si, samples, outputs


def format_sensitivity_results(Si: Dict, problem: Dict) -> pd.DataFrame:
    """
    Format sensitivity analysis results into a readable DataFrame.
    """
    results = pd.DataFrame({
        'Factor': problem['names'],
        'S1': Si['S1'],  # First-order index
        'S1_conf': Si['S1_conf'],  # Confidence interval
        'ST': Si['ST'],  # Total-order index
        'ST_conf': Si['ST_conf'],  # Confidence interval
    })
    
    results['S1'] = results['S1'].round(4)
    results['ST'] = results['ST'].round(4)
    results['S1_conf'] = results['S1_conf'].round(4)
    results['ST_conf'] = results['ST_conf'].round(4)
    
    # Sort by total sensitivity
    results = results.sort_values('ST', ascending=False).reset_index(drop=True)
    
    return results


def interpret_sensitivity(results_df: pd.DataFrame) -> List[str]:
    """
    Generate interpretive statements about the sensitivity analysis.
    """
    interpretations = []
    
    # Most influential factor
    top_factor = results_df.iloc[0]
    interpretations.append(
        f"Most influential factor: {top_factor['Factor']} (ST = {top_factor['ST']:.3f})"
    )
    
    # Interaction effects
    for _, row in results_df.iterrows():
        interaction = row['ST'] - row['S1']
        if interaction > 0.1:
            interpretations.append(
                f"{row['Factor']} has significant interactions with other factors "
                f"(interaction effect = {interaction:.3f})"
            )
    
    # Low sensitivity factors
    low_sens = results_df[results_df['ST'] < 0.05]
    if len(low_sens) > 0:
        factors = ', '.join(low_sens['Factor'].tolist())
        interpretations.append(
            f"Low sensitivity factors (can be fixed): {factors}"
        )
    
    return interpretations


def run_weight_sweep(df_normalized: pd.DataFrame, 
                     n_points: int = 50) -> pd.DataFrame:
    """
    Run a weight sweep varying k_Skills vs k_Equity ratio.
    
    This generates data for Pareto front visualization.
    """
    results = []
    
    for i in range(n_points + 1):
        # Vary the skills-equity trade-off
        skill_weight = i / n_points * 0.8 + 0.1  # 0.1 to 0.9
        equity_weight = (n_points - i) / n_points * 0.8 + 0.1  # 0.9 to 0.1
        
        # Fixed weights for other factors
        weights = np.array([
            skill_weight,  # Skills
            0.15,          # Cultural
            0.10,          # Ethics
            equity_weight, # Equity
            0.10,          # Integrity
        ])
        weights = weights / weights.sum()  # Normalize
        
        # Compute scores
        attr_cols = ['U_x_Skills', 'U_x_Cultural', 'U_x_Ethics', 'U_x_Equity', 'U_x_Integrity']
        scores = np.zeros(len(df_normalized))
        for j, col in enumerate(attr_cols):
            if col in df_normalized.columns:
                scores += weights[j] * df_normalized[col].values
        
        # Compute metrics for top 30%
        threshold = np.percentile(scores, 70)
        selected = scores >= threshold
        
        # Efficiency metric: average skills score of selected
        efficiency = df_normalized.loc[selected, 'U_x_Skills'].mean()
        
        # Equity metric: minority representation in selected
        minority_rate = df_normalized.loc[selected, 'x_Minority'].mean()
        
        # Cultural fit metric
        cultural = df_normalized.loc[selected, 'U_x_Cultural'].mean()
        
        results.append({
            'skill_weight': weights[0],
            'equity_weight': weights[3],
            'efficiency_score': efficiency,
            'equity_score': minority_rate,
            'cultural_score': cultural,
            'n_selected': selected.sum(),
        })
    
    return pd.DataFrame(results)


def generate_3d_pareto_data(df_normalized: pd.DataFrame, 
                            n_points: int = 20) -> pd.DataFrame:
    """
    Generate 3D Pareto front data by sweeping two weight dimensions.
    
    Creates a grid of weight combinations for k_Skills, k_Equity, and k_Cultural
    to visualize the 3D trade-off surface.
    
    Returns DataFrame with efficiency, equity, and cultural scores for each
    weight combination.
    """
    results = []
    
    for i in range(n_points + 1):
        for j in range(n_points + 1 - i):
            # Create weight combination ensuring sum <= 0.9 (leave room for Ethics/Integrity)
            skill_ratio = i / n_points
            equity_ratio = j / n_points
            cultural_ratio = (n_points - i - j) / n_points
            
            # Scale to actual weights (leaving 20% for Ethics + Integrity)
            skill_weight = skill_ratio * 0.6 + 0.1
            equity_weight = equity_ratio * 0.5 + 0.05
            cultural_weight = cultural_ratio * 0.4 + 0.1
            
            weights = np.array([
                skill_weight,     # Skills
                cultural_weight,  # Cultural
                0.10,             # Ethics (fixed)
                equity_weight,    # Equity
                0.10,             # Integrity (fixed)
            ])
            weights = weights / weights.sum()  # Normalize
            
            # Compute scores
            attr_cols = ['U_x_Skills', 'U_x_Cultural', 'U_x_Ethics', 'U_x_Equity', 'U_x_Integrity']
            scores = np.zeros(len(df_normalized))
            for k, col in enumerate(attr_cols):
                if col in df_normalized.columns:
                    scores += weights[k] * df_normalized[col].values
            
            # Compute metrics for top 30%
            threshold = np.percentile(scores, 70)
            selected = scores >= threshold
            
            if selected.sum() > 0:
                # Efficiency: average skills score
                efficiency = df_normalized.loc[selected, 'U_x_Skills'].mean()
                # Equity: minority representation
                equity = df_normalized.loc[selected, 'x_Minority'].mean()
                # Cultural fit
                cultural = df_normalized.loc[selected, 'U_x_Cultural'].mean()
                
                results.append({
                    'skill_weight': weights[0],
                    'equity_weight': weights[3],
                    'cultural_weight': weights[1],
                    'efficiency_score': efficiency,
                    'equity_score': equity,
                    'cultural_score': cultural,
                    'n_selected': selected.sum(),
                })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    from data_generator import generate_mock_candidates, apply_data_integrity_mitigation
    from maut_scorer import normalize_all_attributes
    
    # Generate and prepare data
    df = generate_mock_candidates(100)
    df = apply_data_integrity_mitigation(df)
    df = normalize_all_attributes(df)
    
    print("=" * 60)
    print("GLOBAL SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    # Run Sobol analysis
    print("\nRunning Sobol analysis (this may take a moment)...")
    Si, samples, outputs = run_sobol_analysis(df, n_samples=512)
    
    # Format and display results
    problem = define_weight_problem()
    results_df = format_sensitivity_results(Si, problem)
    
    print("\nSensitivity Indices:")
    print(results_df.to_string())
    
    print("\nInterpretation:")
    for interp in interpret_sensitivity(results_df):
        print(f"  • {interp}")
    
    # Run weight sweep for Pareto front
    print("\n" + "=" * 60)
    print("WEIGHT SWEEP FOR PARETO FRONT")
    print("=" * 60)
    
    sweep_results = run_weight_sweep(df, n_points=20)
    print("\nEfficiency vs Equity Trade-off:")
    print(sweep_results[['skill_weight', 'equity_weight', 'efficiency_score', 'equity_score']].to_string())
