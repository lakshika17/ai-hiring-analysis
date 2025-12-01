"""
Decision Framing and Analytics (DFA) - AI in Hiring Decisions Prototype
========================================================================

This prototype demonstrates how different decision frames (priorities)
affect candidate selection outcomes in AI-assisted hiring, enabling
A Posteriori Preference Articulation through visualization of trade-offs.

Phases:
1. Mock Data Generation with Embedded Bias
2. Multi-Criteria Assessment using MAUT
3. Global Sensitivity Analysis (Sobol Method)
4. Visualization of Trade-offs (Pareto Fronts, Parallel Axes)
5. Machine Learning Classifier with Fairness Analysis

Author: DFA Project Team
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from src.data_generator import (
    generate_mock_candidates,
    apply_data_integrity_mitigation,
    save_dataset,
    get_dataset_summary
)
from src.maut_scorer import (
    DECISION_FRAMES,
    apply_all_frames,
    compare_frames,
    compute_demographic_parity
)
from src.sensitivity_analysis import (
    run_sobol_analysis,
    run_weight_sweep,
    format_sensitivity_results,
    define_weight_problem,
    interpret_sensitivity
)
from src.visualizations import generate_all_visualizations, plot_ml_analysis
from src.ml_classifier import run_ml_analysis, format_ml_report


def print_header(text: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70)


def main():
    """
    Main pipeline for the DFA Hiring Prototype.
    Executes all 4 phases and generates visualizations.
    """
    print_header("DFA HIRING DECISION PROTOTYPE")
    print("AI in Hiring Decisions - Decision Framing and Analytics\n")
    
    # =========================================================================
    # PHASE 1: Mock Data Generation
    # =========================================================================
    print_header("PHASE 1: Mock Data Generation")
    
    # Increased from 150 to 500 candidates for:
    # - Better ML model training and generalization
    # - Smoother visualizations and Pareto fronts
    # - More reliable sensitivity analysis
    # - Statistically significant bias detection
    N_CANDIDATES = 500
    
    print(f"Generating {N_CANDIDATES} synthetic candidates with embedded bias...")
    df = generate_mock_candidates(n_candidates=N_CANDIDATES, random_seed=42)
    df = apply_data_integrity_mitigation(df)
    
    # Save dataset
    output_path = save_dataset(df)
    print(f"Dataset saved to: {output_path}")
    
    # Display summary
    summary = get_dataset_summary(df)
    print("\nDataset Summary:")
    print(f"  Total Candidates: {summary['total_candidates']}")
    print(f"  Female: {summary['female_count']} ({summary['female_pct']}%)")
    print(f"  Minority: {summary['minority_count']} ({summary['minority_pct']}%)")
    print(f"\nEmbedded Bias Detection:")
    print(f"  Avg Skills (Non-Minority): {summary['avg_skills_non_minority']}")
    print(f"  Avg Skills (Minority): {summary['avg_skills_minority']}")
    print(f"  Skills Gap: {summary['skills_gap']} points (bias indicator)")
    
    # =========================================================================
    # PHASE 2: Multi-Criteria Assessment (MAUT)
    # =========================================================================
    print_header("PHASE 2: Multi-Criteria Assessment (MAUT)")
    
    print("Applying decision frames and computing MAUT scores...")
    df = apply_all_frames(df)
    
    print("\nDecision Frames:")
    for frame_name, frame_config in DECISION_FRAMES.items():
        print(f"\n  {frame_name}:")
        print(f"    Description: {frame_config['description']}")
        weights_str = ", ".join([f"{k.replace('x_', '')}: {v:.0%}" 
                                  for k, v in frame_config['weights'].items()])
        print(f"    Weights: {weights_str}")
    
    # Compare frames
    comparison = compare_frames(df, top_n=10)
    
    print("\n" + "-" * 50)
    print("TOP 10 CANDIDATE ANALYSIS BY FRAME")
    print("-" * 50)
    
    for frame_name in DECISION_FRAMES.keys():
        frame_data = comparison[frame_name]
        parity = frame_data['demographic_parity']
        
        print(f"\n{frame_name} Frame:")
        print(f"  Top Candidates: {', '.join(frame_data['top_candidates'][:5])}...")
        print(f"  Minority in Top 10: {frame_data['minority_count']}/10")
        print(f"  Female in Top 10: {frame_data['female_count']}/10")
        print(f"  Demographic Parity Ratio: {parity['demographic_parity_ratio']:.3f}", end="")
        print(" ✓ Fair" if parity['fair'] else " ✗ Potential Disparate Impact")
    
    print(f"\nOverlap Analysis:")
    print(f"  Efficiency ∩ Equity: {comparison['overlap']['efficiency_equity']} candidates")
    print(f"  Common to All Frames: {comparison['overlap']['all_three']} candidates")
    
    # =========================================================================
    # PHASE 3: Global Sensitivity Analysis
    # =========================================================================
    print_header("PHASE 3: Global Sensitivity Analysis (GSA)")
    
    print("Running Sobol variance-based sensitivity analysis...")
    print("(Analyzing how weight uncertainty affects minority representation)")
    
    # Increased Sobol samples from 512 to 1024 for more stable sensitivity indices
    Si, samples, outputs = run_sobol_analysis(df, n_samples=1024, output_type='minority_rate')
    
    problem = define_weight_problem()
    sensitivity_df = format_sensitivity_results(Si, problem)
    
    print("\nSensitivity Indices (sorted by Total-Order ST):")
    print("-" * 50)
    print(f"{'Factor':<15} {'S1 (First)':<12} {'ST (Total)':<12} {'Interaction':<12}")
    print("-" * 50)
    for _, row in sensitivity_df.iterrows():
        interaction = row['ST'] - row['S1']
        factor = row['Factor'].replace('k_', '')
        print(f"{factor:<15} {row['S1']:<12.4f} {row['ST']:<12.4f} {interaction:<12.4f}")
    
    print("\nInterpretation:")
    for interp in interpret_sensitivity(sensitivity_df):
        print(f"  • {interp}")
    
    # Run weight sweep for Pareto front
    print("\nGenerating Pareto front data (weight sweep)...")
    # Increased sweep points from 50 to 100 for smoother Pareto front curves
    sweep_df = run_weight_sweep(df, n_points=100)
    
    # =========================================================================
    # PHASE 4: Visualization Generation
    # =========================================================================
    print_header("PHASE 4: Visualization Generation")
    
    print("Generating all visualizations...")
    generated = generate_all_visualizations(df, sweep_df, sensitivity_df)
    
    print("\nGenerated Files:")
    for name, path in generated.items():
        print(f"  • {name}: {path}")
    
    # =========================================================================
    # PHASE 5: Machine Learning Classifier
    # =========================================================================
    print_header("PHASE 5: Machine Learning Classifier")
    
    print("Training Random Forest classifier for hire/no-hire prediction...")
    print("(Using Balanced frame MAUT scores as training labels)")
    
    ml_results = run_ml_analysis(df, label_method='balanced', hire_rate=0.3)
    print(format_ml_report(ml_results))
    
    # Add ML predictions to DataFrame
    df['ML_Prediction'] = ml_results['predictions']
    df['ML_Probability'] = ml_results['probabilities']
    
    # Generate ML visualizations
    print("\nGenerating ML visualizations...")
    ml_generated = plot_ml_analysis(ml_results, df)
    generated.update(ml_generated)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_header("PROTOTYPE EXECUTION COMPLETE")
    
    print("""
Key Findings:
─────────────
1. EMBEDDED BIAS DETECTION: The DFA analysis reveals a {:.1f}-point skills gap
   between minority and non-minority candidates, demonstrating how AI-assisted
   analysis can uncover subtle historical discrimination patterns.

2. FRAME IMPACT: AI-assisted decision framing produces transparent trade-offs:
   - Efficiency frame: {} minorities in top 10 (skills-focused)
   - Equity frame: {} minorities in top 10 (diversity-focused)
   - Balanced frame: {} minorities in top 10 (compromise solution)

3. SENSITIVITY INSIGHT: The {} weight has the highest sensitivity (ST = {:.3f}),
   helping organizations understand which priorities most affect outcomes.

4. PARETO-OPTIMAL SOLUTIONS: The visualizations show ALL efficient trade-offs,
   enabling stakeholders to make INFORMED decisions rather than arbitrary ones.

5. VALUE OF AI-ASSISTED DFA:
   - Uncovers hidden bias in historical data
   - Makes trade-offs explicit and transparent
   - Identifies high-performing diverse candidates that might be overlooked
   - Supports A Posteriori Preference Articulation (choose after seeing options)

6. ML CLASSIFIER INSIGHTS:
   - Random Forest model trained on MAUT Balanced frame labels
   - Fairness analysis reveals potential bias in ML predictions
   - Comparison with MAUT shows agreement rate: {:.1%}

Visualizations saved to: outputs/figures/
    """.format(
        summary['skills_gap'],
        comparison['Efficiency']['minority_count'],
        comparison['Equity']['minority_count'],
        comparison['Balanced']['minority_count'],
        sensitivity_df.iloc[0]['Factor'].replace('k_', ''),
        sensitivity_df.iloc[0]['ST'],
        ml_results['maut_comparison']['agreement_rate']
    ))
    
    return df, sweep_df, sensitivity_df, ml_results


if __name__ == "__main__":
    df, sweep_df, sensitivity_df, ml_results = main()
