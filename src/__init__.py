"""
src/__init__.py
DFA Hiring Prototype - Source Package
"""

from .data_generator import (
    generate_mock_candidates,
    apply_data_integrity_mitigation,
    save_dataset,
    get_dataset_summary
)

from .maut_scorer import (
    DECISION_FRAMES,
    normalize_all_attributes,
    apply_all_frames,
    compute_demographic_parity,
    get_top_candidates,
    compare_frames
)

from .sensitivity_analysis import (
    run_sobol_analysis,
    run_weight_sweep,
    format_sensitivity_results,
    define_weight_problem,
    interpret_sensitivity,
    generate_3d_pareto_data
)

from .visualizations import (
    generate_all_visualizations,
    plot_parallel_coordinates,
    plot_pareto_front_2d,
    plot_pareto_front_3d,
    plot_frame_comparison,
    plot_gsa_tornado,
    plot_correlation_heatmap
)

from .ml_classifier import (
    run_ml_analysis,
    generate_hire_labels,
    train_random_forest,
    compute_fairness_metrics,
    format_ml_report
)

__all__ = [
    # Data generation
    'generate_mock_candidates',
    'apply_data_integrity_mitigation', 
    'save_dataset',
    'get_dataset_summary',
    
    # MAUT scoring
    'DECISION_FRAMES',
    'normalize_all_attributes',
    'apply_all_frames',
    'compute_demographic_parity',
    'get_top_candidates',
    'compare_frames',
    
    # Sensitivity analysis
    'run_sobol_analysis',
    'run_weight_sweep',
    'format_sensitivity_results',
    'define_weight_problem',
    'interpret_sensitivity',
    'generate_3d_pareto_data',
    
    # Visualizations
    'generate_all_visualizations',
    'plot_parallel_coordinates',
    'plot_pareto_front_2d',
    'plot_pareto_front_3d',
    'plot_frame_comparison',
    'plot_gsa_tornado',
    'plot_correlation_heatmap',
    
    # ML Classifier
    'run_ml_analysis',
    'generate_hire_labels',
    'train_random_forest',
    'compute_fairness_metrics',
    'format_ml_report',
]
