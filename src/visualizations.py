"""
Phase 4: Visualization Module
Implements all visualization types for the DFA Hiring Prototype:
- Parallel Axis Plots
- 2D Pareto Fronts
- 3D Pareto Fronts
- Frame Comparison Charts
- GSA Tornado Charts
- Correlation Heatmaps
"""

import numpy as np
import pandas as pd

# Use non-interactive backend to prevent tkinter thread issues
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Optional


# Set style for matplotlib
plt.style.use('seaborn-v0_8-whitegrid')


def ensure_output_dir(output_dir: str = None) -> Path:
    """Ensure output directory exists."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / 'outputs' / 'figures'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ============================================================================
# 1. PARALLEL AXIS PLOTS
# ============================================================================

def plot_parallel_coordinates(df: pd.DataFrame, 
                               color_by: str = 'Score_Balanced',
                               title: str = 'Candidate Multi-Objective Trade-offs',
                               output_path: str = None) -> go.Figure:
    """
    Create an interactive parallel coordinates plot showing trade-offs
    across all candidate attributes.
    
    This visualization allows stakeholders to see how candidates perform
    across multiple objectives simultaneously.
    """
    # Select columns for parallel coordinates
    dimensions = [
        dict(label='Skills', values=df['U_x_Skills']),
        dict(label='Cultural Fit', values=df['U_x_Cultural']),
        dict(label='Ethics', values=df['U_x_Ethics']),
        dict(label='Equity', values=df['U_x_Equity']),
        dict(label='Integrity', values=df['U_x_Integrity']),
    ]
    
    # Add score columns if available
    for frame in ['Efficiency', 'Equity', 'Balanced']:
        score_col = f'Score_{frame}'
        if score_col in df.columns:
            dimensions.append(dict(label=f'{frame} Score', values=df[score_col]))
    
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=df[color_by] if color_by in df.columns else df['U_x_Skills'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=color_by.replace('_', ' '))
            ),
            dimensions=dimensions,
            labelangle=-30,
            labelside='bottom'
        )
    )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18), x=0.5, xanchor='center'),
        font=dict(size=12),
        height=600,
        margin=dict(l=100, r=100, t=80, b=80)
    )
    
    if output_path:
        fig.write_html(
            output_path,
            full_html=True,
            include_plotlyjs=True,
            config={'displayModeBar': True}
        )
        _add_centering_css(output_path)
        print(f"Saved: {output_path}")
    
    return fig


def plot_parallel_coordinates_by_frame(df: pd.DataFrame,
                                        output_dir: str = None) -> Dict[str, go.Figure]:
    """
    Create parallel coordinates plots highlighting top candidates per frame.
    """
    output_dir = ensure_output_dir(output_dir)
    figures = {}
    
    for frame in ['Efficiency', 'Equity', 'Balanced']:
        rank_col = f'Rank_{frame}'
        if rank_col not in df.columns:
            continue
        
        # Create highlight column
        df_plot = df.copy()
        df_plot['Highlight'] = (df_plot[rank_col] <= 10).astype(int)
        
        fig = plot_parallel_coordinates(
            df_plot,
            color_by=f'Score_{frame}',
            title=f'Parallel Coordinates - {frame} Frame (Top 10 Highlighted)',
            output_path=str(output_dir / f'parallel_coords_{frame.lower()}.html')
        )
        figures[frame] = fig
    
    return figures


# ============================================================================
# 2. 2D PARETO FRONT
# ============================================================================

def plot_pareto_front_2d(sweep_df: pd.DataFrame,
                          x_col: str = 'efficiency_score',
                          y_col: str = 'equity_score',
                          title: str = 'Pareto Front: Efficiency vs Equity Trade-off',
                          output_path: str = None) -> go.Figure:
    """
    Create a 2D Pareto front visualization showing the trade-off between
    efficiency (skills-based selection) and equity (diversity).
    """
    # Identify Pareto-optimal points
    pareto_mask = identify_pareto_points(sweep_df[[x_col, y_col]].values)
    
    fig = go.Figure()
    
    # All points
    fig.add_trace(go.Scatter(
        x=sweep_df[x_col],
        y=sweep_df[y_col],
        mode='markers',
        marker=dict(size=10, color='lightblue', line=dict(width=1, color='darkblue')),
        name='All Configurations',
        hovertemplate=(
            f'Efficiency: %{{x:.3f}}<br>'
            f'Equity: %{{y:.3f}}<br>'
            f'Skill Weight: %{{customdata[0]:.2f}}<br>'
            f'Equity Weight: %{{customdata[1]:.2f}}'
        ),
        customdata=sweep_df[['skill_weight', 'equity_weight']].values
    ))
    
    # Pareto front line
    pareto_df = sweep_df[pareto_mask].sort_values(x_col)
    fig.add_trace(go.Scatter(
        x=pareto_df[x_col],
        y=pareto_df[y_col],
        mode='lines+markers',
        line=dict(color='red', width=3),
        marker=dict(size=12, color='red', symbol='diamond'),
        name='Pareto Front'
    ))
    
    # Add annotations for extreme points
    fig.add_annotation(
        x=sweep_df[x_col].max(),
        y=sweep_df.loc[sweep_df[x_col].idxmax(), y_col],
        text="Max Efficiency",
        showarrow=True,
        arrowhead=2
    )
    fig.add_annotation(
        x=sweep_df.loc[sweep_df[y_col].idxmax(), x_col],
        y=sweep_df[y_col].max(),
        text="Max Equity",
        showarrow=True,
        arrowhead=2
    )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18), x=0.5, xanchor='center'),
        xaxis_title='Efficiency Score (Average Skills of Selected)',
        yaxis_title='Equity Score (Minority Representation)',
        legend=dict(x=0.02, y=0.98),
        height=600,
        width=800
    )
    
    if output_path:
        # Center the plot on the webpage
        fig.write_html(
            output_path,
            full_html=True,
            include_plotlyjs=True,
            config={'displayModeBar': True},
            div_id='pareto-2d',
            default_width='100%',
            default_height='100%'
        )
        # Add centering CSS
        _add_centering_css(output_path)
        print(f"Saved: {output_path}")
    
    return fig


def _add_centering_css(html_path: str):
    """Add CSS to center the plot on the webpage."""
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    centering_css = '''
    <style>
        body {
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            padding: 20px;
            box-sizing: border-box;
            background-color: #f5f5f5;
        }
        .plotly-graph-div {
            margin: 0 auto;
        }
    </style>
    '''
    
    # Insert CSS after <head> tag
    if '<head>' in content:
        content = content.replace('<head>', '<head>' + centering_css)
    else:
        content = centering_css + content
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(content)


def identify_pareto_points(points: np.ndarray) -> np.ndarray:
    """
    Identify Pareto-optimal points (non-dominated solutions).
    For maximization of both objectives.
    """
    n_points = len(points)
    pareto_mask = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                # Check if point j dominates point i
                if (points[j, 0] >= points[i, 0] and 
                    points[j, 1] >= points[i, 1] and
                    (points[j, 0] > points[i, 0] or points[j, 1] > points[i, 1])):
                    pareto_mask[i] = False
                    break
    
    return pareto_mask


# ============================================================================
# 3. 3D PARETO FRONT
# ============================================================================

def plot_pareto_front_3d(sweep_df: pd.DataFrame,
                          x_col: str = 'efficiency_score',
                          y_col: str = 'equity_score',
                          z_col: str = 'cultural_score',
                          title: str = '3D Pareto Front: Efficiency vs Equity vs Cultural Fit',
                          output_path: str = None) -> go.Figure:
    """
    Create a 3D Pareto front visualization showing trade-offs between
    three objectives simultaneously.
    """
    # Identify 3D Pareto-optimal points
    pareto_mask = identify_pareto_points_3d(
        sweep_df[[x_col, y_col, z_col]].values
    )
    
    fig = go.Figure()
    
    # All points
    fig.add_trace(go.Scatter3d(
        x=sweep_df[x_col],
        y=sweep_df[y_col],
        z=sweep_df[z_col],
        mode='markers',
        marker=dict(
            size=6,
            color=sweep_df['skill_weight'],
            colorscale='Viridis',
            colorbar=dict(title='Skill Weight', x=1.1),
            opacity=0.7
        ),
        name='All Configurations',
        hovertemplate=(
            f'Efficiency: %{{x:.3f}}<br>'
            f'Equity: %{{y:.3f}}<br>'
            f'Cultural: %{{z:.3f}}'
        )
    ))
    
    # Pareto front points
    pareto_df = sweep_df[pareto_mask]
    fig.add_trace(go.Scatter3d(
        x=pareto_df[x_col],
        y=pareto_df[y_col],
        z=pareto_df[z_col],
        mode='markers',
        marker=dict(size=10, color='red', symbol='diamond'),
        name='Pareto Front'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18), x=0.5, xanchor='center'),
        scene=dict(
            xaxis_title='Efficiency Score',
            yaxis_title='Equity Score',
            zaxis_title='Cultural Fit Score',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        height=700,
        width=900,
        legend=dict(x=0.02, y=0.98)
    )
    
    if output_path:
        fig.write_html(
            output_path,
            full_html=True,
            include_plotlyjs=True,
            config={'displayModeBar': True}
        )
        _add_centering_css(output_path)
        print(f"Saved: {output_path}")
    
    return fig


def identify_pareto_points_3d(points: np.ndarray) -> np.ndarray:
    """
    Identify Pareto-optimal points in 3D (non-dominated solutions).
    For maximization of all three objectives.
    """
    n_points = len(points)
    pareto_mask = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                # Check if point j dominates point i
                if (points[j, 0] >= points[i, 0] and 
                    points[j, 1] >= points[i, 1] and
                    points[j, 2] >= points[i, 2] and
                    (points[j, 0] > points[i, 0] or 
                     points[j, 1] > points[i, 1] or
                     points[j, 2] > points[i, 2])):
                    pareto_mask[i] = False
                    break
    
    return pareto_mask


def generate_3d_pareto_data(df_normalized: pd.DataFrame, 
                            n_samples: int = 200) -> pd.DataFrame:
    """
    Generate weight configurations for 3D Pareto front visualization.
    Varies three weights: Skills, Equity, and Cultural.
    """
    np.random.seed(42)
    results = []
    
    # Generate random weight combinations
    for _ in range(n_samples):
        # Random weights for main three factors
        main_weights = np.random.dirichlet(np.ones(3))
        skill_w = main_weights[0] * 0.7 + 0.1  # Scale to reasonable range
        equity_w = main_weights[1] * 0.7 + 0.1
        cultural_w = main_weights[2] * 0.7 + 0.1
        
        # Fixed small weights for ethics and integrity
        ethics_w = 0.1
        integrity_w = 0.1
        
        weights = np.array([skill_w, cultural_w, ethics_w, equity_w, integrity_w])
        weights = weights / weights.sum()
        
        # Compute scores
        attr_cols = ['U_x_Skills', 'U_x_Cultural', 'U_x_Ethics', 'U_x_Equity', 'U_x_Integrity']
        scores = np.zeros(len(df_normalized))
        for j, col in enumerate(attr_cols):
            if col in df_normalized.columns:
                scores += weights[j] * df_normalized[col].values
        
        # Compute metrics for top 30%
        threshold = np.percentile(scores, 70)
        selected = scores >= threshold
        
        efficiency = df_normalized.loc[selected, 'U_x_Skills'].mean()
        equity = df_normalized.loc[selected, 'x_Minority'].mean()
        cultural = df_normalized.loc[selected, 'U_x_Cultural'].mean()
        
        results.append({
            'skill_weight': weights[0],
            'cultural_weight': weights[1],
            'equity_weight': weights[3],
            'efficiency_score': efficiency,
            'equity_score': equity,
            'cultural_score': cultural,
        })
    
    return pd.DataFrame(results)


# ============================================================================
# 4. FRAME COMPARISON CHARTS
# ============================================================================

def plot_frame_comparison(df: pd.DataFrame,
                          top_n: int = 10,
                          output_path: str = None) -> go.Figure:
    """
    Create a grouped bar chart comparing top candidates across decision frames.
    """
    frames = ['Efficiency', 'Equity', 'Balanced']
    
    # Get top candidates for each frame
    comparison_data = []
    for frame in frames:
        rank_col = f'Rank_{frame}'
        score_col = f'Score_{frame}'
        top_df = df[df[rank_col] <= top_n].copy()
        for _, row in top_df.iterrows():
            comparison_data.append({
                'Candidate': row['Candidate_ID'],
                'Frame': frame,
                'Score': row[score_col],
                'Minority': 'Yes' if row['x_Minority'] == 1 else 'No',
                'Gender': 'Female' if row['x_Gender'] == 1 else 'Male'
            })
    
    comp_df = pd.DataFrame(comparison_data)
    
    fig = px.bar(
        comp_df,
        x='Candidate',
        y='Score',
        color='Frame',
        barmode='group',
        pattern_shape='Minority',
        title=f'Top {top_n} Candidates by Decision Frame',
        color_discrete_map={
            'Efficiency': '#2ecc71',
            'Equity': '#3498db',
            'Balanced': '#9b59b6'
        }
    )
    
    fig.update_layout(
        title=dict(x=0.5, xanchor='center'),
        xaxis_title='Candidate ID',
        yaxis_title='MAUT Score',
        legend_title='Decision Frame',
        height=500,
        xaxis_tickangle=-45
    )
    
    if output_path:
        fig.write_html(
            output_path,
            full_html=True,
            include_plotlyjs=True,
            config={'displayModeBar': True}
        )
        _add_centering_css(output_path)
        print(f"Saved: {output_path}")
    
    return fig


def plot_frame_demographics(df: pd.DataFrame,
                            top_n: int = 10,
                            output_path: str = None) -> plt.Figure:
    """
    Create a stacked bar chart showing demographic composition of top candidates
    under each decision frame.
    """
    frames = ['Efficiency', 'Equity', 'Balanced']
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    for idx, frame in enumerate(frames):
        rank_col = f'Rank_{frame}'
        top_df = df[df[rank_col] <= top_n]
        
        # Count demographics
        minority_yes = top_df['x_Minority'].sum()
        minority_no = top_n - minority_yes
        female = top_df['x_Gender'].sum()
        male = top_n - female
        
        # Create stacked bars
        categories = ['Minority Status', 'Gender']
        yes_values = [minority_yes, female]
        no_values = [minority_no, male]
        
        x = np.arange(len(categories))
        width = 0.6
        
        axes[idx].bar(x, yes_values, width, label='Minority/Female', color='#3498db')
        axes[idx].bar(x, no_values, width, bottom=yes_values, label='Non-Minority/Male', color='#ecf0f1')
        
        axes[idx].set_ylabel('Count')
        axes[idx].set_title(f'{frame} Frame\n(Top {top_n})')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(categories)
        axes[idx].set_ylim(0, top_n + 1)
        axes[idx].legend(loc='upper right', fontsize=8)
        
        # Add count labels
        for i, (y, n) in enumerate(zip(yes_values, no_values)):
            axes[idx].text(i, y/2, str(y), ha='center', va='center', fontweight='bold')
            axes[idx].text(i, y + n/2, str(n), ha='center', va='center')
    
    plt.suptitle('Demographic Composition by Decision Frame', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close(fig)
    
    return fig


# ============================================================================
# 5. GSA TORNADO CHART
# ============================================================================

def plot_gsa_tornado(sensitivity_df: pd.DataFrame,
                      title: str = 'Global Sensitivity Analysis: Weight Influence',
                      output_path: str = None) -> plt.Figure:
    """
    Create a tornado chart showing sensitivity indices for each weight factor.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by total sensitivity
    df_sorted = sensitivity_df.sort_values('ST', ascending=True)
    
    y_pos = np.arange(len(df_sorted))
    
    # Plot total-order indices
    bars = ax.barh(y_pos, df_sorted['ST'], height=0.4, 
                   label='Total Order (ST)', color='#e74c3c', alpha=0.8)
    
    # Plot first-order indices
    ax.barh(y_pos - 0.2, df_sorted['S1'], height=0.4,
            label='First Order (S1)', color='#3498db', alpha=0.8)
    
    # Error bars for confidence
    ax.errorbar(df_sorted['ST'], y_pos, xerr=df_sorted['ST_conf'], 
                fmt='none', color='darkred', capsize=3)
    ax.errorbar(df_sorted['S1'], y_pos - 0.2, xerr=df_sorted['S1_conf'],
                fmt='none', color='darkblue', capsize=3)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted['Factor'].str.replace('k_', ''))
    ax.set_xlabel('Sensitivity Index')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Add value labels
    for i, (s1, st) in enumerate(zip(df_sorted['S1'], df_sorted['ST'])):
        ax.text(st + 0.02, i, f'{st:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close(fig)
    
    return fig


# ============================================================================
# 6. CORRELATION HEATMAP (BIAS VISUALIZATION)
# ============================================================================

def plot_correlation_heatmap(df: pd.DataFrame,
                              title: str = 'Attribute Correlation Matrix (Embedded Bias Visualization)',
                              output_path: str = None) -> plt.Figure:
    """
    Create a heatmap showing correlations between candidate attributes.
    This helps visualize the embedded bias (e.g., negative correlation
    between Minority status and Skills).
    """
    # Select relevant columns
    cols = ['x_Skills', 'x_Cultural', 'x_Ethics', 'x_Integrity', 'x_Gender', 'x_Minority']
    corr_df = df[cols].copy()
    corr_df.columns = ['Skills', 'Cultural', 'Ethics', 'Integrity', 'Gender', 'Minority']
    
    corr_matrix = corr_df.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
        ax=ax
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add annotation about bias
    bias_corr = corr_matrix.loc['Minority', 'Skills']
    ax.annotate(
        f'Note: Minority-Skills correlation ({bias_corr:.2f})\nreflects embedded historical bias',
        xy=(0.5, -0.1), xycoords='axes fraction',
        ha='center', fontsize=10, style='italic',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close(fig)
    
    return fig


# ============================================================================
# 7. SUMMARY DASHBOARD
# ============================================================================

def create_summary_dashboard(df: pd.DataFrame,
                              sweep_df: pd.DataFrame,
                              sensitivity_df: pd.DataFrame,
                              output_path: str = None) -> go.Figure:
    """
    Create a comprehensive dashboard combining key visualizations.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Pareto Front: Efficiency vs Equity',
            'Demographic Parity by Frame',
            'Sensitivity Analysis',
            'Score Distribution by Minority Status'
        ),
        specs=[[{'type': 'scatter'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'box'}]]
    )
    
    # 1. Pareto Front (simplified)
    fig.add_trace(
        go.Scatter(
            x=sweep_df['efficiency_score'],
            y=sweep_df['equity_score'],
            mode='markers+lines',
            marker=dict(color=sweep_df['skill_weight'], colorscale='Viridis'),
            name='Pareto Front'
        ),
        row=1, col=1
    )
    
    # 2. Demographic counts per frame
    for i, frame in enumerate(['Efficiency', 'Equity', 'Balanced']):
        rank_col = f'Rank_{frame}'
        if rank_col in df.columns:
            top_df = df[df[rank_col] <= 10]
            minority_count = top_df['x_Minority'].sum()
            fig.add_trace(
                go.Bar(
                    x=[frame],
                    y=[minority_count],
                    name=frame,
                    marker_color=['#2ecc71', '#3498db', '#9b59b6'][i]
                ),
                row=1, col=2
            )
    
    # 3. Sensitivity indices
    fig.add_trace(
        go.Bar(
            x=sensitivity_df['Factor'].str.replace('k_', ''),
            y=sensitivity_df['ST'],
            name='Total Sensitivity',
            marker_color='#e74c3c'
        ),
        row=2, col=1
    )
    
    # 4. Score distribution by minority status
    for status, name in [(0, 'Non-Minority'), (1, 'Minority')]:
        subset = df[df['x_Minority'] == status]
        if 'Score_Balanced' in subset.columns:
            fig.add_trace(
                go.Box(
                    y=subset['Score_Balanced'],
                    name=name,
                    marker_color='#3498db' if status == 1 else '#ecf0f1'
                ),
                row=2, col=2
            )
    
    fig.update_layout(
        height=800,
        width=1200,
        title=dict(
            text='DFA Hiring Decision Prototype - Summary Dashboard',
            x=0.5,
            xanchor='center'
        ),
        showlegend=False
    )
    
    if output_path:
        fig.write_html(
            output_path,
            full_html=True,
            include_plotlyjs=True,
            config={'displayModeBar': True}
        )
        _add_centering_css(output_path)
        print(f"Saved: {output_path}")
    
    return fig


# ============================================================================
# 8. ML CLASSIFIER VISUALIZATIONS
# ============================================================================

def plot_ml_analysis(results: Dict, df: pd.DataFrame,
                     output_dir: str = None) -> Dict[str, str]:
    """
    Generate visualizations for ML classifier analysis.
    
    Args:
        results: Dictionary from run_ml_analysis()
        df: DataFrame with candidate data
        output_dir: Output directory for figures
    
    Returns:
        Dictionary of {visualization_name: file_path}
    """
    output_dir = ensure_output_dir(output_dir)
    generated = {}
    
    print("\n8. ML Classifier Visualizations...")
    
    # 1. Feature Importance Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    fi = results['feature_importance']
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(fi)))
    bars = ax.barh(fi['Feature'].str.replace('U_x_', ''), fi['Importance'], color=colors)
    ax.set_xlabel('Importance')
    ax.set_title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    for bar, val in zip(bars, fi['Importance']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center')
    plt.tight_layout()
    path = str(output_dir / 'ml_feature_importance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    generated['ml_feature_importance'] = path
    print(f"Saved: {path}")
    plt.close()
    
    # 2. Fairness Comparison Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    groups = ['Minority', 'Non-Minority', 'Female', 'Male']
    rates = [
        results['fairness']['minority']['selection_rates']['minority'],
        results['fairness']['minority']['selection_rates']['non_minority'],
        results['fairness']['gender']['selection_rates']['minority'],
        results['fairness']['gender']['selection_rates']['non_minority'],
    ]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    bars = ax.bar(groups, rates, color=colors)
    ax.axhline(y=0.3, color='red', linestyle='--', label='Target Hire Rate (30%)')
    ax.set_ylabel('Selection Rate')
    ax.set_title('ML Model Selection Rates by Demographic Group', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(rates) * 1.3 if rates else 1)
    ax.legend()
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.1%}', ha='center')
    plt.tight_layout()
    path = str(output_dir / 'ml_fairness_rates.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    generated['ml_fairness_rates'] = path
    print(f"Saved: {path}")
    plt.close()
    
    # 3. Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = results['test_metrics']['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Hire', 'Hire'], yticklabels=['No Hire', 'Hire'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = str(output_dir / 'ml_confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    generated['ml_confusion_matrix'] = path
    print(f"Saved: {path}")
    plt.close()
    
    # 4. Prediction Distribution by Group (Interactive)
    df_plot = df.copy()
    df_plot['ML_Prediction'] = results['predictions']
    df_plot['ML_Probability'] = results['probabilities']
    df_plot['Group'] = df_plot.apply(
        lambda r: 'Minority' if r['x_Minority'] == 1 else 'Non-Minority', axis=1
    )
    
    fig = px.histogram(
        df_plot, x='ML_Probability', color='Group', 
        nbins=20, barmode='overlay', opacity=0.7,
        title='ML Hire Probability Distribution by Group',
        labels={'ML_Probability': 'Hire Probability', 'count': 'Count'}
    )
    fig.add_vline(x=0.5, line_dash='dash', line_color='red', 
                  annotation_text='Decision Threshold')
    fig.update_layout(title=dict(x=0.5, xanchor='center'))
    path = str(output_dir / 'ml_probability_distribution.html')
    fig.write_html(path)
    _add_centering_css(path)
    generated['ml_probability_distribution'] = path
    print(f"Saved: {path}")
    
    # 5. ML vs MAUT Agreement Visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    mc = results['maut_comparison']
    categories = ['Agreement', 'ML Hire/MAUT Reject', 'ML Reject/MAUT Hire']
    total = len(df)
    values = [
        total - mc['total_disagreements'],
        mc['ml_hire_maut_reject'],
        mc['ml_reject_maut_hire']
    ]
    colors = ['#2ecc71', '#e74c3c', '#f39c12']
    wedges, texts, autotexts = ax.pie(values, labels=categories, colors=colors,
                                        autopct='%1.1f%%', startangle=90)
    ax.set_title('ML vs MAUT Decision Agreement', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = str(output_dir / 'ml_maut_agreement.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    generated['ml_maut_agreement'] = path
    print(f"Saved: {path}")
    plt.close()
    
    print(f"Generated {len(generated)} ML visualizations")
    return generated


# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_all_visualizations(df: pd.DataFrame,
                                 sweep_df: pd.DataFrame,
                                 sensitivity_df: pd.DataFrame,
                                 output_dir: str = None) -> Dict[str, str]:
    """
    Generate all visualizations and save to output directory.
    Returns dict of {visualization_name: file_path}.
    """
    output_dir = ensure_output_dir(output_dir)
    generated = {}
    
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    # 1. Parallel Coordinates
    print("\n1. Parallel Axis Plots...")
    fig = plot_parallel_coordinates(
        df, color_by='Score_Balanced',
        output_path=str(output_dir / 'parallel_coordinates.html')
    )
    generated['parallel_coordinates'] = str(output_dir / 'parallel_coordinates.html')
    
    # 2. 2D Pareto Front
    print("2. 2D Pareto Front...")
    fig = plot_pareto_front_2d(
        sweep_df,
        output_path=str(output_dir / 'pareto_front_2d.html')
    )
    generated['pareto_front_2d'] = str(output_dir / 'pareto_front_2d.html')
    
    # 3. 3D Pareto Front
    print("3. 3D Pareto Front...")
    sweep_3d = generate_3d_pareto_data(df, n_samples=200)
    fig = plot_pareto_front_3d(
        sweep_3d,
        output_path=str(output_dir / 'pareto_front_3d.html')
    )
    generated['pareto_front_3d'] = str(output_dir / 'pareto_front_3d.html')
    
    # 4. Frame Comparison
    print("4. Frame Comparison Charts...")
    fig = plot_frame_comparison(
        df, top_n=10,
        output_path=str(output_dir / 'frame_comparison.html')
    )
    generated['frame_comparison'] = str(output_dir / 'frame_comparison.html')
    
    plot_frame_demographics(
        df, top_n=10,
        output_path=str(output_dir / 'frame_demographics.png')
    )
    generated['frame_demographics'] = str(output_dir / 'frame_demographics.png')
    
    # 5. GSA Tornado Chart
    print("5. GSA Tornado Chart...")
    plot_gsa_tornado(
        sensitivity_df,
        output_path=str(output_dir / 'gsa_tornado.png')
    )
    generated['gsa_tornado'] = str(output_dir / 'gsa_tornado.png')
    
    # 6. Correlation Heatmap
    print("6. Correlation Heatmap...")
    plot_correlation_heatmap(
        df,
        output_path=str(output_dir / 'correlation_heatmap.png')
    )
    generated['correlation_heatmap'] = str(output_dir / 'correlation_heatmap.png')
    
    # 7. Summary Dashboard
    print("7. Summary Dashboard...")
    fig = create_summary_dashboard(
        df, sweep_df, sensitivity_df,
        output_path=str(output_dir / 'dashboard.html')
    )
    generated['dashboard'] = str(output_dir / 'dashboard.html')
    
    print("\n" + "=" * 60)
    print(f"Generated {len(generated)} visualizations in: {output_dir}")
    print("=" * 60)
    
    return generated


if __name__ == "__main__":
    # Test with sample data
    from data_generator import generate_mock_candidates, apply_data_integrity_mitigation
    from maut_scorer import apply_all_frames
    from sensitivity_analysis import run_sobol_analysis, run_weight_sweep, format_sensitivity_results, define_weight_problem
    
    print("Generating test visualizations...")
    
    # Generate data
    df = generate_mock_candidates(100)
    df = apply_data_integrity_mitigation(df)
    df = apply_all_frames(df)
    
    # Run analyses
    sweep_df = run_weight_sweep(df, n_points=50)
    Si, _, _ = run_sobol_analysis(df, n_samples=256)
    sensitivity_df = format_sensitivity_results(Si, define_weight_problem())
    
    # Generate all visualizations
    generated = generate_all_visualizations(df, sweep_df, sensitivity_df)
    
    for name, path in generated.items():
        print(f"  {name}: {path}")
