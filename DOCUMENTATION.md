# Decision Framing and Analytics (DFA) - AI in Hiring Decisions

## Complete Technical Documentation

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Architecture](#2-project-architecture)
3. [Phase 1: Mock Data Generation](#3-phase-1-mock-data-generation)
4. [Phase 2: Multi-Criteria Assessment (MAUT)](#4-phase-2-multi-criteria-assessment-maut)
5. [Phase 3: Global Sensitivity Analysis (GSA)](#5-phase-3-global-sensitivity-analysis-gsa)
6. [Phase 4: Visualization Generation](#6-phase-4-visualization-generation)
7. [Phase 5: Machine Learning Classifier](#7-phase-5-machine-learning-classifier)
8. [Output Files](#8-output-files)
9. [Key Findings & Interpretations](#9-key-findings--interpretations)
10. [Technical Dependencies](#10-technical-dependencies)

---

## 1. Project Overview

### What is This Project?

This project is a **Decision Framing and Analytics (DFA) Prototype** that demonstrates how AI-assisted decision-making can be applied to hiring decisions. It showcases:

- How different **decision frames** (priorities) affect candidate selection outcomes
- How to detect and visualize **embedded bias** in historical data
- How to enable **A Posteriori Preference Articulation** (choosing AFTER seeing all trade-offs)
- How **Machine Learning** models can be analyzed for fairness

### Core Concept: Decision Framing

**Decision Framing** refers to how the same hiring problem can be approached with different priority weights:

| Frame | Philosophy | Primary Focus |
|-------|-----------|---------------|
| **Efficiency** | Normative approach | Maximize skills and experience |
| **Equity** | Constructive approach | Prioritize demographic diversity |
| **Balanced** | Compromise solution | Balance skills with diversity goals |

### Why This Matters

Traditional AI hiring systems make opaque decisions. This prototype makes the trade-offs **explicit and transparent**, allowing stakeholders to:
1. See ALL Pareto-optimal solutions (no dominated choices)
2. Understand how weight changes affect outcomes
3. Detect hidden bias in training data
4. Make informed decisions rather than arbitrary ones

---

## 2. Project Architecture

### File Structure

```
DFA Project/
├── main.py                    # Main orchestrator - runs all 5 phases
├── pyproject.toml             # Project dependencies (uv/pip)
├── README.md                  # Basic project info
├── DOCUMENTATION.md           # This file
│
├── src/                       # Source modules
│   ├── __init__.py
│   ├── data_generator.py      # Phase 1: Synthetic data with bias
│   ├── maut_scorer.py         # Phase 2: MAUT multi-criteria scoring
│   ├── sensitivity_analysis.py # Phase 3: Sobol GSA
│   ├── visualizations.py      # Phase 4: All plots and charts
│   └── ml_classifier.py       # Phase 5: Random Forest + fairness
│
├── data/
│   └── mock_candidates.csv    # Generated candidate dataset
│
└── outputs/
    └── figures/               # All generated visualizations
        ├── parallel_coordinates.html
        ├── pareto_front_2d.html
        ├── pareto_front_3d.html
        ├── frame_comparison.html
        ├── frame_demographics.png
        ├── gsa_tornado.png
        ├── correlation_heatmap.png
        ├── dashboard.html
        ├── ml_feature_importance.png
        ├── ml_fairness_rates.png
        ├── ml_confusion_matrix.png
        ├── ml_probability_distribution.html
        └── ml_maut_agreement.png
```

### Execution Flow

```
main.py
    │
    ├──► Phase 1: generate_mock_candidates() ──► mock_candidates.csv
    │
    ├──► Phase 2: apply_all_frames() ──► MAUT scores & rankings
    │
    ├──► Phase 3: run_sobol_analysis() ──► Sensitivity indices
    │
    ├──► Phase 4: generate_all_visualizations() ──► 8 visualizations
    │
    └──► Phase 5: run_ml_analysis() ──► ML model + 5 visualizations
```

---

## 3. Phase 1: Mock Data Generation

### File: `src/data_generator.py`

### Purpose
Generate synthetic candidate data with **embedded historical bias** to simulate real-world discrimination patterns that AI systems might learn from.

### Candidate Attributes

| Attribute | Type | Range | Description |
|-----------|------|-------|-------------|
| `Candidate_ID` | String | C001-C500 | Unique identifier |
| `x_Skills` | Float | 20-100 | Technical competencies and experience |
| `x_Cultural` | Float | 20-100 | Cultural alignment and organizational fit |
| `x_Ethics` | Float | 25-100 | Ethics and professionalism score |
| `x_Gender` | Binary | 0/1 | Gender indicator (0=Male, 1=Female) |
| `x_Minority` | Binary | 0/1 | Minority/historically excluded status |
| `x_Integrity` | Float | 30-100 | Data integrity/credential validation |

### Embedded Bias Mechanism

The generator **intentionally embeds subtle bias** to simulate historical discrimination:

```python
# EMBEDDED BIAS: 10% reduction in skills for minorities
bias_factor = 0.90
minority_mask = df['x_Minority'] == 1
df.loc[minority_mask, 'x_Skills'] = df.loc[minority_mask, 'x_Skills'] * bias_factor
```

This creates a **5-6 point skills gap** between minority and non-minority candidates, mimicking how historical prejudice might appear in real training data.

### High-Performing Minority Injection

To demonstrate why DFA helps find overlooked talent:

```python
# 5% exceptional minority candidates with high scores
n_high_performers = int(n_candidates * 0.05)
df.loc[high_perf_indices, 'x_Skills'] = np.random.uniform(75, 90, ...)
```

### Data Integrity Mitigation

Flags candidates with suspiciously high scores across ALL metrics:

```python
df['Integrity_Flag'] = (
    (df['x_Skills'] > 85) & 
    (df['x_Cultural'] > 85) & 
    (df['x_Ethics'] > 85)
).astype(int)
# Apply 5% penalty to integrity score for flagged candidates
```

### Key Functions

| Function | Purpose |
|----------|---------|
| `generate_mock_candidates(n, seed)` | Creates n synthetic candidates |
| `apply_data_integrity_mitigation(df)` | Flags suspicious candidates |
| `save_dataset(df, path)` | Exports to CSV |
| `get_dataset_summary(df)` | Returns statistics including bias metrics |

---

## 4. Phase 2: Multi-Criteria Assessment (MAUT)

### File: `src/maut_scorer.py`

### Purpose
Apply **Multiattribute Utility Theory (MAUT)** to compute aggregate scores under different decision frames.

### Mathematical Foundation

**MAUT Formula:**
$$U(\text{Candidate}) = \sum_{i=1}^{n} k_i \cdot U_i(x)$$

Where:
- $U_i(x)$ = Normalized utility of attribute $i$ (range 0-1)
- $k_i$ = Weight for attribute $i$ (sums to 1.0)

### Normalization

All attributes are normalized using **min-max scaling**:

$$U_i(x) = \frac{x - x_{min}}{x_{max} - x_{min}}$$

```python
def normalize_min_max(series, maximize=True):
    normalized = (series - series.min()) / (series.max() - series.min())
    return normalized if maximize else 1 - normalized
```

### Decision Frames

Three pre-defined weight configurations:

#### Efficiency Frame
```python
'Efficiency': {
    'weights': {
        'x_Skills': 0.45,      # Heavily weighted
        'x_Cultural': 0.25,
        'x_Ethics': 0.15,
        'x_Equity': 0.05,      # Minimally weighted
        'x_Integrity': 0.10,
    }
}
```
**Philosophy:** Maximize organizational performance through skills-based selection.

#### Equity Frame
```python
'Equity': {
    'weights': {
        'x_Skills': 0.15,      # Reduced importance
        'x_Cultural': 0.20,
        'x_Ethics': 0.15,
        'x_Equity': 0.40,      # Heavily weighted
        'x_Integrity': 0.10,
    }
}
```
**Philosophy:** Prioritize demographic diversity and correct historical imbalances.

#### Balanced Frame
```python
'Balanced': {
    'weights': {
        'x_Skills': 0.25,
        'x_Cultural': 0.20,
        'x_Ethics': 0.15,
        'x_Equity': 0.25,
        'x_Integrity': 0.15,
    }
}
```
**Philosophy:** Seek compromise between efficiency and equity objectives.

### Equity Score Computation

The equity attribute combines gender and minority status:

```python
def compute_equity_score(df):
    equity_base = df['x_Gender'] * 0.5 + df['x_Minority'] * 0.5
    noise = np.random.uniform(0, 0.1, len(df))  # Differentiate within groups
    return (equity_base + noise).clip(0, 1)
```

### Demographic Parity Analysis

Measures fairness using the **4/5ths Rule**:

$$\text{Demographic Parity Ratio} = \frac{P(\text{Selected} | \text{Minority})}{P(\text{Selected} | \text{Non-Minority})}$$

- Ratio = 1.0 → Perfect parity
- Ratio < 0.8 → Potential disparate impact (legal concern)
- Ratio > 1.25 → Reverse discrimination concern

### Key Functions

| Function | Purpose |
|----------|---------|
| `normalize_all_attributes(df)` | Creates U_x_* columns |
| `compute_maut_score(df, weights)` | Calculates weighted aggregate |
| `apply_decision_frame(df, frame)` | Adds Score_* and Rank_* columns |
| `apply_all_frames(df)` | Applies all 3 frames |
| `compute_demographic_parity(df, score_col)` | Fairness metrics |
| `compare_frames(df, top_n)` | Cross-frame comparison |

---

## 5. Phase 3: Global Sensitivity Analysis (GSA)

### File: `src/sensitivity_analysis.py`

### Purpose
Determine which weight parameters have the **most influence** on hiring outcomes using variance-based **Sobol sensitivity analysis**.

### Why Sensitivity Analysis?

When decision-makers adjust weights, they need to know:
1. Which weights matter most for outcomes?
2. Which weights can be fixed without affecting results?
3. How do weights interact with each other?

### Sobol Method

The Sobol method decomposes output variance into contributions from each input:

$$V(Y) = \sum_i V_i + \sum_{i<j} V_{ij} + ... + V_{1,2,...,n}$$

**First-Order Index (S1):**
$$S_1 = \frac{V_i}{V(Y)}$$
Measures the direct effect of weight $i$ alone.

**Total-Order Index (ST):**
$$S_T = \frac{E[V(Y|X_{\sim i})]}{V(Y)}$$
Measures the total effect including all interactions.

### Weight Sampling Space

```python
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
```

### Saltelli Sampling

Uses Saltelli's extension of Sobol sequences:
- Total evaluations = N × (2D + 2)
- For N=1024 and D=5: **12,288 model evaluations**

```python
samples = saltelli.sample(problem, n_samples=1024, calc_second_order=False)
```

### Output Metric

The analysis tracks **minority representation rate** in top 30% of candidates:

```python
def evaluate_model(df, weight_samples):
    for weights in weight_samples:
        # Compute scores with these weights
        scores = compute_weighted_scores(df, weights)
        
        # Select top 30%
        threshold = np.percentile(scores, 70)
        selected = scores >= threshold
        
        # Output: minority representation
        minority_rate = (selected & minority_mask).sum() / selected.sum()
```

### Typical Results

| Factor | S1 (First-Order) | ST (Total-Order) | Interpretation |
|--------|-----------------|------------------|----------------|
| k_Equity | ~0.86 | ~0.90 | **Dominant factor** - controls minority representation |
| k_Skills | ~0.08 | ~0.10 | Secondary effect |
| k_Cultural | ~0.05 | ~0.06 | Minor effect |
| k_Ethics | ~0.01 | ~0.01 | Can be fixed |
| k_Integrity | ~0.00 | ~0.01 | Can be fixed |

### Weight Sweep for Pareto Front

Generates data for visualizing the efficiency-equity trade-off:

```python
def run_weight_sweep(df, n_points=100):
    for i in range(n_points + 1):
        skill_weight = i / n_points * 0.8 + 0.1  # 0.1 to 0.9
        equity_weight = (n_points - i) / n_points * 0.8 + 0.1
        
        # Compute efficiency (avg skills) and equity (minority %) for each config
```

### Key Functions

| Function | Purpose |
|----------|---------|
| `define_weight_problem()` | Sets up Sobol problem definition |
| `generate_sobol_samples(problem, n)` | Creates Saltelli samples |
| `evaluate_model(df, samples)` | Computes output for each sample |
| `run_sobol_analysis(df, n_samples)` | Full GSA pipeline |
| `run_weight_sweep(df, n_points)` | 2D Pareto front data |
| `generate_3d_pareto_data(df, n_samples)` | 3D Pareto front data |
| `format_sensitivity_results(Si, problem)` | Creates results DataFrame |
| `interpret_sensitivity(results_df)` | Generates text interpretations |

---

## 6. Phase 4: Visualization Generation

### File: `src/visualizations.py`

### Purpose
Generate interactive and static visualizations that enable stakeholders to explore trade-offs and make informed decisions.

### Technical Setup

```python
# Use non-interactive backend to prevent tkinter thread issues
import matplotlib
matplotlib.use('Agg')
```

### Visualization 1: Parallel Coordinates Plot

**File:** `parallel_coordinates.html`

**Purpose:** Show how each candidate performs across ALL attributes simultaneously.

**Technology:** Plotly interactive plot

**Features:**
- Each line = one candidate
- Each vertical axis = one attribute
- Color = Balanced frame score
- Interactive: hover, zoom, filter

```python
fig = go.Figure(data=go.Parcoords(
    line=dict(color=df['Score_Balanced'], colorscale='Viridis'),
    dimensions=[
        dict(label='Skills', values=df['U_x_Skills']),
        dict(label='Cultural Fit', values=df['U_x_Cultural']),
        # ... more dimensions
    ]
))
```

### Visualization 2: 2D Pareto Front

**File:** `pareto_front_2d.html`

**Purpose:** Show the fundamental trade-off between Efficiency and Equity.

**Key Concept:** **Pareto Optimality**
- A point is Pareto-optimal if no other point is better in ALL objectives
- The Pareto front shows ALL efficient solutions
- Points below the front are "dominated" (inferior)

```python
def identify_pareto_points(points):
    for i in range(n_points):
        for j in range(n_points):
            if point_j dominates point_i:
                pareto_mask[i] = False
```

**Interpretation:**
- Moving right → Higher efficiency (avg skills)
- Moving up → Higher equity (minority %)
- You cannot improve one without sacrificing the other

### Visualization 3: 3D Pareto Front

**File:** `pareto_front_3d.html`

**Purpose:** Add Cultural Fit as a third objective dimension.

**Features:**
- Interactive 3D rotation
- Color = skill weight used
- Diamond markers = Pareto-optimal points

### Visualization 4: Frame Comparison Charts

**Files:** `frame_comparison.html`, `frame_demographics.png`

**Purpose:** Compare which candidates rank highly under each decision frame.

**Grouped Bar Chart:** Shows MAUT scores for top candidates by frame

**Stacked Bar Chart:** Shows demographic composition:
- Minority count in top 10
- Gender distribution in top 10

### Visualization 5: GSA Tornado Chart

**File:** `gsa_tornado.png`

**Purpose:** Visualize sensitivity analysis results.

**Components:**
- Horizontal bars showing S1 (first-order) and ST (total-order) indices
- Error bars for confidence intervals
- Sorted by total sensitivity (most influential at top)

### Visualization 6: Correlation Heatmap

**File:** `correlation_heatmap.png`

**Purpose:** Reveal embedded bias through attribute correlations.

**Key Insight:** Negative correlation between Minority and Skills reveals the embedded historical bias.

```python
corr_matrix = df[['Skills', 'Cultural', 'Ethics', 'Integrity', 'Gender', 'Minority']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0)
```

### Visualization 7: Summary Dashboard

**File:** `dashboard.html`

**Purpose:** Single-page overview combining key insights.

**4-Panel Layout:**
1. Pareto Front (Efficiency vs Equity)
2. Demographic Parity by Frame
3. Sensitivity Indices
4. Score Distribution by Minority Status

### ML Visualizations (Phase 5)

| File | Purpose |
|------|---------|
| `ml_feature_importance.png` | Which features drive ML predictions |
| `ml_fairness_rates.png` | Selection rates by demographic group |
| `ml_confusion_matrix.png` | Prediction accuracy breakdown |
| `ml_probability_distribution.html` | Hire probability by group |
| `ml_maut_agreement.png` | ML vs MAUT decision comparison |

### Key Functions

| Function | Purpose |
|----------|---------|
| `plot_parallel_coordinates(df)` | Interactive parallel axes |
| `plot_pareto_front_2d(sweep_df)` | 2D trade-off visualization |
| `plot_pareto_front_3d(sweep_df)` | 3D trade-off visualization |
| `plot_frame_comparison(df)` | Grouped bar chart |
| `plot_frame_demographics(df)` | Stacked demographic bars |
| `plot_gsa_tornado(sensitivity_df)` | Sensitivity tornado chart |
| `plot_correlation_heatmap(df)` | Bias detection heatmap |
| `create_summary_dashboard(...)` | Combined 4-panel view |
| `plot_ml_analysis(results, df)` | All 5 ML visualizations |
| `generate_all_visualizations(...)` | Master generation function |

---

## 7. Phase 5: Machine Learning Classifier

### File: `src/ml_classifier.py`

### Purpose
Train a **Random Forest classifier** for hire/no-hire predictions and analyze it for **fairness and bias**.

### Why Include ML?

This phase demonstrates:
1. How ML can automate MAUT-based decisions
2. How to analyze ML models for demographic bias
3. How ML predictions differ from explicit decision frames

### Label Generation

Training labels are derived from MAUT Balanced frame scores:

```python
def generate_hire_labels(df, method='Balanced', hire_rate=0.3):
    threshold = df[f'Score_{method}'].quantile(1 - hire_rate)
    labels = (df[f'Score_{method}'] >= threshold).astype(int)
    return labels  # Top 30% = hire (1), rest = no-hire (0)
```

### Feature Selection

Only normalized utility scores are used (NOT demographic attributes):

```python
feature_cols = ['U_x_Skills', 'U_x_Cultural', 'U_x_Ethics', 'U_x_Integrity']
# Note: Gender and Minority are NOT features (would be illegal)
# But the model might learn proxies for these...
```

### Model Configuration

```python
model = RandomForestClassifier(
    n_estimators=200,        # 200 decision trees
    max_depth=8,             # Limit tree depth
    min_samples_split=10,    # Prevent overfitting
    min_samples_leaf=4,      # Smooth predictions
    class_weight='balanced', # Handle class imbalance
    n_jobs=-1                # Parallel processing
)
```

### Cross-Validation

Uses 10-fold stratified cross-validation:

```python
cv_scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
# Reports mean ± std accuracy
```

### Performance Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| Precision | TP/(TP+FP) | Of predicted hires, how many correct? |
| Recall | TP/(TP+FN) | Of actual hires, how many found? |
| F1 Score | 2×(P×R)/(P+R) | Harmonic mean of P and R |

### Fairness Analysis

Computes selection rates by demographic group:

```python
def compute_fairness_metrics(df, predictions, sensitive_col='x_Minority'):
    rate_minority = predictions[minority_mask].mean()
    rate_non_minority = predictions[~minority_mask].mean()
    
    disparate_impact = rate_minority / rate_non_minority
    
    return {
        'disparate_impact_ratio': disparate_impact,
        'passes_4_5ths_rule': disparate_impact >= 0.8,
        'bias_detected': disparate_impact < 0.8 or disparate_impact > 1.25,
    }
```

### ML vs MAUT Comparison

Identifies where ML and MAUT disagree:

```python
def compare_ml_vs_maut(df, ml_predictions, frame='Balanced'):
    maut_hire = (df[f'Score_{frame}'] >= threshold).astype(int)
    
    agreement = (ml_predictions == maut_hire).mean()
    ml_hire_maut_reject = ((ml_predictions == 1) & (maut_hire == 0)).sum()
    ml_reject_maut_hire = ((ml_predictions == 0) & (maut_hire == 1)).sum()
```

**Interpretation:**
- High agreement → ML learned MAUT well
- Disagreements → ML found different patterns (proxy discrimination?)

### Feature Importance

Random Forest provides feature importance scores:

```python
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
```

Shows which attributes drive hiring predictions (potential bias indicators).

### Key Functions

| Function | Purpose |
|----------|---------|
| `generate_hire_labels(df, method, rate)` | Create binary labels from MAUT |
| `prepare_ml_features(df)` | Select feature columns |
| `train_random_forest(X, y)` | Train and evaluate model |
| `compute_model_metrics(y_true, y_pred)` | Calculate accuracy, precision, etc. |
| `compute_fairness_metrics(df, predictions)` | Demographic parity analysis |
| `compare_ml_vs_maut(df, predictions)` | Agreement analysis |
| `run_ml_analysis(df)` | Complete ML pipeline |
| `format_ml_report(results)` | Generate text report |

---

## 8. Output Files

### Data Output

| File | Description |
|------|-------------|
| `data/mock_candidates.csv` | 500 synthetic candidates with all attributes |

### Visualization Outputs

| File | Type | Description |
|------|------|-------------|
| `parallel_coordinates.html` | Interactive | All candidates across all attributes |
| `pareto_front_2d.html` | Interactive | Efficiency vs Equity trade-off |
| `pareto_front_3d.html` | Interactive | 3D trade-off surface |
| `frame_comparison.html` | Interactive | Top candidates by frame |
| `frame_demographics.png` | Static | Demographic composition |
| `gsa_tornado.png` | Static | Sensitivity analysis results |
| `correlation_heatmap.png` | Static | Attribute correlations (bias detection) |
| `dashboard.html` | Interactive | Combined summary view |
| `ml_feature_importance.png` | Static | RF feature importance |
| `ml_fairness_rates.png` | Static | Selection rates by group |
| `ml_confusion_matrix.png` | Static | Prediction accuracy |
| `ml_probability_distribution.html` | Interactive | Hire probability by group |
| `ml_maut_agreement.png` | Static | ML vs MAUT comparison |

---

## 9. Key Findings & Interpretations

### Finding 1: Embedded Bias Detection

**Skills Gap: ~5.7 points**

The DFA analysis reveals a statistically significant difference in skills scores between minority and non-minority candidates. This demonstrates how AI-assisted analysis can uncover subtle historical discrimination patterns that might otherwise go unnoticed.

### Finding 2: Frame Impact on Selection

| Frame | Minorities in Top 10 | Interpretation |
|-------|---------------------|----------------|
| Efficiency | 3/10 | Skills-focused selection underrepresents minorities |
| Equity | 10/10 | Diversity-focused selection maximizes representation |
| Balanced | 9/10 | Compromise achieves near-parity |

**Key Insight:** The same candidates, evaluated by the same system, produce dramatically different outcomes based solely on priority weights.

### Finding 3: Sensitivity Dominance

**Equity weight ST = 0.895**

The equity weight has overwhelming influence on minority representation. This tells stakeholders:
- Small changes to equity weight → Large outcome changes
- Ethics and Integrity weights can be fixed (low sensitivity)
- Focus negotiation on Skills vs Equity trade-off

### Finding 4: Pareto-Optimal Trade-offs

The visualizations reveal that:
- Maximum efficiency and maximum equity are NOT simultaneously achievable
- There's a continuous frontier of "fair" solutions
- Stakeholders can choose their preferred position on the frontier

### Finding 5: ML Fairness

**Typical Results:**
- ML passes the 4/5ths rule for both minority and gender
- 84% agreement with MAUT Balanced frame
- Feature importance reveals which attributes drive predictions

**Caution:** Even without using demographic features directly, ML models can learn proxy patterns that correlate with protected attributes.

---

## 10. Technical Dependencies

### Required Packages

```toml
[project]
dependencies = [
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "plotly>=5.15.0",
    "scikit-learn>=1.3.0",
    "SALib>=1.4.7",
]
```

### Package Purposes

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computations |
| `pandas` | Data manipulation |
| `matplotlib` | Static plotting (with Agg backend) |
| `seaborn` | Statistical visualizations |
| `plotly` | Interactive visualizations |
| `scikit-learn` | Random Forest, metrics, cross-validation |
| `SALib` | Sobol sensitivity analysis |

### Running the Project

```powershell
# Install dependencies
uv sync

# Run complete prototype
uv run main.py
```

### Execution Time

- 500 candidates
- 1024 Sobol samples (12,288 evaluations)
- ~30-60 seconds total runtime

---

## Summary

This DFA Hiring Prototype demonstrates a complete AI-assisted decision-making pipeline:

1. **Data with Bias** → Simulates real-world discrimination patterns
2. **MAUT Scoring** → Makes trade-offs explicit through weighted aggregation
3. **Sensitivity Analysis** → Reveals which priorities matter most
4. **Visualizations** → Enables informed A Posteriori decision-making
5. **ML Analysis** → Shows how automation can be audited for fairness

The key value is **transparency**: instead of a black-box "hire/no-hire" decision, stakeholders see ALL Pareto-optimal options and can make informed choices that align with organizational values.

---

*Documentation generated for DFA Hiring Decision Prototype v1.0*
*Last updated: November 2025*
