# Decision Framing and Analytics (DFA) - AI in Hiring Decisions

A prototype demonstrating how different decision frames (priorities) affect candidate selection outcomes in AI-assisted hiring. This project enables transparent trade-off analysis and bias detection in hiring decisions.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Project Structure](#project-structure)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Output Files](#output-files)
8. [Understanding the Results](#understanding-the-results)
9. [Configuration](#configuration)
10. [Troubleshooting](#troubleshooting)
11. [License](#license)

---

## Overview

Traditional AI hiring systems make opaque decisions. This prototype makes trade-offs **explicit and transparent**, allowing stakeholders to:

- See all Pareto-optimal solutions (no dominated choices)
- Understand how weight changes affect outcomes
- Detect hidden bias in training data
- Make informed decisions through A Posteriori Preference Articulation

The project implements a 5-phase pipeline:

1. **Mock Data Generation** - Synthetic candidates with embedded historical bias
2. **Multi-Criteria Assessment (MAUT)** - Weighted scoring under different decision frames
3. **Global Sensitivity Analysis (GSA)** - Sobol variance-based analysis of weight influence
4. **Visualization Generation** - Interactive Pareto fronts and trade-off charts
5. **Machine Learning Classifier** - Random Forest with fairness analysis

---

## Key Features

- **Decision Framing**: Compare Efficiency, Equity, and Balanced hiring approaches
- **Bias Detection**: Automatically identify skills gaps between demographic groups
- **Sensitivity Analysis**: Determine which priorities most affect outcomes
- **Fairness Metrics**: Disparate impact ratio and 4/5ths rule compliance
- **Interactive Visualizations**: Plotly-based dashboards and Pareto fronts
- **ML Comparison**: Compare explicit MAUT decisions with ML predictions

---

## Project Structure

```
DFA Project/
├── main.py                     # Main orchestrator - runs all 5 phases
├── pyproject.toml              # Project dependencies
├── README.md                   # This file
├── DOCUMENTATION.md            # Detailed technical documentation
│
├── src/                        # Source modules
│   ├── __init__.py
│   ├── data_generator.py       # Phase 1: Synthetic data with bias
│   ├── maut_scorer.py          # Phase 2: MAUT multi-criteria scoring
│   ├── sensitivity_analysis.py # Phase 3: Sobol GSA
│   ├── visualizations.py       # Phase 4: All plots and charts
│   └── ml_classifier.py        # Phase 5: Random Forest + fairness
│
├── data/
│   └── mock_candidates.csv     # Generated candidate dataset
│
└── outputs/
    └── figures/                # All generated visualizations
        ├── parallel_coordinates.html
        ├── pareto_front_2d.html
        ├── pareto_front_3d.html
        ├── frame_comparison.html
        ├── dashboard.html
        ├── ml_probability_distribution.html
        └── ... (additional PNG files)
```

---

## Requirements

- Python 3.12 or higher
- Operating System: Windows, macOS, or Linux

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | >= 2.0 | Data manipulation |
| numpy | >= 1.24 | Numerical computations |
| matplotlib | >= 3.7 | Static plotting |
| seaborn | >= 0.12 | Statistical visualizations |
| plotly | >= 5.15 | Interactive visualizations |
| scikit-learn | >= 1.3 | Random Forest and metrics |
| SALib | >= 1.4 | Sobol sensitivity analysis |
| scipy | >= 1.10 | Scientific computing |

---

## Installation

### Option 1: Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager. If you don't have it installed:

```powershell
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the project:

```powershell
# Clone the repository
git clone https://github.com/lakshika17/ai-hiring-analysis.git
cd ai-hiring-analysis

# Create virtual environment and install dependencies
uv sync
```

### Option 2: Using pip

```powershell
# Clone the repository
git clone https://github.com/lakshika17/ai-hiring-analysis.git
cd ai-hiring-analysis

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.\.venv\Scripts\activate.bat

# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install pandas>=2.0 numpy>=1.24 matplotlib>=3.7 seaborn>=0.12 plotly>=5.15 scikit-learn>=1.3 SALib>=1.4 scipy>=1.10
```

### Option 3: Using pip with requirements

If you prefer a requirements.txt approach:

```powershell
# Create requirements.txt manually or use:
pip install pandas numpy matplotlib seaborn plotly scikit-learn SALib scipy

# Or install from pyproject.toml
pip install .
```

---

## Usage

### Running the Full Pipeline

Execute all 5 phases with a single command:

```powershell
# Using uv
uv run main.py

# Using activated virtual environment
python main.py
```

### Expected Output

The pipeline will:

1. Generate 500 synthetic candidates with embedded bias
2. Apply three decision frames (Efficiency, Equity, Balanced)
3. Run Sobol sensitivity analysis (12,288 model evaluations)
4. Generate 8+ interactive and static visualizations
5. Train and evaluate a Random Forest classifier

Typical runtime: 30-60 seconds

### Sample Console Output

```
======================================================================
 DFA HIRING DECISION PROTOTYPE
======================================================================
AI in Hiring Decisions - Decision Framing and Analytics

======================================================================
 PHASE 1: Mock Data Generation
======================================================================
Generating 500 synthetic candidates with embedded bias...
Dataset saved to: data/mock_candidates.csv

Dataset Summary:
  Total Candidates: 500
  Female: 250 (50.0%)
  Minority: 250 (50.0%)

Embedded Bias Detection:
  Avg Skills (Non-Minority): 68.5
  Avg Skills (Minority): 62.8
  Skills Gap: 5.7 points (bias indicator)

...
```

### Running Individual Modules

You can run each module independently for testing:

```powershell
# Test data generation
python -m src.data_generator

# Test MAUT scoring
python -m src.maut_scorer

# Test sensitivity analysis
python -m src.sensitivity_analysis

# Test ML classifier
python -m src.ml_classifier
```

---

## Output Files

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

## Understanding the Results

### Decision Frames

| Frame | Philosophy | Skills Weight | Equity Weight |
|-------|-----------|---------------|---------------|
| Efficiency | Maximize performance | 45% | 5% |
| Equity | Prioritize diversity | 15% | 40% |
| Balanced | Compromise solution | 25% | 25% |

### Key Metrics

**Demographic Parity Ratio**
- Ratio = 1.0: Perfect parity
- Ratio < 0.8: Potential disparate impact (4/5ths rule violation)
- Ratio > 1.25: Reverse discrimination concern

**Sensitivity Indices**
- S1 (First-Order): Direct effect of a weight
- ST (Total-Order): Total effect including interactions
- Higher ST means the weight has more influence on outcomes

### Interpreting Visualizations

**Pareto Front (2D)**
- X-axis: Efficiency (average skills of selected candidates)
- Y-axis: Equity (minority representation rate)
- Points on the curve: Pareto-optimal solutions (can't improve one without sacrificing the other)

**Tornado Chart**
- Horizontal bars show sensitivity of each weight
- Longer bars indicate more influential factors
- Typically, Equity weight dominates minority representation outcomes

---

## Configuration

### Adjusting Parameters

Edit `main.py` to modify:

```python
# Number of candidates (default: 500)
N_CANDIDATES = 500

# Sobol samples for sensitivity analysis (default: 1024)
Si, samples, outputs = run_sobol_analysis(df, n_samples=1024)

# Weight sweep points for Pareto front (default: 100)
sweep_df = run_weight_sweep(df, n_points=100)
```

### Decision Frame Weights

Edit `src/maut_scorer.py` to modify frame definitions:

```python
DECISION_FRAMES = {
    'Efficiency': {
        'weights': {
            'x_Skills': 0.45,
            'x_Cultural': 0.25,
            'x_Ethics': 0.15,
            'x_Equity': 0.05,
            'x_Integrity': 0.10,
        }
    },
    # ... other frames
}
```

### ML Classifier Settings

Edit `src/ml_classifier.py` to adjust model parameters:

```python
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=4,
)
```

---

## Troubleshooting

### Common Issues

**ModuleNotFoundError: No module named 'src'**

Ensure you're running from the project root directory:

```powershell
cd "path/to/DFA Project"
python main.py
```

**Matplotlib/Tkinter errors on Linux servers**

The project uses the 'Agg' backend by default to avoid display issues. If you encounter errors, ensure this is set in `src/visualizations.py`:

```python
import matplotlib
matplotlib.use('Agg')
```

**SALib import errors**

Ensure SALib is installed correctly:

```powershell
pip install SALib>=1.4.7
```

**Virtual environment not activating (Windows)**

If you get execution policy errors:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Performance Issues

- Reduce `n_samples` in `run_sobol_analysis()` for faster (but less accurate) sensitivity analysis
- Reduce `N_CANDIDATES` for quicker testing
- Reduce `n_points` in `run_weight_sweep()` for faster Pareto front generation

---

## License

This project is for educational and demonstration purposes.

---

## Further Reading

See `DOCUMENTATION.md` for detailed technical documentation including:

- Mathematical foundations of MAUT
- Sobol sensitivity analysis methodology
- Fairness metrics definitions
- Interpretation guidelines for all visualizations
