"""
Phase 1: Mock Dataset Generation with Embedded Bias
Generates synthetic candidate data for the AI Hiring Decision prototype.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_mock_candidates(n_candidates: int = 150, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic candidate dataset with 6 attributes.
    
    Attributes:
        x_Skills: Technical competencies and experience (0-100)
        x_Cultural: Cultural alignment and organizational fit (0-100)
        x_Ethics: Ethics and professionalism score (0-100)
        x_Gender: Gender indicator (0=Male, 1=Female)
        x_Minority: Minority/historically excluded group status (0=No, 1=Yes)
        x_Integrity: Data integrity/credential validation score (0-100)
    
    Embedded Bias: Skills scores are systematically lower for minority candidates
    to simulate historical prejudice patterns in training data. The bias is subtle
    (6-8 point gap) to demonstrate how DFA helps uncover hidden discrimination.
    
    Key Design Choices:
    - Some high-performing minority candidates exist (demonstrating why AI-assisted
      DFA can help identify overlooked talent)
    - Bias is subtle enough that naive hiring might miss it
    - Diversity in all attributes allows meaningful trade-off analysis
    """
    np.random.seed(random_seed)
    
    # Generate base attributes with realistic distributions
    data = {
        'Candidate_ID': [f'C{str(i+1).zfill(3)}' for i in range(n_candidates)],
        'x_Skills': np.random.normal(68, 14, n_candidates).clip(20, 100),
        'x_Cultural': np.random.normal(65, 12, n_candidates).clip(20, 100),
        'x_Ethics': np.random.normal(72, 11, n_candidates).clip(25, 100),
        'x_Gender': np.random.binomial(1, 0.48, n_candidates),  # ~48% female
        'x_Minority': np.random.binomial(1, 0.35, n_candidates),  # ~35% minority
        'x_Integrity': np.random.normal(78, 10, n_candidates).clip(30, 100),
    }
    
    df = pd.DataFrame(data)
    
    # EMBED SUBTLE HISTORICAL BIAS: Reduce skills scores for minority candidates
    # This simulates biased training data where minority candidates were
    # historically underrated due to systemic discrimination
    # Using a SUBTLE bias (10% reduction) to show how DFA reveals hidden patterns
    bias_factor = 0.90  # 10% reduction (moderate but detectable)
    minority_mask = df['x_Minority'] == 1
    noise = np.random.normal(0, 3, minority_mask.sum())
    df.loc[minority_mask, 'x_Skills'] = (
        df.loc[minority_mask, 'x_Skills'] * bias_factor + noise
    ).clip(20, 100)
    
    # Add some HIGH-PERFORMING minority candidates (demonstrates why DFA helps)
    # These are candidates who would be overlooked without proper analysis
    n_high_performers = int(n_candidates * 0.05)  # ~5% exceptional minority candidates
    high_perf_indices = df[minority_mask].sample(n=min(n_high_performers, minority_mask.sum()), 
                                                   random_state=random_seed).index
    df.loc[high_perf_indices, 'x_Skills'] = np.random.uniform(75, 90, len(high_perf_indices))
    df.loc[high_perf_indices, 'x_Ethics'] = np.random.uniform(78, 92, len(high_perf_indices))
    
    # Subtle correlation: minority candidates may have slightly lower cultural fit scores
    # (simulating bias in subjective evaluations) - reduced from 0.92 to 0.96
    df.loc[minority_mask, 'x_Cultural'] = (
        df.loc[minority_mask, 'x_Cultural'] * 0.96 + np.random.normal(0, 2, minority_mask.sum())
    ).clip(20, 100)
    
    # Round numerical columns
    for col in ['x_Skills', 'x_Cultural', 'x_Ethics', 'x_Integrity']:
        df[col] = df[col].round(2)
    
    return df


def apply_data_integrity_mitigation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 1 Data Integrity Mitigation:
    Simulate validation of credentials and adjustment of scores based on
    data quality concerns (inflated credentials, unverified qualifications).
    """
    df = df.copy()
    
    # Flag candidates with suspiciously high scores across all metrics
    high_threshold = 85
    df['Integrity_Flag'] = (
        (df['x_Skills'] > high_threshold) & 
        (df['x_Cultural'] > high_threshold) & 
        (df['x_Ethics'] > high_threshold)
    ).astype(int)
    
    # Apply small penalty to flagged candidates' integrity score
    df.loc[df['Integrity_Flag'] == 1, 'x_Integrity'] = (
        df.loc[df['Integrity_Flag'] == 1, 'x_Integrity'] * 0.95
    ).round(2)
    
    return df


def save_dataset(df: pd.DataFrame, output_path: str = None) -> str:
    """Save the generated dataset to CSV."""
    if output_path is None:
        output_path = Path(__file__).parent.parent / 'data' / 'mock_candidates.csv'
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    return str(output_path)


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """Generate summary statistics for the dataset."""
    return {
        'total_candidates': len(df),
        'female_count': df['x_Gender'].sum(),
        'female_pct': (df['x_Gender'].mean() * 100).round(1),
        'minority_count': df['x_Minority'].sum(),
        'minority_pct': (df['x_Minority'].mean() * 100).round(1),
        'avg_skills': df['x_Skills'].mean().round(2),
        'avg_skills_minority': df[df['x_Minority'] == 1]['x_Skills'].mean().round(2),
        'avg_skills_non_minority': df[df['x_Minority'] == 0]['x_Skills'].mean().round(2),
        'skills_gap': (
            df[df['x_Minority'] == 0]['x_Skills'].mean() - 
            df[df['x_Minority'] == 1]['x_Skills'].mean()
        ).round(2),
    }


if __name__ == "__main__":
    # Generate and save mock dataset
    df = generate_mock_candidates(150)
    df = apply_data_integrity_mitigation(df)
    path = save_dataset(df)
    
    print(f"Dataset saved to: {path}")
    print("\nDataset Summary:")
    summary = get_dataset_summary(df)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\nSample rows:")
    print(df.head(10).to_string())
