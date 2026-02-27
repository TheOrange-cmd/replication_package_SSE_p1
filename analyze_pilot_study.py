# General imports
import pandas as pd
import numpy as np
import re
import os
import sys

# Imports for statistical analysis and plotting
from statsmodels.stats.power import TTestIndPower
from scipy.stats import shapiro
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Attempt to import config, handle if it fails
from config import (
    OUTPUT_FILE,
    POWER_COLUMN
)

# OUTPUT_FILE = "pilot_run.csv"
OUTPUT_FILE = "output.csv"

# --- Analysis Parameters ---

# Significance level (alpha): The probability of a Type I error (false positive). Standard is 0.05.
ALPHA = 0.05

# Desired statistical power (1 - beta): The probability of detecting a true effect (avoiding a false negative). Standard is 0.80.
POWER = 0.80

# Your practical limit for sample size per group in the main experiment.
MAX_FEASIBLE_N = 20

# Directory to save the generated plots
PLOT_OUTPUT_DIR = "pilot_study_plots"

# --- Helper Functions ---

def extract_base_name(experiment_name):
    """
    Removes '_with_adblock' or '_no_adblock' from the experiment name
    to create a common identifier for a configuration (e.g., 'youtube_chrome').
    """
    return re.sub(r'_(with|no)_adblock$', '', experiment_name)

def get_recommendation(cohen_d, required_n, max_feasible_n):
    """
    Categorizes the experiment based on effect size and required sample size.
    - Group A (Exclude): Effect is negligible.
    - Group B (Run): Effect is measurable within feasible sample size.
    - Group C (Run with Caution): Effect is present but requires more samples than is feasible.
    """
    # Using standard interpretation of Cohen's d for "small" effect
    if cohen_d < 0.2:
        return "A: Negligible Effect (Exclude)"
    if required_n <= max_feasible_n:
        return "B: Measurable Effect (Run)"
    else:
        return "C: Borderline/Large N (Run with Caution)"

# --- Main Analysis Function ---

def analyze_pilot_data(file_path, power_column):
    """
    Loads pilot study data, visualizes distributions, checks for normality,
    calculates effect sizes, and performs a power analysis to determine the
    required sample size and provide a recommendation for the full study.
    """
    # --- 1. Load and Validate Data ---
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        print("Please ensure you have run the pilot study and the file exists.")
        return

    if power_column not in df.columns:
        print(f"Error: The specified power column '{power_column}' does not exist in the CSV.")
        print(f"Available columns are: {df.columns.tolist()}")
        return

    print(f"Analyzing pilot data from '{file_path}' using column: '{power_column}'\n")

    # Create directory for plots if it doesn't exist
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    print(f"Plots will be saved to the '{PLOT_OUTPUT_DIR}/' directory.\n")

    # --- 2. Aggregate Data ---
    # Calculate the mean of the power column for each unique experimental run.
    aggregated_data = df.groupby(['experiment_name', 'run_id'])[power_column].mean().reset_index()
    
    # Create a common identifier for each configuration (e.g., 'youtube_chrome').
    aggregated_data['config_name'] = aggregated_data['experiment_name'].apply(extract_base_name)

    # --- 3. Analyze Each Configuration ---
    unique_configs = sorted(aggregated_data['config_name'].unique())
    print(f"Found {len(unique_configs)} unique experiment configurations to analyze.")
    print("-" * 70)

    results = []

    for config in unique_configs:
        print(f"Analyzing Configuration: {config}")

        # Separate the data for the current configuration into adblock/no-adblock
        config_df = aggregated_data[aggregated_data['config_name'] == config]
        
        group_no_adblock = config_df[config_df['experiment_name'].str.contains('no_adblock')][power_column]
        group_with_adblock = config_df[config_df['experiment_name'].str.contains('with_adblock')][power_column]

        # Ensure we have data for both groups to compare
        if group_no_adblock.empty or group_with_adblock.empty:
            print("  -> SKIPPING: Missing data for one or both groups ('with_adblock' or 'no_adblock').\n")
            continue

        n1, n2 = len(group_no_adblock), len(group_with_adblock)
        mean1, mean2 = group_no_adblock.mean(), group_with_adblock.mean()
        std1, std2 = group_no_adblock.std(ddof=1), group_with_adblock.std(ddof=1)

        # --- 4. Visualization (Violin Plots) ---
        plt.figure(figsize=(12, 7))
        sns.violinplot(data=config_df, x='experiment_name', y=power_column, inner='quartile', palette='muted')
        sns.stripplot(data=config_df, x='experiment_name', y=power_column, color='k', alpha=0.7, jitter=0.05)
        plt.title(f'Energy Distribution for: {config}\n(Pilot N={n1} vs N={n2})', fontsize=16)
        plt.ylabel(f'{power_column} (Watts)', fontsize=12)
        plt.xlabel('Experiment Condition', fontsize=12)
        plt.xticks(rotation=10, ha='right')
        plt.tight_layout()
        plot_filename = os.path.join(PLOT_OUTPUT_DIR, f"{config}_distribution.png")
        plt.savefig(plot_filename)
        plt.close() # Close the plot to free memory

        # --- 5. Normality Test (Shapiro-Wilk) ---
        # The test requires at least 3 data points.
        print("  - Normality Check (Shapiro-Wilk Test):")
        is_normal = True
        if n1 > 2:
            stat, p_val = shapiro(group_no_adblock)
            print(f"    - No Adblock group:   p-value = {p_val:.3f}")
            if p_val < ALPHA:
                print("      [WARNING] No Adblock data may not be normally distributed.")
                is_normal = False
        if n2 > 2:
            stat, p_val = shapiro(group_with_adblock)
            print(f"    - With Adblock group: p-value = {p_val:.3f}")
            if p_val < ALPHA:
                print("      [WARNING] With Adblock data may not be normally distributed.")
                is_normal = False
        
        if not is_normal:
            print("    [INFO] Non-normal data can affect t-test validity. Consider investigating outliers or using non-parametric tests if this persists in the full study.")

        # --- 6. Effect Size & Power Analysis ---
        print("  - Pilot Statistics:")
        print(f"    - No Adblock:   N={n1}, Mean={mean1:.2f}, StdDev={std1:.2f}")
        print(f"    - With Adblock: N={n2}, Mean={mean2:.2f}, StdDev={std2:.2f}")

        # Check for sufficient data to calculate pooled standard deviation
        if n1 < 2 or n2 < 2 or np.isnan(std1) or np.isnan(std2):
            print("  -> SKIPPING Power Analysis: Not enough data (N<2) to calculate standard deviation.\n")
            continue

        # Calculate Pooled Standard Deviation and Cohen's d (effect size)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            print("  -> SKIPPING Power Analysis: Pooled standard deviation is zero (no variance in data).\n")
            continue
            
        cohen_d = (mean1 - mean2) / pooled_std
        print(f"  - Calculated Effect Size (Cohen's d): {cohen_d:.3f}")

        # Perform Power Analysis to find the required sample size
        power_analysis = TTestIndPower()
        required_n = power_analysis.solve_power(
            effect_size=cohen_d,
            alpha=ALPHA,
            power=POWER,
            ratio=1.0,  # We want equal group sizes (n2/n1 = 1)
            alternative='two-sided'
        )
        
        required_n_per_group = int(np.ceil(required_n))
        
        print(f"  -> RESULT: To achieve {POWER*100}% power, you need ~{required_n_per_group} samples PER GROUP.")
        
        recommendation = get_recommendation(cohen_d, required_n_per_group, MAX_FEASIBLE_N)
        print(f"  -> RECOMMENDATION: {recommendation}")
        
        results.append({
            'Configuration': config,
            'Mean (No Adblock)': mean1,
            'Mean (With Adblock)': mean2,
            'Effect Size (d)': cohen_d,
            'Pilot N': f"{n1} vs {n2}",
            'Required N per Group': required_n_per_group,
            'Recommendation': recommendation
        })
        print("-" * 70)


    # --- 7. Final Summary ---
    if not results:
        print("\nNo results were generated. Please check your data and script configuration.")
        return

    print("\n" + "="*80)
    print("                 PILOT STUDY POWER ANALYSIS SUMMARY")
    print("="*80)
    summary_df = pd.DataFrame(results)
    
    # Reorder columns for clarity
    summary_df = summary_df[[
        'Configuration', 
        'Recommendation', 
        'Required N per Group',
        'Effect Size (d)', 
        'Mean (No Adblock)', 
        'Mean (With Adblock)', 
        'Pilot N'
    ]]
    
    # Format floats for better readability
    pd.options.display.float_format = '{:,.3f}'.format
    
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    analyze_pilot_data(OUTPUT_FILE, POWER_COLUMN)