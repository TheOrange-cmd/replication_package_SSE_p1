import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
warnings.filterwarnings('ignore')

DATA_PATH = Path(__file__).parent / "output.csv"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

POWER_COL = "Cpu_AMD_Ryzen_7_5700U_with_Radeon_Graphics_Package_Power"
CPU_LOAD_COL = "Cpu_AMD_Ryzen_7_5700U_with_Radeon_Graphics_CPU_Total_Load"
MEMORY_LOAD_COL = "Memory_Total_Memory_Memory_Load"
GPU_LOAD_COL = "GpuAmd_AMD_Radeon_TM__Graphics_GPU_Core_Load"
TIMESTAMP_COL = "timestamp"
Z_SCORE_THRESHOLD = 3  # Lecture 04: remove outliers > 3 std from mean

sns.set_theme(style="whitegrid", font_scale=1.1)

def load_data():
    df = pd.read_csv(DATA_PATH, sep=None, engine='python')
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    parts = df['experiment_name'].str.extract(
        r'^(.+?)_(chrome|firefox|edge)_(with_adblock|no_adblock)$'
    )
    df['website'] = parts[0]
    df['browser'] = parts[1].str.capitalize()

    expected_adblock = parts[2] == 'with_adblock'
    mismatch = df['adblock_enabled'] != expected_adblock
    if mismatch.any():
        print(f"  [WARNING] {mismatch.sum()} rows have adblock_enabled inconsistent with experiment_name")

    return df


def compute_energy_per_run(df):
    results = []

    for (exp, run_id), group in df.groupby(['experiment_name', 'run_id']):
        group = group.sort_values(TIMESTAMP_COL)
        power = group[POWER_COL].values
        timestamps = group[TIMESTAMP_COL]

        # Convert timestamps to seconds from start
        t_seconds = (timestamps - timestamps.iloc[0]).dt.total_seconds().values

        if len(power) < 2:
            continue

        # Trapezoid rule: energy in Joules
        energy_j = np.trapz(power, t_seconds)
        duration_s = t_seconds[-1] - t_seconds[0]
        avg_power_w = energy_j / duration_s if duration_s > 0 else 0

        # Energy Delay Product (Lecture 05): EDP = E * t
        edp = energy_j * duration_s
        
        avg_cpu_load = group[CPU_LOAD_COL].mean()
        avg_memory_load = group[MEMORY_LOAD_COL].mean()
        avg_gpu_load = group[GPU_LOAD_COL].mean()

        total_bytes = group['total_bytes'].max()
        total_requests = group['total_requests'].max()
        ad_bytes = group['ad_bytes'].max()
        ad_requests = group['ad_requests'].max()

        results.append({
            'experiment_name': exp,
            'run_id': run_id,
            'website': group['website'].iloc[0],
            'browser': group['browser'].iloc[0],
            'adblock_enabled': group['adblock_enabled'].iloc[0],
            'energy_j': energy_j,
            'avg_power_w': avg_power_w,
            'duration_s': duration_s,
            'edp': edp,
            'total_bytes': total_bytes,
            'ad_bytes': ad_bytes,
            'total_requests': total_requests,
            'ad_requests': ad_requests,
            'n_samples': len(power),
            'avg_cpu_load': avg_cpu_load,
            'avg_memory_load': avg_memory_load,
            'avg_gpu_load': avg_gpu_load
        })

    return pd.DataFrame(results)


def remove_outliers_zscore(energy_df, metric='energy_j'):
    clean_dfs = []
    removed_count = 0

    for exp, group in energy_df.groupby('experiment_name'):
        z_scores = np.abs(stats.zscore(group[metric], ddof=1))
        mask = z_scores < Z_SCORE_THRESHOLD
        removed_count += (~mask).sum()
        clean_dfs.append(group[mask])

    clean_df = pd.concat(clean_dfs, ignore_index=True)
    print(f"  Outliers removed (z-score > {Z_SCORE_THRESHOLD}): {removed_count}")
    print(f"  Remaining data points: {len(clean_df)}")
    return clean_df


def test_normality(energy_df, metric='energy_j'):
    results = []
    for exp, group in energy_df.groupby('experiment_name'):
        if len(group) >= 3:
            stat, p_value = stats.shapiro(group[metric])
            results.append({
                'experiment_name': exp,
                'shapiro_stat': stat,
                'shapiro_p': p_value,
                'is_normal': p_value >= 0.05,
                'n': len(group),
            })
    return pd.DataFrame(results)


def cohens_d(group1, group2):
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt((var1 + var2) / 2)
    if pooled_std == 0:
        return 0.0
    return (group1.mean() - group2.mean()) / pooled_std


def interpret_cohens_d(d):
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def hypothesis_tests(energy_df, normality_df, metric='energy_j'):
    results = []

    for (website, browser), group in energy_df.groupby(['website', 'browser']):
        adblock = group[group['adblock_enabled'] == True][metric]
        no_adblock = group[group['adblock_enabled'] == False][metric]

        if len(adblock) < 3 or len(no_adblock) < 3:
            continue

        # Check normality of both groups
        exp_adblock = f"{website}_{browser.lower()}_with_adblock"
        exp_no_adblock = f"{website}_{browser.lower()}_no_adblock"

        norm_adblock = normality_df[normality_df['experiment_name'] == exp_adblock]
        norm_no = normality_df[normality_df['experiment_name'] == exp_no_adblock]

        both_normal = (
            (len(norm_adblock) > 0 and norm_adblock['is_normal'].iloc[0]) and
            (len(norm_no) > 0 and norm_no['is_normal'].iloc[0])
        )

        # Choose test based on normality
        if both_normal:
            test_name = "Welch's t-test"
            stat, p_value = stats.ttest_ind(no_adblock, adblock, equal_var=False)
            d = cohens_d(no_adblock, adblock)
            cles = None
            effect_magnitude = interpret_cohens_d(d)
            mean_no = no_adblock.mean()
            mean_ad = adblock.mean()
        else:
            test_name = "Mann-Whitney U"
            stat, p_value = stats.mannwhitneyu(no_adblock, adblock, alternative='two-sided')
            d = None
            cles = stat / (len(no_adblock) * len(adblock))
            effect_magnitude = "N/A"
            mean_no = no_adblock.median()
            mean_ad = adblock.median()

        diff_pct = ((mean_no - mean_ad) / mean_no) * 100

        results.append({
            'website': website,
            'browser': browser,
            'test': test_name,
            'statistic': stat,
            'p_value': p_value,
            'cohens_d': d,
            'cles': cles,
            'effect_magnitude': effect_magnitude,
            'mean_no_adblock': mean_no,
            'mean_adblock': mean_ad,
            'diff_pct': diff_pct,
            'both_normal': both_normal,
        })

    res_df = pd.DataFrame(results)

    # Benjamini-Hochberg FDR correction using statsmodels
    from statsmodels.stats.multitest import multipletests
    
    reject, p_corrected, _, _ = multipletests(res_df['p_value'], method='fdr_bh')
    res_df['p_value_corrected'] = p_corrected
    res_df['is_significant_fdr'] = reject

    return res_df

def plot_violin_by_browser(energy_df, metric='energy_j', label='Energy (J)'):
    browsers = sorted(energy_df['browser'].unique())
    fig, axes = plt.subplots(1, len(browsers), figsize=(6 * len(browsers), 7), sharey=False)
    if len(browsers) == 1:
        axes = [axes]

    for ax, browser in zip(axes, browsers):
        data = energy_df[energy_df['browser'] == browser]
        data = data.copy()
        data['Adblock'] = data['adblock_enabled'].map({True: 'With Adblock', False: 'No Adblock'})

        sns.violinplot(data=data, x='Adblock', y=metric, ax=ax, inner='box',
                       palette=['#ff6b6b', '#51cf66'], cut=0)
        ax.set_title(f'{browser}', fontsize=14, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(label)

    fig.suptitle(f'{label}: Adblock vs No Adblock by Browser', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'violin_{metric}_by_browser.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_violin_by_website(energy_df, metric='energy_j', label='Energy (J)'):
    websites = sorted(energy_df['website'].unique())
    n_cols = 4
    n_rows = (len(websites) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    for i, website in enumerate(websites):
        data = energy_df[energy_df['website'] == website].copy()
        data['Adblock'] = data['adblock_enabled'].map({True: 'Adblock', False: 'No Adblock'})

        sns.violinplot(data=data, x='Adblock', y=metric, ax=axes[i], inner='box',
                       palette=['#ff6b6b', '#51cf66'], cut=0)
        axes[i].set_title(website, fontsize=11, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel(label if i % n_cols == 0 else '')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'{label}: Adblock vs No Adblock by Website', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'violin_{metric}_by_website.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_heatmap_significance(test_results):
    pivot = test_results.pivot_table(index='website', columns='browser', values='p_value_corrected')

    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn_r', vmin=0, vmax=0.1,
                ax=ax, linewidths=0.5)
    ax.set_title('P-values: Adblock vs No Adblock\n(green = not significant, red = significant)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'heatmap_pvalues.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_effect_size_heatmap(test_results):
    pivot = test_results.pivot_table(index='website', columns='browser', values='cohens_d')

    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, linewidths=0.5)
    ax.set_title("Cohen's d Effect Size: No Adblock vs Adblock\n(positive = no adblock uses MORE energy)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'heatmap_cohens_d.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_energy_diff_pct(test_results):
    fig, ax = plt.subplots(figsize=(14, 7))

    pivot = test_results.pivot_table(index='website', columns='browser', values='diff_pct')
    pivot.plot(kind='bar', ax=ax, width=0.8, edgecolor='black', linewidth=0.5)

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_ylabel('Energy Difference (%)\n(positive = adblock saves energy)')
    ax.set_title('Energy Savings from Adblock by Website and Browser', fontsize=14, fontweight='bold')
    ax.set_xlabel('')
    plt.xticks(rotation=45, ha='right')
    ax.legend(title='Browser')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'energy_diff_pct.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_network_savings(energy_df):
    agg = energy_df.groupby(['website', 'adblock_enabled']).agg(
        total_bytes=('total_bytes', 'mean'),
        ad_bytes=('ad_bytes', 'mean'),
        total_requests=('total_requests', 'mean'),
        ad_requests=('ad_requests', 'mean'),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, metric, title in zip(
        axes,
        ['total_bytes', 'total_requests'],
        ['Total Bytes Transferred', 'Total Requests']
    ):
        pivot = agg.pivot_table(index='website', columns='adblock_enabled', values=metric)
        pivot.columns = ['No Adblock', 'With Adblock']
        pivot.plot(kind='bar', ax=ax, width=0.8, color=['#ff6b6b', '#51cf66'],
                   edgecolor='black', linewidth=0.5)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('')
        ax.legend()
        plt.sca(ax)
        plt.xticks(rotation=45, ha='right')

    plt.suptitle('Network Traffic: Adblock vs No Adblock', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'network_savings.png', dpi=150, bbox_inches='tight')
    plt.close()

def analyze_rq2(energy_df):
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from sklearn.preprocessing import StandardScaler

    print("\n[Step 4.5] RQ2 Regression Analysis...")

    # --- Helper -----------------------------------------------------------
    def fit_and_report(df, label):
        """Fit a standardized OLS with website fixed effects, print summary,
        return the fitted model and the scaled feature matrix."""

        feature_cols = ['adblock_enabled', 'total_bytes_mb', 'avg_cpu_load',
                        'avg_memory_load', 'avg_gpu_load']

        # Website dummies (fixed effects) — absorb between-site variance
        website_dummies = pd.get_dummies(df['website'], drop_first=True, dtype=float)

        # Standardize only the continuous/binary predictors of interest,
        # leave website dummies unscaled (they are already 0/1 indicators)
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(df[feature_cols]),
            columns=feature_cols,
            index=df.index
        )
        X = pd.concat([X_scaled, website_dummies], axis=1)
        X = sm.add_constant(X)
        y = df['energy_j']

        model = sm.OLS(y, X).fit()

        print(f"\n{'='*60}")
        print(f"  MODEL: {label}  (n={len(df)})")
        print(f"{'='*60}")
        # Print only the key predictors, not all 13 website dummies
        params_of_interest = model.params[feature_cols]
        pvals_of_interest  = model.pvalues[feature_cols]
        ci_of_interest     = model.conf_int().loc[feature_cols]

        summary_df = pd.DataFrame({
            'std_coef': params_of_interest,
            'p_value':  pvals_of_interest,
            'CI_low':   ci_of_interest[0],
            'CI_high':  ci_of_interest[1],
        })
        summary_df['significant'] = summary_df['p_value'] < 0.05
        print(summary_df.round(4).to_string())
        print(f"\n  R² = {model.rsquared:.3f}  |  Adj. R² = {model.rsquared_adj:.3f}"
              f"  |  F p-value = {model.f_pvalue:.2e}")

        # VIF on the features of interest only (not website dummies)
        vif = pd.DataFrame({
            'feature': feature_cols,
            'VIF': [variance_inflation_factor(X_scaled.values, i)
                    for i in range(len(feature_cols))]
        })
        print("\n  VIF (features of interest):")
        print(vif.to_string(index=False))

        return model, X_scaled, feature_cols, scaler

    # --- Prepare data -----------------------------------------------------
    df = energy_df.copy()
    df['total_bytes_mb'] = df['total_bytes'] / (1024 * 1024)
    df['adblock_enabled'] = df['adblock_enabled'].astype(float)

    # --- Overall model ----------------------------------------------------
    overall_model, _, feature_cols, _ = fit_and_report(df, "ALL BROWSERS")

    # --- Per-browser models -----------------------------------------------
    browser_models = {}
    for browser in sorted(df['browser'].unique()):
        browser_df = df[df['browser'] == browser].copy()
        model, X_scaled, _, _ = fit_and_report(browser_df, f"BROWSER: {browser.upper()}")
        browser_models[browser] = model

    # --- Comparison table: standardized adblock coefficient per browser ---
    print(f"\n{'='*60}")
    print("  SUMMARY: Standardized coefficients of interest by browser")
    print(f"{'='*60}")
    rows = []
    for browser, model in browser_models.items():
        row = {'browser': browser}
        for feat in feature_cols:
            row[f'{feat}_coef'] = round(model.params[feat], 3)
            row[f'{feat}_p']    = round(model.pvalues[feat], 4)
        rows.append(row)
    comp_df = pd.DataFrame(rows).set_index('browser')
    # Print in a readable two-row-per-feature format
    for feat in feature_cols:
        print(f"\n  {feat}:")
        print(comp_df[[f'{feat}_coef', f'{feat}_p']].to_string())

    # --- Residual diagnostics for overall model ---------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(overall_model.fittedvalues, overall_model.resid, alpha=0.4, s=10)
    axes[0].axhline(0, color='red', linewidth=0.8)
    axes[0].set_xlabel('Fitted values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Fitted (all browsers)')
    stats.probplot(overall_model.resid, plot=axes[1])
    axes[1].set_title('Q-Q Plot of Residuals')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'rq2_residuals.png', dpi=150, bbox_inches='tight')
    plt.close()

    # --- Coefficient plot: per-browser, features of interest only ---------
    fig, axes = plt.subplots(1, len(browser_models), figsize=(5 * len(browser_models), 5),
                             sharey=True)
    feature_labels = ['Adblock\nenabled', 'Δ Bytes\n(MB)', 'Δ CPU\nLoad (%)',
                      'Δ Memory\nLoad (%)', 'Δ GPU\nLoad (%)']

    for ax, (browser, model) in zip(axes, browser_models.items()):
        coef = model.params[feature_cols]
        ci   = model.conf_int().loc[feature_cols]
        pval = model.pvalues[feature_cols]
        colors = ['#e74c3c' if p < 0.05 else '#95a5a6' for p in pval]
        y_pos = range(len(coef))
        ax.barh(y_pos, coef.values,
                xerr=[coef.values - ci[0].values, ci[1].values - coef.values],
                align='center', color=colors, capsize=4)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_labels)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Std. coefficient (J per σ)')
        ax.set_title(f'{browser}\n(R²={model.rsquared:.2f}, n={int(model.nobs)})',
                     fontsize=11)
        ax.text(0.02, 0.02, 'red = p < 0.05', transform=ax.transAxes,
                fontsize=8, color='#e74c3c')

    fig.suptitle('RQ2: Standardized Regression Coefficients by Browser\n'
                 '(website fixed effects included)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'rq2_coefficients.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n  -> rq2_residuals.png")
    print("  -> rq2_coefficients.png")

def main():
    print("=" * 70)
    print("  ENERGY CONSUMPTION ANALYSIS: ADBLOCK VS NO ADBLOCK")
    print("  Based on CS4575 Sustainable Software Engineering methodology")
    print("=" * 70)

    # Load data
    print("\n[Step 0] Loading data...")
    df = load_data()
    print(f"  Loaded {len(df)} rows, {df['experiment_name'].nunique()} experiments")
    print(f"  Browsers: {sorted(df['browser'].dropna().unique())}")
    print(f"  Websites: {sorted(df['website'].dropna().unique())}")

    #Compute energy per run via trapezoid rule
    print("\n[Step 1] Computing energy per run (trapezoid rule)...")
    energy_df = compute_energy_per_run(df)
    print(f"  Computed energy for {len(energy_df)} runs")
    print(f"  Energy range: {energy_df['energy_j'].min():.2f} - {energy_df['energy_j'].max():.2f} J")
    print(f"  Avg power range: {energy_df['avg_power_w'].min():.2f} - {energy_df['avg_power_w'].max():.2f} W")

    #Outlier removal (z-score)
    print(f"\n[Step 2] Removing outliers (z-score > {Z_SCORE_THRESHOLD})...")
    clean_df = remove_outliers_zscore(energy_df)

    #Normality testing (Shapiro-Wilk)
    print("\n[Step 3] Normality testing (Shapiro-Wilk)...")
    normality_df = test_normality(clean_df)
    n_normal = normality_df['is_normal'].sum()
    n_total = len(normality_df)
    print(f"  Normal distributions: {n_normal}/{n_total} ({100*n_normal/n_total:.1f}%)")
    print(f"  Non-normal distributions: {n_total - n_normal}/{n_total} ({100*(n_total-n_normal)/n_total:.1f}%)")

    #Hypothesis testing
    print("\n[Step 4] Hypothesis testing (adblock vs no-adblock)...")
    test_results = hypothesis_tests(clean_df, normality_df)
    n_sig = test_results['is_significant_fdr'].sum()
    print(f"  Significant differences (FDR corrected p < 0.05): {n_sig}/{len(test_results)}")

    print("\n  Results summary (FDR corrected p-values):")
    for _, row in test_results.sort_values('p_value_corrected').iterrows():
        sig_marker = "***" if row['is_significant_fdr'] and row['p_value_corrected'] < 0.001 else "**" if row['is_significant_fdr'] and row['p_value_corrected'] < 0.01 else "*" if row['is_significant_fdr'] else "   "
        direction = "adblock saves" if row['diff_pct'] > 0 else "adblock costs more"
        if row['both_normal']:
            effect_str = f"d={row['cohens_d']:+.2f} ({row['effect_magnitude']:10s})"
        else:
            effect_str = f"CLES={row['cles']:.2f}             "
        print(f"    {sig_marker} {row['website']:15s} | {row['browser']:8s} | "
              f"p={row['p_value_corrected']:.4f} | {effect_str} | "
              f"{row['diff_pct']:+.1f}% ({direction})")

    analyze_rq2(clean_df)

    #Aggregate analysis by browser
    print("\n[Step 5] Aggregate analysis by browser...")
    for browser in sorted(clean_df['browser'].unique()):
        b_data = clean_df[clean_df['browser'] == browser]
        adblock = b_data[b_data['adblock_enabled'] == True]['energy_j']
        no_adblock = b_data[b_data['adblock_enabled'] == False]['energy_j']
        stat, p = stats.mannwhitneyu(no_adblock, adblock, alternative='two-sided')
        
        # Bonferroni correction for 3 browser tests
        p_corrected = min(p * 3, 1.0)
        
        cles = stat / (len(no_adblock) * len(adblock))
        diff = ((no_adblock.median() - adblock.median()) / no_adblock.median()) * 100
        print(f"  {browser:8s}: no_adblock={no_adblock.median():.2f}J, adblock={adblock.median():.2f}J, "
              f"diff={diff:+.1f}%, p_corrected={p_corrected:.4f}, CLES={cles:.2f}")

    #Generate visualizations
    print("\n[Step 6] Generating visualizations...")

    # energy distribution by browser
    plot_violin_by_browser(clean_df, 'energy_j', 'Energy (J)')
    print("  -> violin_energy_j_by_browser.png")

    # average power draw by browser
    plot_violin_by_browser(clean_df, 'avg_power_w', 'Average Power (W)')
    print("  -> violin_avg_power_w_by_browser.png")

    # energy distribution per website
    plot_violin_by_website(clean_df, 'energy_j', 'Energy (J)')
    print("  -> violin_energy_j_by_website.png")

    # p-value heatmap (website x browser)
    plot_heatmap_significance(test_results)
    print("  -> heatmap_pvalues.png")

    # effect size heatmap (Cohen's d)
    plot_effect_size_heatmap(test_results)
    print("  -> heatmap_cohens_d.png")

    # % energy diff per website and browser
    plot_energy_diff_pct(test_results)
    print("  -> energy_diff_pct.png")

    # network traffic with/without adblock
    plot_network_savings(clean_df)
    print("  -> network_savings.png")

    # Save results to CSV
    test_results.to_csv(OUTPUT_DIR / 'hypothesis_test_results.csv', index=False)
    normality_df.to_csv(OUTPUT_DIR / 'normality_test_results.csv', index=False)
    clean_df.to_csv(OUTPUT_DIR / 'energy_per_run.csv', index=False)
    print("\n  Results saved to CSV files in results/")

    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print(f"  All outputs saved to: {OUTPUT_DIR.resolve()}")
    print("=" * 70)


if __name__ == '__main__':
    main()
