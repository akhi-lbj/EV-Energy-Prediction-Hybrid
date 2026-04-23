import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# --- Configuration & Styling ---
# Use a professional style for academic publication
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Inter', 'Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

# Colors - Academic and distinguishable
COLOR_TRUE = '#2C3E50'      # Dark blue-grey
COLOR_PRED = '#E74C3C'      # Soft red
COLOR_SHADE = '#E74C3C'     # Same as pred for consistency in shading

def plot_energy_prediction_results(csv_path='probabilistic_predictions_test.csv', 
                                  original_data_path='acn_enhanced_final_2019_data.csv',
                                  subset_range=(0, 250)):
    """
    Produces a high-quality visualization for EV energy prediction.
    """
    # 1. Load prediction results
    try:
        df_pred = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return

    # 2. Try to recover timestamps from original data (using deterministic split logic)
    df_test_full = None
    try:
        df_orig = pd.read_csv(original_data_path)
        df_orig['connectionTime'] = pd.to_datetime(df_orig['connectionTime'], errors='coerce', utc=True)
        df_orig['month'] = df_orig['connectionTime'].dt.month
        
        # Features used for dropping NaNs (must match generation script exactly)
        feature_cols = ['parsed_milesRequested', 'parsed_WhPerMile', 'parsed_minutesAvailable', 'parsed_kWhRequested']
        df_orig = df_orig.dropna(subset=['kWhDelivered'] + feature_cols).copy()

        # Re-apply splitting logic (fixed seed 42)
        test_indices = []
        for month in range(1, 13):
            month_df = df_orig[df_orig['month'] == month].copy()
            n = len(month_df)
            test_sample = month_df.sample(n=int(0.4 * n), random_state=42)
            test_indices.extend(test_sample.index)
        
        test_mask = df_orig.index.isin(test_indices)
        df_test_full = df_orig[test_mask].copy()
        
        # Ensure rows match
        if len(df_test_full) == len(df_pred):
            df_pred['timestamp'] = df_test_full['connectionTime'].values
            df_pred = df_pred.sort_values('timestamp')
        else:
            print(f"Warning: Row count mismatch ({len(df_test_full)} vs {len(df_pred)}). Using index instead.")
            df_test_full = None
    except Exception as e:
        print(f"Note: Could not recover timestamps ({e}). Falling back to index.")

    # 3. Prepare Subset
    start, end = subset_range
    subset = df_pred.iloc[start:end].copy()
    
    # 4. Create Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # X-axis definition
    x = subset['timestamp'] if 'timestamp' in subset.columns else np.arange(len(subset))
    
    # Plot Confidence Intervals (Conformal Bounds if available)
    if 'lower_conformal' in subset.columns and 'upper_conformal' in subset.columns:
        ax.fill_between(x, subset['lower_conformal'], subset['upper_conformal'], 
                        color=COLOR_SHADE, alpha=0.15, label='90% Prediction Interval (Conformal)')
    elif 'pred_05' in subset.columns and 'pred_95' in subset.columns:
        ax.fill_between(x, subset['pred_05'], subset['pred_95'], 
                        color=COLOR_SHADE, alpha=0.15, label='90% Prediction Interval (Quantile)')

    # Plot True Values
    ax.plot(x, subset['true_kWh'], label='Observed Energy Delivery', 
            color=COLOR_TRUE, linewidth=2.2, alpha=0.9, marker='o', markersize=3, markevery=5)

    # Plot Predicted Median
    ax.plot(x, subset['pred_median'], label='Predicted Median (Ensemble)', 
            color=COLOR_PRED, linewidth=1.8, linestyle='--', alpha=0.9)

    # 5. Formatting
    ax.set_xlabel('Arrival Time' if 'timestamp' in subset.columns else 'Charging Session Sequence')
    ax.set_ylabel('Energy Delivered (kWh)')
    ax.set_title('Comparative Analysis: Observed vs. Predicted EV Charging Energy Delivery', pad=20)
    
    # Subtitle or supplementary info
    ax.text(0.5, 1.02, f'Visualizing sessions {start} to {end} of test set', 
            transform=ax.transAxes, ha='center', fontsize=10, style='italic', alpha=0.7)

    # Handle time axis formatting
    if 'timestamp' in subset.columns:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
        plt.xticks(rotation=0)
    
    # Grid & Spines
    ax.grid(True, linestyle='--', alpha=0.4, which='major')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)

    # Tight Layout
    plt.tight_layout()

    # 6. Save Outputs
    pdf_filename = 'ev_energy_prediction_publication.pdf'
    png_filename = 'ev_energy_prediction_publication.png'
    plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
    plt.savefig(png_filename, format='png', bbox_inches='tight', dpi=300)
    
    print(f"Visualization complete.")
    print(f"Saved high-quality PDF: {pdf_filename}")
    print(f"Saved preview PNG: {png_filename}")
    # plt.show() # Commented out for non-interactive execution

if __name__ == "__main__":
    plot_energy_prediction_results()
