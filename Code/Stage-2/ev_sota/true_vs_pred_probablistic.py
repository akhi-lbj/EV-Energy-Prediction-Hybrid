import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('probabilistic_predictions_test.csv')

# Create figure
plt.figure(figsize=(12, 6))

# Plot true vs predicted
plt.plot(df['true_kWh'], label='True Energy Delivered', linewidth=1.5)
plt.plot(df['pred_median'], label='Predicted Median Energy', linewidth=1.5)

# Labels and title
plt.xlabel('Index')
plt.ylabel('Energy (kWh)')
plt.title('True vs Predicted Energy Delivered')

# Legend and grid
plt.legend()
plt.grid(True, alpha=0.3)

# Save as PDF
plt.savefig('energy_comparison.pdf', format='pdf', bbox_inches='tight')

# Show plot
plt.show()