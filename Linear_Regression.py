
import numpy as np
from astropy.table import Table
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2


file_path = 'global_CCl4_MM.dat'

data_table = Table.read(file_path, format='ascii')

selected_columns = ['date', 'global mean concentration', 'global mean concentration sd']
selected_data_table = data_table[selected_columns].values


data_df = selected_columns.to_pandas()


plt.errorbar(data_df['date'], data_df['global mean concentration'], yerr=data_df['global mean concentration sd'], fmt='o', ecolor='r', capsize=5)
plt.xlabel('Date')
plt.ylabel('Global Mean Concentration')
plt.title('Global Mean Concentration with Error Bars')
plt.grid(True)
plt.show()

data_df['date_ordinal'] = data_df['date'].apply(lambda x: x.toordinal())

X = data_df['date_ordinal'].values
y = data_df['global mean concentration'].values

coefficients, residuals, _, _, _ = np.polyfit(X, y, deg=1, full=True)
slope, intercept = coefficients

y_pred = np.polyval(coefficients, X)

sigma_slope = np.sqrt(residuals[0] / len(X))
sigma_intercept = np.sqrt(residuals[0] / len(X) * np.mean(X ** 2) / np.var(X))

chi_squared = np.sum((residuals / data_df['global mean concentration sd']) ** 2)
reduced_chi_squared = chi_squared / (len(X) - 2)

print("Parameters:")
print("Slope (m):", slope)
print("Intercept (b):", intercept)
print("\nErrors:")
print("Error in slope (σ_m):", sigma_slope)
print("Error in intercept (σ_b):", sigma_intercept)
print("\nFinal Equation:")
print("Global Mean Concentration = {:.4f} * Date + {:.4f}".format(slope, intercept))
print("\nReduced Chi-squared value:", reduced_chi_squared)
