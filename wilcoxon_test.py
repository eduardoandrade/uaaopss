import numpy as np
from scipy.stats import wilcoxon

# Generating sample data for N = 10
# Data before the treatment (m0)
before_treatment = np.array([61.6, 61.8, 61.9, 62.4, 62.5, 62.8, 62.9, 63.3, 63.4, 64.0]) # mAP
#before_treatment = np.array([63.9, 64.2, 64.2, 64.5, 64.6, 64.9, 64.9, 65.4, 65.5, 66.0]) # Rank-1

# Data after the treatment (m)
after_treatment = np.array([14.4, 14.2, 14.5, 14.6, 15.0, 15.1, 15.3, 15.5, 15.7, 15.9]) # mAP
#after_treatment = np.array([12.1, 11.8, 12.1, 12.2, 12.5, 12.6, 12.8, 13.0, 13.0, 13.2]) # Rank-1

# Performing the Wilcoxon signed-rank test
stat, p_value = wilcoxon(before_treatment, after_treatment)

# Setting the significance level
alpha = 0.025

# Displaying the results
print("Wilcoxon Test Statistic:", stat)
print("P-value:", p_value)

if p_value < alpha:
    print(f"Reject the null hypothesis (p-value < {alpha}): There is a significant difference.")
else:
    print(f"Fail to reject the null hypothesis (p-value >= {alpha}): There is no significant difference.")
