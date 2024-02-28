import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns




df = pd.read_csv("nba_games.csv")

# Count NaN values in the 'ortg' and 'won' columns
nan_count_ortg = df['ortg'].isna().sum()
nan_count_won = df['won'].isna().sum()

print(f"Number of NaN values in 'ortg': {nan_count_ortg}")
print(f"Number of NaN values in 'won': {nan_count_won}")

# Create a copy of the DataFrame without NaN values in 'ortg' and 'won' columns
df_non_nan = df.dropna(subset=['ortg', 'won'])

# If you also need to ensure 'drtg' has no NaNs for the correlation calculation:
df_non_nan = df_non_nan.dropna(subset=['drtg'])


# Recalculate correlations with the non-NaN DataFrame
ortg_win_correlation, _ = stats.pearsonr(df_non_nan['ortg'], df_non_nan['won'])
drtg_win_correlation, _ = stats.pearsonr(df_non_nan['drtg'], df_non_nan['won'])

print(f"Number of NaN values in 'ortg' after cleaning: {df_non_nan['ortg'].isna().sum()}")
print(f"Number of NaN values in 'won' after cleaning: {df_non_nan['won'].isna().sum()}")

print(f"Correlation between Offensive Rating and Wins (cleaned data): {ortg_win_correlation}")
print(f"Correlation between Defensive Rating and Wins (cleaned data): {drtg_win_correlation}")




#Correlation between Offensive Rating and Wins (cleaned data): 0.5139528802382525
#Correlation between Defensive Rating and Wins (cleaned data): -0.5139528802382525

# Create a contingency table
contingency_table = pd.crosstab(df['3p'], df['won'])

# Perform Chi-Square test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print(f"Chi-Square Statistic: {chi2}, p-value: {p}")

df_non_nan['fg_percentage'] = (df_non_nan['fg'] / df_non_nan['fga']) * 100
# Perform two-way ANOVA
model = ols('fg_percentage ~ C(home) * C(won)', data=df).fit()
anova_results = sm.stats.anova_lm(model, typ=2)

print(anova_results)

# Logistic Regression
X = df_non_nan[['stat1', 'stat2', 'statN']]  # Replace with actual stat columns
y = df_non_nan['win_loss']

X = sm.add_constant(X)  # Adds a constant term to the predictors
logit_model = sm.Logit(y, X).fit()

print(logit_model.summary())




# Plot 1: Correlation Matrix of 'ortg', 'drtg', 'won'
corr_columns = ['ortg', 'drtg', 'won']  # Add or remove columns as needed
corr = df_non_nan[corr_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix for ORTG, DRTG, and Wins')
plt.show()

# Plot 2: Boxplot of Field Goal Attempts (FGA) by Home/Away and Win/Loss
plt.figure(figsize=(10, 8))
sns.boxplot(x='home', y='fga', hue='won', data=df)
plt.title('Field Goal Attempts by Home/Away and Win/Loss')
plt.show()

# Plot 3: Count Plot for Three-Point Outcome by Game Outcome
plt.figure(figsize=(10, 8))
sns.countplot(x='three_point_outcome', hue='game_outcome', data=df)
plt.title('Three-Point Outcome by Game Result')
plt.show()

# Assuming 'stat1', 'stat2', 'statN' are placeholders, replace them with actual column names for the regression
# For demonstration, let's assume these columns exist. Replace them with actual statistics columns.
# Plot 4: Distributions of a Statistic by Win/Loss - Example with 'ortg'
plt.figure(figsize=(10, 8))
sns.histplot(data=df_non_nan, x='ortg', hue='won', kde=True, element="step", stat="density", common_norm=False)
plt.title('Distribution of Offensive Rating by Win/Loss')
plt.show()
