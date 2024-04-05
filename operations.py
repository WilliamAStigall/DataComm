import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve

# Load and prepare data function
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path, low_memory=False).dropna()
    columns_to_drop = [col for col in df.columns if '_max' in col]
    df = df.drop(columns=columns_to_drop + ['date'], axis=1, errors='ignore')
    df['season'] = df['season'].astype(int)
    return df

# Preprocess data function
def preprocess_data(df, season, team=None):
    team_df = df[(df['season'] == season) & (df['team'] == team)] if team else df[df['season'] == season]
    features = pd.get_dummies(team_df.drop(['won', 'ortg', 'drtg', 'team', 'season'], axis=1), drop_first=True)
    target = team_df['won']
    return features, target

# Standardize features function
def standardize_features(train_features, test_features=None):
    scaler = StandardScaler().fit(train_features)
    train_scaled = scaler.transform(train_features)
    test_scaled = scaler.transform(test_features) if test_features is not None else None
    return train_scaled, test_scaled

# Train and evaluate model function
def train_and_evaluate_model(X_train, y_train, X_test=None, y_test=None, max_iter=10000):
    model = LogisticRegression(max_iter=max_iter).fit(X_train, y_train)
    if X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        print(f"Model accuracy: {accuracy_score(y_test, y_pred):.2f}")
    return model

# Modify multiple features function
def modify_multiple_features(features, feature_names, changes, scaler):
    modified_features = features.copy()
    for column, value in changes.items():
        if column in feature_names:
            column_index = feature_names.index(column)
            modified_features[:, column_index] += value / scaler.scale_[column_index]
    return modified_features

# Evaluate model predictions function
def evaluate_model_predictions(model, X, y):
    y_pred = model.predict(X)
    print(f"Accuracy: {accuracy_score(y, y_pred):.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))
    print("\nClassification Report:")
    print(classification_report(y, y_pred))

# Plot ROC curve function
def plot_roc_curve(model, X, y):
    y_scores = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# Plot learning curve function
def plot_learning_curve(estimator, X, y, title="Learning Curve"):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), scoring="accuracy")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()

# Correctly calculating additional wins
def calculate_additional_wins(y_pred_original, y_pred_modified, y_true):
    """
    Calculates the additional wins as a result of modifying the input features for a logistic regression model.
    Ensures that modifications with (0, 0) result in no additional wins.

    Args:
    - y_pred_original: Array of predictions from the model using the original features.
    - y_pred_modified: Array of predictions from the model using the modified features.
    - y_true: Array of the true outcome labels.

    Returns:
    - additional_wins: The number of additional wins predicted after the feature modifications.
    """
    # Identifying wins that are only predicted after modifications
    additional_wins = ((y_pred_modified == 1) & (y_pred_original == 0)).sum()
    return additional_wins

# Example: Plot learning curve for the Hawks model

# Your main workflow, corrections, and improvements applied as necessary
df = load_and_prepare_data("nba_games.csv")
features_hawks, target_hawks = preprocess_data(df, 2022, 'ATL')
features_remaining, target_remaining = preprocess_data(df, 2022)
features_all, target_all = preprocess_data(df, 2022)
features_hawks_scaled, _ = standardize_features(features_hawks)
_, features_all_scaled = standardize_features(features_remaining, features_all)

model_hawks = train_and_evaluate_model(features_hawks_scaled, target_hawks)
model_all = train_and_evaluate_model(features_all_scaled, target_all)

feature_names = features_hawks.columns.tolist()
changes = {'stl': 1, 'pf': 1}
scaler_for_all = StandardScaler().fit(features_all)
features_hawks_scaled_modified_multiple = modify_multiple_features(features_hawks_scaled, feature_names, changes, scaler_for_all)
coefficients = pd.DataFrame({'Feature': feature_names, 'Importance': model_hawks.coef_[0]}).sort_values(by='Importance', key=abs, ascending=False).head(10)
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=coefficients)
plt.title('Top 10 Important Features for Hawks Win Prediction', fontsize=16)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

y_pred_hawks_multiple_changes = model_all.predict(features_hawks_scaled_modified_multiple)
evaluate_model_predictions(model_all, features_hawks_scaled_modified_multiple, target_hawks)
# Predict outcomes with the Hawks model based on the modified Hawks features
y_pred_hawks_modified = model_hawks.predict(features_hawks_scaled_modified_multiple)

# Evaluate the impact of multiple changes on the Hawks model predictions

def calculate_correct_additional_wins(y_pred_original, y_pred_modified, y_true):
    correct_predictions_original = (y_pred_original == y_true).sum()
    correct_predictions_modified = (y_pred_modified == y_true).sum()
    additional_correct_predictions = correct_predictions_modified - correct_predictions_original
    return max(0, additional_correct_predictions)

y_pred_original_hawks = model_hawks.predict(features_hawks_scaled)
y_pred_modified_hawks = model_hawks.predict(features_hawks_scaled_modified_multiple)
additional_wins_hawks = calculate_correct_additional_wins(y_pred_original_hawks, y_pred_modified_hawks, target_hawks)
print(f"Correct additional wins for Hawks model: {additional_wins_hawks}")

y_pred_original_all = model_all.predict(features_all_scaled)
features_all_scaled_modified = modify_multiple_features(features_all_scaled, feature_names, changes, scaler_for_all)
y_pred_modified_all = model_all.predict(features_all_scaled_modified)
additional_wins_all = calculate_correct_additional_wins(y_pred_original_all, y_pred_modified_all, target_all)
print(f"Correct additional wins for league-wide model: {additional_wins_all}")


from sklearn.metrics import accuracy_score

def calculate_additional_wins_and_accuracy(y_pred_original, y_pred_modified, y_true):
    # Calculate additional correct predictions
    additional_correct_predictions = (y_pred_modified == y_true).sum() - (y_pred_original == y_true).sum()
    
    # Ensure no negative additional wins
    additional_wins = max(0, additional_correct_predictions)
    
    # Calculate prediction accuracy after modification
    accuracy_after_modification = accuracy_score(y_true, y_pred_modified)
    
    return additional_wins, accuracy_after_modification

# Predict outcomes with the original and modified features for Hawks model
y_pred_original_hawks = model_hawks.predict(features_hawks_scaled)
y_pred_modified_hawks = model_hawks.predict(features_hawks_scaled_modified_multiple)

# Calculate additional wins and prediction accuracy for Hawks model
additional_wins_hawks, accuracy_hawks_modified = calculate_additional_wins_and_accuracy(y_pred_original_hawks, y_pred_modified_hawks, target_hawks)
print(f"Additional wins for Hawks model: {additional_wins_hawks}")
print(f"Prediction accuracy after modification for Hawks model: {accuracy_hawks_modified:.2%}")

# Predict outcomes with the original and modified features for the league-wide model
y_pred_original_all = model_all.predict(features_all_scaled)
y_pred_modified_all = model_all.predict(features_all_scaled_modified)

# Calculate additional wins and prediction accuracy for league-wide model
additional_wins_all, accuracy_all_modified = calculate_additional_wins_and_accuracy(y_pred_original_all, y_pred_modified_all, target_all)
print(f"Additional wins for league-wide model: {additional_wins_all}")
print(f"Prediction accuracy after modification for league-wide model: {accuracy_all_modified:.2%}")




# Filter for Hawks' road games only
df_hawks_road = df[(df['team'] == 'ATL') & (df['home'] == 0.0)]
features_hawks_road, target_hawks_road = preprocess_data(df_hawks_road, 2022, 'ATL')

# Standardize features for Hawks' road games
features_hawks_road_scaled, _ = standardize_features(features_hawks_road)

# Train the model for Hawks' road games
model_hawks_road = train_and_evaluate_model(features_hawks_road_scaled, target_hawks_road)

# Modify features for Hawks' road games
features_hawks_road_scaled_modified = modify_multiple_features(features_hawks_road_scaled, feature_names, changes, scaler_for_all)

# Predict and evaluate for Hawks' road games with modified features
y_pred_hawks_road_original = model_hawks_road.predict(features_hawks_road_scaled)
y_pred_hawks_road_modified = model_hawks_road.predict(features_hawks_road_scaled_modified)

# Calculate additional wins and prediction accuracy for Hawks' road games model
additional_wins_hawks_road, accuracy_hawks_road_modified = calculate_additional_wins_and_accuracy(y_pred_hawks_road_original, y_pred_hawks_road_modified, target_hawks_road)
print(f"Additional wins for Hawks' road games model: {additional_wins_hawks_road}")
print(f"Prediction accuracy after modification for Hawks' road games model: {accuracy_hawks_road_modified:.2%}")

# For the league-wide model, continue using the entire dataset as before
# Here, we already have features_all_scaled and target_all prepared

# Predict outcomes with the original and modified features for the league-wide model
y_pred_original_all = model_all.predict(features_all_scaled)
y_pred_modified_all = model_all.predict(features_all_scaled_modified)

# Calculate additional wins and prediction accuracy for the league-wide model
additional_wins_all, accuracy_all_modified = calculate_additional_wins_and_accuracy(y_pred_original_all, y_pred_modified_all, target_all)
print(f"Additional wins for league-wide model: {additional_wins_all}")
print(f"Prediction accuracy after modification for league-wide model: {accuracy_all_modified:.2%}")
