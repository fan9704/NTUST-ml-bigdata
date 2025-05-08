import pandas as pd
import seaborn as sns
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

train_data = pd.read_csv('datasets/mitbih_train.csv', header=None)
test_data = pd.read_csv('datasets/mitbih_test.csv', header=None)

plt.figure(figsize=(10, 5))
train_class_counts = train_data[187].value_counts()
train_class_counts.sort_index(inplace=True)
train_class_counts.plot(kind='bar', color='#66CDAA', edgecolor='black')
plt.title('Train Data')
plt.xlabel('Labels')
plt.ylabel('Count')
for idx, count in enumerate(train_class_counts):
    plt.text(idx, count + 500, str(count), ha='center', va='bottom', fontsize=10)
plt.show()
print('')
plt.figure(figsize=(10, 5))
test_class_counts = test_data[187].value_counts()
test_class_counts.sort_index(inplace=True)
test_class_counts.plot(kind='bar', color='#FF7F7F', edgecolor='black')
plt.title('Test Data')
plt.xlabel('Labels')
plt.ylabel('Count')
for idx, count in enumerate(test_class_counts):
    plt.text(idx, count + 200, str(count), ha='center', va='bottom', fontsize=10)
plt.show()
# === Generate Heatmap ===
plt.figure(figsize=(10, 5))
correlation_matrix = train_data.corr()
sns.heatmap(correlation_matrix, cmap='plasma', annot=False)
plt.title('Heatmap of Dataset Features ')
plt.xlabel('Features')
plt.ylabel('Features')
plt.show()

# === Data Preprocessing ===
# Separate Features and Labels
train_data_features = train_data.iloc[:, :-1]
train_data_labels = train_data.iloc[:, -1]
test_data_features = test_data.iloc[:, :-1]
test_data_labels = test_data.iloc[:, -1]

# Encode Labels as Discrete Classes
le = LabelEncoder()
train_data_labels = le.fit_transform(train_data_labels)
test_data_labels = le.transform(test_data_labels)

# Impute Missing Values (Replace NaN with column mean)
train_data_features = train_data_features.fillna(train_data_features.mean())
test_data_features = test_data_features.fillna(test_data_features.mean())

# Normalize Data
scaler = MinMaxScaler()
train_data_features = pd.DataFrame(scaler.fit_transform(train_data_features))
test_data_features = pd.DataFrame(scaler.transform(test_data_features))

# Apply PCA
pca = PCA(n_components=30)
train_data_features = pd.DataFrame(pca.fit_transform(train_data_features))
test_data_features = pd.DataFrame(pca.transform(test_data_features))

# Save preprocessed data if needed for reusability
pd.DataFrame(train_data_features).to_csv('train_data_features.csv', index=False)
pd.DataFrame(test_data_features).to_csv('test_data_features.csv', index=False)
pd.Series(train_data_labels).to_csv('train_data_labels.csv', index=False)
pd.Series(test_data_labels).to_csv('test_data_labels.csv', index=False)

print("Preprocessing completed. Training and testing datasets are ready.")

# === Initialize models ===

models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Linear SVC": LinearSVC(random_state=42, max_iter=10000),
    "SVC": SVC(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric='mlogloss')
}

# === Train and evaluate models ===
results = []

for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(train_data_features, train_data_labels)  # Train the model

    y_pred = model.predict(test_data_features)  # Predict on test data

    # Evaluate model performance
    acc = accuracy_score(test_data_labels, y_pred)
    f1 = f1_score(test_data_labels, y_pred, average='weighted')
    precision = precision_score(test_data_labels, y_pred, average='weighted')
    recall = recall_score(test_data_labels, y_pred, average='weighted')

    results.append({
        "Model": model_name,
        "Accuracy": acc,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall
    })

# Display the results
results_df = pd.DataFrame(results)
print(results_df)

# === The performance metrics obtained from XGboost, SVC, random forest, and linearSVC models ===

models = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

# Extract metrics dynamically
random_forest = results_df[results_df['Model'] == 'Random Forest'].iloc[0, 1:].values
linear_svc = results_df[results_df['Model'] == 'Linear SVC'].iloc[0, 1:].values
svc = results_df[results_df['Model'] == 'SVC'].iloc[0, 1:].values
xgboost = results_df[results_df['Model'] == 'XGBoost'].iloc[0, 1:].values

# Combine results into a single array
data = [random_forest, linear_svc, svc, xgboost]
model_labels = ['Random Forest', 'Linear SVC', 'SVC', 'XGBoost']

# Plot each model's performance metrics
plt.figure(figsize=(10, 5))
for i, label in enumerate(model_labels):
    plt.plot(models, data[i], marker='o', label=label, linestyle=['-', '--', '-.', ':'][i])

# Customize the plot
plt.title('Performance Metrics of Models', fontsize=18)
plt.xlabel('Metrics', fontsize=14)
plt.ylabel('Performance', fontsize=14)
plt.ylim(0.65, 1.0)  # Adjust y-axis range for better visualization
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Add annotations for each data point
for j, metric in enumerate(models):
    for i, label in enumerate(model_labels):
        plt.text(j, data[i][j] + 0.005, f"{data[i][j]:.3f}", ha='center', fontsize=10)

# Display the plot
plt.tight_layout()
plt.show()

# === The performance metrics obtained from XBboost, and XGboost with Original, LevyJA, JADE, and EnhancedAEO hybrid models===

# Plot results
metrics_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
linestyles = ['-', '--', '-.', ':']  # Define line styles
plt.figure(figsize=(10, 5))

# Loop through results and assign a linestyle based on the index
for i, (opt, result) in enumerate(results.items()):
    metrics = result['Metrics']
    plt.plot(metrics_names, list(metrics.values()), marker="o", label=opt, linestyle=linestyles[i % len(linestyles)])

# Customize the plot
plt.xlabel("Metrics")
plt.ylabel("Performance")
plt.title("Performance Metrics for Different Optimizers")
plt.legend(fontsize=12)
plt.ylim(0.65, 1.0)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

import pprint

pprint.pprint(results)
