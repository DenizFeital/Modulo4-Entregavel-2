import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = pd.read_csv('seeds_dataset.csv')

# Step 2: Separate features (X) and target (y)
X = data.drop('Target', axis=1)  # All features
y = data['Target']  # Target variable

# Step 3: Split the data into training and testing sets (70% for training, 30% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Normalize the data (scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Define the classifiers and parameter grids for GridSearchCV or RandomizedSearchCV
models = {
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000)
}

param_grids = {
    'K-Nearest Neighbors': {
        'n_neighbors': [3, 5, 7, 10],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Logistic Regression': {
        'C': [0.1, 1, 10],
        'penalty': ['l2', 'none'],
        'solver': ['liblinear', 'saga']
    }
}

# Step 6: Use GridSearchCV or RandomizedSearchCV for hyperparameter optimization
results = {}

for model_name, model in models.items():
    print(f"\nOptimizing {model_name}...")

    # Using GridSearchCV for optimization
    grid_search = GridSearchCV(model, param_grids[model_name], cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train_scaled, y_train)
    
    # Best parameters found
    best_params = grid_search.best_params_
    print(f"Best parameters for {model_name}: {best_params}")

    # Retrain the model with the best parameters
    optimized_model = grid_search.best_estimator_

    # Make predictions on the test set
    y_pred = optimized_model.predict(X_test_scaled)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Store the results
    results[model_name] = {
        'Best Parameters': best_params,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Confusion Matrix': conf_matrix
    }

# Step 7: Display the results
for model_name, metrics in results.items():
    print(f"\n{model_name} Performance (After Optimization):")
    print(f"Best Parameters: {metrics['Best Parameters']}")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print(f"F1 Score: {metrics['F1 Score']:.4f}")
    
    # Plot the confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(metrics['Confusion Matrix'], annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title(f"{model_name} Confusion Matrix (After Optimization)")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Step 8: Compare the models (optimized performance)
comparison_df = pd.DataFrame(results).T
print("\nModel Comparison (After Optimization):")
print(comparison_df)
