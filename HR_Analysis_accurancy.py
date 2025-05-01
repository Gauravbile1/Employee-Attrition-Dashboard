import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(r"C:\Users\Gaurav Bile\Videos\1Study\SKY internship\Employee Attrition Analysis And Visualization\Preprocessed_Data.csv")

# Encode categorical variables
encoder = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = encoder.fit_transform(data[col])

# Define features (X) and target (y)
X = data.drop('Attrition', axis=1)
y = data['Attrition']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Hyperparameter Tuning ---
param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': range(3, 10)}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)

# Train the best model
best_model = grid_search.best_estimator_

# --- Cross-Validation ---
scores = cross_val_score(best_model, X, y, cv=5)
print("Cross-validation scores:", scores)
print("Average Cross-validation score:", scores.mean())

# --- Model Evaluation ---
y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- Feature Importance Visualization ---
plt.figure(figsize=(10, 6))
plt.barh(X.columns, best_model.feature_importances_)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance in Decision Tree Model')
plt.show()