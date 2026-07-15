# ============================================
# Task 5 - Decision Tree and Random Forest
# Heart Disease Prediction
# ============================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# --------------------------------------------
# Load Dataset
# --------------------------------------------

df = pd.read_csv("heart.csv")

print("\nFirst Five Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nMissing Values")
print(df.isnull().sum())

# --------------------------------------------
# Features and Target
# --------------------------------------------

X = df.drop("target", axis=1)
y = df["target"]

# --------------------------------------------
# Train Test Split
# --------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ============================================
# Decision Tree
# ============================================

decision_tree = DecisionTreeClassifier(
    random_state=42
)

decision_tree.fit(X_train, y_train)

prediction_dt = decision_tree.predict(X_test)

accuracy_dt = accuracy_score(
    y_test,
    prediction_dt
)

print("\nDecision Tree Accuracy")
print(accuracy_dt)

# ============================================
# Tree Visualization
# ============================================

plt.figure(figsize=(18,10))

plot_tree(
    decision_tree,
    filled=True,
    feature_names=X.columns,
    class_names=["No Disease","Disease"],
    fontsize=8
)

plt.title("Decision Tree")

plt.show()

# ============================================
# Overfitting Analysis
# ============================================

depths = [1,2,3,4,5,6,7,8,9,10]

train_scores = []

test_scores = []

for depth in depths:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train,y_train)

    train_scores.append(
        model.score(X_train,y_train)
    )

    test_scores.append(
        model.score(X_test,y_test)
    )

plt.figure(figsize=(8,5))

plt.plot(
    depths,
    train_scores,
    marker='o',
    label="Training Accuracy"
)

plt.plot(
    depths,
    test_scores,
    marker='o',
    label="Testing Accuracy"
)

plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.title("Overfitting Analysis")
plt.legend()

plt.show()

# ============================================
# Random Forest
# ============================================

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest.fit(
    X_train,
    y_train
)

prediction_rf = random_forest.predict(
    X_test
)

accuracy_rf = accuracy_score(
    y_test,
    prediction_rf
)

print("\nRandom Forest Accuracy")
print(accuracy_rf)

# ============================================
# Feature Importance
# ============================================

importance = pd.Series(
    random_forest.feature_importances_,
    index=X.columns
)

importance = importance.sort_values(
    ascending=False
)

print("\nFeature Importance")

print(importance)

plt.figure(figsize=(10,6))

importance.plot(
    kind="bar"
)

plt.title("Feature Importance")

plt.ylabel("Importance")

plt.show()

# ============================================
# Cross Validation
# ============================================

cv_score = cross_val_score(
    random_forest,
    X,
    y,
    cv=5
)

print("\nCross Validation Scores")

print(cv_score)

print("\nAverage Cross Validation Accuracy")

print(cv_score.mean())

# ============================================
# Accuracy Comparison
# ============================================

print("\n--------------------------------")
print("Model Comparison")
print("--------------------------------")

print("Decision Tree Accuracy :",accuracy_dt)
print("Random Forest Accuracy :",accuracy_rf)

if accuracy_rf > accuracy_dt:
    print("\nRandom Forest performed better.")
else:
    print("\nDecision Tree performed better.")