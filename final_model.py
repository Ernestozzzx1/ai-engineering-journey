
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Reproduce our original split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create selected model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        C=10,
        max_iter=1000
    )
)

# Train using all 120 training samples
model.fit(X_train, y_train)

# Predict the 30 test samples
predictions = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, predictions)

print(f"Test Accuracy: {accuracy:.2%}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=iris.target_names
))
