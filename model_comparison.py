
from sklearn.datasets import load_iris
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 2: Reproduce our original split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 3: Create the two models
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=10,
            max_iter=1000
        )
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        random_state=42
    )
}

# Step 4: Configure cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Step 5: Compare models
for name, model in models.items():

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        return_train_score=True
    )

    train_accuracy = scores["train_score"].mean()
    validation_accuracy = scores["test_score"].mean()

    print(f"\n{name}")
    print("-" * 35)

    print(f"Training Accuracy: {train_accuracy:.2%}")
    print(f"Validation Accuracy: {validation_accuracy:.2%}")

    print("Individual validation fold scores:")
    print(scores["test_score"])
