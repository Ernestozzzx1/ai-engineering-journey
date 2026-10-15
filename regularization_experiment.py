
from sklearn.datasets import load_iris
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    StratifiedKFold,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load our dataset
iris = load_iris()
X = iris.data
y = iris.target

# Reproduce our original train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Different regularization strengths
c_values = [0.01, 0.1, 1, 10, 100]

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("Regularization Experiment")
print("-" * 45)

for c in c_values:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c,
            max_iter=1000
        )
    )

    
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        return_train_score=True
    )

    training_accuracy = scores["train_score"].mean()
    validation_accuracy = scores["test_score"].mean()

    print(
        f"C={c:<6} "
        f"Train: {training_accuracy:.2%} "
        f"Validation: {validation_accuracy:.2%}"
    )
