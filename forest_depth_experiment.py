
from sklearn.datasets import load_iris
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)
from sklearn.ensemble import RandomForestClassifier

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Same training/test split as earlier
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Same cross-validation folds
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

depths = [1, 2, 3, 5, 10, None]

print("Random Forest Depth Experiment")
print("-" * 55)

for depth in depths:

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=depth,
        random_state=42
    )

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        return_train_score=True
    )

    train_acc = scores["train_score"].mean()
    val_acc = scores["test_score"].mean()
    gap = train_acc - val_acc

    print(
        f"Depth={str(depth):<5} "
        f"Train={train_acc:.2%} "
        f"Validation={val_acc:.2%} "
        f"Gap={gap:.2%}"
    )
