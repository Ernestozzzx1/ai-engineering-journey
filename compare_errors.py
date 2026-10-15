
from sklearn.datasets import load_iris
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Keep our original test set separate
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Same five validation folds
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(C=10, max_iter=1000)
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=1,
        random_state=42
    )
}

for name, model in models.items():

    # Predictions made without training on
    # the validation example being predicted
    predictions = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv
    )

    print(f"\n{name}")
    print("-" * 35)

    print("Confusion Matrix:")
    print(confusion_matrix(y_train, predictions))

    print("\nIncorrect predictions:")

    for i in range(len(y_train)):
        if predictions[i] != y_train[i]:
            print(
                f"Sample {i}: "
                f"Actual={iris.target_names[y_train[i]]}, "
                f"Predicted={iris.target_names[predictions[i]]}"
            )


# Compare the two models' incorrect predictions

lr_predictions = cross_val_predict(
    models["Logistic Regression"],
    X_train,
    y_train,
    cv=cv
)

rf_predictions = cross_val_predict(
    models["Random Forest"],
    X_train,
    y_train,
    cv=cv
)

print("\nSamples where the models disagree:")
print("-" * 45)

for i in range(len(y_train)):
    if lr_predictions[i] != rf_predictions[i]:
        print(f"\nSample {i}")
        print("Measurements:", X_train[i])
        print("Actual:",
              iris.target_names[y_train[i]])
        print("Logistic Regression:",
              iris.target_names[lr_predictions[i]])
        print("Random Forest:",
              iris.target_names[rf_predictions[i]])


# Compare prediction probabilities for both models

print("\nPrediction probabilities:")
print("-" * 45)

for name, model in models.items():

    probabilities = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba"
    )

    print(f"\n{name}")

    for i in [7, 73]:
        print(f"\nSample {i}")

        for class_index, probability in enumerate(probabilities[i]):
            print(
                f"{iris.target_names[class_index]}: "
                f"{probability:.2%}"
            )



# Calculate prediction margins

print("\nPrediction Margins")
print("-" * 40)

for name, model in models.items():

    probabilities = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba"
    )

    print(f"\n{name}")

    for sample in [7, 73]:

        # Sort probabilities from lowest to highest
        sorted_probs = sorted(
            probabilities[sample],
            reverse=True
        )

        # Difference between the top two probabilities
        margin = sorted_probs[0] - sorted_probs[1]

        print(
            f"Sample {sample}: "
            f"Margin = {margin:.2%}"
        )
