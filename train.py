#part 1
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#part 2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

#part 4
from sklearn.metrics import confusion_matrix, classification_report



#part 1
# Step 1: Load our dataset
iris = load_iris()

# X contains our four flower measurements
X = iris.data

# y contains the correct flower species
y = iris.target

# Step 2: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 3: Examine the results
print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)

print("\nFirst training example:")
print(X_train[0])

print("\nCorrect species:")
print(iris.target_names[y_train[0]])

#part 2

import numpy as np

print("\nTraining class distribution:")
print(np.bincount(y_train))

print("\nTesting class distribution:")
print(np.bincount(y_test))


#part 3

# Step 4: Create our machine learning model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

# Step 5: Train the model
model.fit(X_train, y_train)

print("\nModel training complete!")

# Step 6: Make predictions using unseen data
predictions = model.predict(X_test)

# Step 7: Compare predictions with actual labels
print("\nFirst 10 predictions:")
print(predictions[:10])

print("\nFirst 10 actual labels:")
print(y_test[:10])

# Step 8: Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print(f"\nModel accuracy: {accuracy * 100:.2f}%")


#part 4

# Step 9: Analyze incorrect predictions

cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)

# Step 10: Detailed evaluation

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=iris.target_names
))

# Step 11: Show the incorrect predictions

print("\nMisclassified flowers:")

for i in range(len(y_test)):
    if predictions[i] != y_test[i]:
        print(f"\nTest sample {i}")
        print("Measurements:", X_test[i])
        print("Actual:", iris.target_names[y_test[i]])
        print("Predicted:", iris.target_names[predictions[i]])

#Part 5


# Investigate the model's prediction probabilities
probabilities = model.predict_proba(X_test)

print("\nPrediction probabilities for mistakes:")

for i in range(len(y_test)):
    if predictions[i] != y_test[i]:
        print(f"\nTest sample {i}")

        print("Actual:",
              iris.target_names[y_test[i]])

        print("Predicted:",
              iris.target_names[predictions[i]])

        for class_index, probability in enumerate(probabilities[i]):
            print(
                f"{iris.target_names[class_index]}: "
                f"{probability:.2%}"
            )
