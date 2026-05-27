import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# 1. Load Dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data["species"] = iris.target

# Convert numbers to names
data["species"] = data["species"].map({
    0: "setosa",
    1: "versicolor",
    2: "virginica"
})

print("\nFirst 5 rows:")
print(data.head())

# 2. VISUALIZATION (GRAPHS)

# Pairplot
sns.pairplot(data, hue="species")
plt.show()

# Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(data.drop("species", axis=1).corr(),
            annot=True,
            cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# 3. Split Data

X = data.drop("species", axis=1)
y = data["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 4. Train Model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
import joblib

joblib.dump(model, "iris_model.pkl")
print("Model saved successfully!")
# 5. Predictions
y_pred = model.predict(X_test)
# 6. Evaluation
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Heatmap of confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.show()

# 7. Predict New Flower
sample = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = model.predict(sample)

print("\nPrediction for sample input:")
print(prediction[0])