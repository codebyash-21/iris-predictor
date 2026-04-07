import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import joblib

# Load the classic Iris dataset (Flower measurements)
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and Train the Model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save the "brain" of our AI to a file
joblib.dump(model, 'iris_model.pkl')
print("Model trained and saved as iris_model.pkl")