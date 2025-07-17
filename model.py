import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv(r"C:\Users\WIN 10\Desktop\mlp\data\StressLevelDataset.csv")

# Define features and target
X = data.drop('stress_level', axis=1)
y = data['stress_level']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate and save the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
joblib.dump(model, 'model_rf.pkl')
