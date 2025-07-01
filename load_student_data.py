import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

try:
    df = pd.read_csv('student-mat.csv', sep=';')
    print("✅ Dataset loaded and code reached this point successfully!")
except FileNotFoundError:
    print("❌ File not found.")
    exit()
print("\n📊 Basic Info:")
print(df.info())

print("\n🧹 Missing Values:")
print(df.isnull().sum())
# Drop target column from features
X = df.drop('G3', axis=1)
y = df['G3']
# Convert categorical text data to numbers
X_encoded = pd.get_dummies(X, drop_first=True)
print(f"\n✅ Encoded shape: {X_encoded.shape}")
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
print(f"✅ Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n📈 Model Evaluation:")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")
input("\nPress Enter to exit...")
import joblib

# Save the model
joblib.dump(model, 'model.pkl')
joblib.dump(X_encoded.columns.tolist(), 'model_columns.pkl')

print("✅ Model saved as model.pkl")

