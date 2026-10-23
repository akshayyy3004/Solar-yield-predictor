import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

print("Loading and cleaning dataset...")
df = pd.read_csv('dataset/solar_weather_data.csv')
df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
df['HOUR'] = df['DATE_TIME'].dt.hour

def categorize_time(hour):
    if hour < 6 or hour >= 19:
        return 'Night'
    return 'Day'

df['TIME_CATEGORY'] = df['HOUR'].apply(categorize_time)
df_clean = df[(df['TIME_CATEGORY'] != 'Night') & (df['IRRADIATION'] > 0)]

X = df_clean[['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']]
y = df_clean['DAILY_YIELD']

print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Splitting data: 70% Train, 15% Validation, 15% Test...")
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=20, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42),
    "Support Vector (SVR)": SVR(kernel='rbf', C=100, epsilon=0.1)
}

best_model = None
best_score = -float('inf')
best_name = ""

print("\n MODEL TRAINING & VALIDATION ")
for name, model in models.items():
    model.fit(X_train, y_train)
    val_predictions = model.predict(X_val)
    val_r2 = r2_score(y_val, val_predictions)
    
    print(f"{name} Validation R²: {val_r2:.4f}")
    
    if val_r2 > best_score:
        best_score = val_r2
        best_model = model
        best_name = name

print(f"\n FINAL TEST SET RESULTS ({best_name}) ")
test_predictions = best_model.predict(X_test)
test_r2 = r2_score(y_test, test_predictions)
test_mae = mean_absolute_error(y_test, test_predictions)
print(f"Test R² Score: {test_r2:.4f}")
print(f"Test MAE: {test_mae:.2f} kWh")

joblib.dump(best_model, 'dataset/solar_model.pkl')
joblib.dump(scaler, 'dataset/scaler.pkl')
print(f"\nSuccess! '{best_name}' won and was saved to 'solar_model.pkl'.")
print("Scaler saved to 'scaler.pkl'.")

print("\nGenerating graph...")
plt.figure(figsize=(8, 6))
plt.scatter(y_test, test_predictions, alpha=0.3, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title(f'Actual vs Predicted Yield ({best_name})')
plt.xlabel('Actual Daily Yield (kWh)')
plt.ylabel('Predicted Daily Yield (kWh)')
plt.grid(True)
plt.savefig('dataset/model_proof.png')
print("Graph saved as 'model_proof.png' in your dataset folder!")
