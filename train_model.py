import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
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
    "rf": RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=20, random_state=42),
    "gb": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42),
    "svr": SVR(kernel='rbf', C=100, epsilon=0.1)
}

display_names = {
    "rf": "Random Forest",
    "gb": "Gradient Boosting",
    "svr": "Support Vector Regressor"
}

print("\n--- MODEL TRAINING, EXPORT & LEARNING CURVES ---")
for key, model in models.items():
    print(f"\nTraining {display_names[key]}...")
    model.fit(X_train, y_train)
    
    test_predictions = model.predict(X_test)
    test_r2 = r2_score(y_test, test_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)
    
    print(f"{display_names[key]} -> Test R²: {test_r2:.4f} | MAE: {test_mae:.2f}")
    
    joblib.dump(model, f'dataset/{key}_model.pkl')
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, test_predictions, alpha=0.3, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(f'Actual vs Predicted Yield ({display_names[key]})')
    plt.xlabel('Actual Daily Yield (kWh)')
    plt.ylabel('Predicted Daily Yield (kWh)')
    plt.grid(True)
    plt.savefig(f'static/{key}_scatter.png') 
    plt.close()

    print(f"Generating Learning Curve for {display_names[key]}...")
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error', 
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )
    
    train_errors = -np.mean(train_scores, axis=1)
    test_errors = -np.mean(test_scores, axis=1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_errors, 'o-', color="red", label="Training Error")
    plt.plot(train_sizes, test_errors, 'o-', color="green", label="Validation Error")
    plt.title(f'Learning Curve: {display_names[key]}')
    plt.xlabel('Training Set Size')
    plt.ylabel('Mean Absolute Error (Lower is Better)')
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(f'static/{key}_learning_curve.png')
    plt.close()

joblib.dump(scaler, 'dataset/scaler.pkl')

print("\nSUCCESS!")
print("Saved 3 scatter plots (e.g., gb_scatter.png)")
print("Saved 3 learning curves (e.g., gb_learning_curve.png)")
