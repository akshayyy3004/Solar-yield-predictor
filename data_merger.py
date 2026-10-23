import pandas as pd

print("Loading raw Kaggle datasets...")
gen_df = pd.read_csv('dataset/Plant_1_Generation_Data.csv')
sensor_df = pd.read_csv('dataset/Plant_1_Weather_Sensor_Data.csv')

print("Standardizing timestamps...")
gen_df['DATE_TIME'] = pd.to_datetime(gen_df['DATE_TIME'], format='%d-%m-%Y %H:%M')
sensor_df['DATE_TIME'] = pd.to_datetime(sensor_df['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')

print("Merging datasets...")
merged_df = pd.merge(gen_df, sensor_df, on='DATE_TIME', how='inner')

final_dataset = merged_df[['DATE_TIME', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'DAILY_YIELD']]

final_dataset.to_csv('dataset/solar_weather_data.csv', index=False)
print(f"Success! Merged {len(merged_df)} rows into solar_weather_data.csv.")
