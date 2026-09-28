import pandas as pd, numpy as np
df = pd.read_csv('car_fuel_efficiency_2026.csv')

len(df)
df.fuel_type.nunique()
(df.isnull().sum() > 0).sum()
df[df.origin == 'Asia'].fuel_efficiency_mpg.max()

m1 = df.horsepower.median()
mode = df.horsepower.mode()[0]
m2 = df.horsepower.fillna(mode).median()

X = df[df.origin == 'Asia'][['vehicle_weight', 'model_year']].iloc[:7].values
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
w = np.linalg.inv(X.T @ X) @ X.T @ y
w.sum()