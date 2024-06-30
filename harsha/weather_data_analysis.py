import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Create a sample dataset
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', end='2023-12-31')
df = pd.DataFrame({
    'Date': dates,
    'MinTemp': np.random.uniform(10, 20, len(dates)),
    'MaxTemp': np.random.uniform(20, 30, len(dates)),
    'Rainfall': np.random.exponential(5, len(dates))
})

print(df.head())
print(df.info())
print(df.describe())

sns.pairplot(df[['MinTemp', 'MaxTemp', 'Rainfall']])
plt.show()

df['Month'] = df['Date'].dt.month
monthly_avg_max_temp = df.groupby('Month')['MaxTemp'].mean()

plt.figure(figsize=(10, 5))
plt.plot(monthly_avg_max_temp.index, monthly_avg_max_temp.values, marker='o')
plt.xlabel('Month')
plt.ylabel('Average Max Temperature')
plt.title('Monthly Average Max Temperature')
plt.grid(True)
plt.show()

X = df[['MinTemp', 'MaxTemp']]
y = df['Rainfall']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error for Rainfall Prediction: {mse:.4f}')

highest_temp_month = monthly_avg_max_temp.idxmax()
lowest_temp_month = monthly_avg_max_temp.idxmin()
print(f'Highest temperature month: {highest_temp_month}, Lowest temperature month: {lowest_temp_month}')