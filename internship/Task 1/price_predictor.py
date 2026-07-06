import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("real_estate.csv")
print("Raw data shape:", df.shape)
print(df.head(), "\n")

print("Missing values before cleaning:\n", df.isna().sum(), "\n")

for col in ["area", "bathrooms"]:
    df[col] = df[col].fillna(df[col].median())

df = pd.get_dummies(df, columns=["location"], drop_first=True)

print("Missing values after cleaning:\n", df.isna().sum(), "\n")
print("Columns after encoding:", list(df.columns), "\n")

X = df.drop(columns=["price"])
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Performance")
print("-" * 30)
print(f"Mean Squared Error (MSE): {mse:,.2f}")
print(f"Root MSE (RMSE):          {rmse:,.2f}")
print(f"R^2 Score:                {r2:.4f}\n")

print("Feature coefficients:")
for name, coef in zip(X.columns, model.coef_):
    print(f"  {name:12s} {coef:,.2f}")
print(f"  {'intercept':12s} {model.intercept_:,.2f}\n")

sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(y_test, y_pred, alpha=0.6, color="#1f3d99", edgecolor="white", s=60)

lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, "r--", linewidth=2, label="Perfect Prediction")

ax.set_xlabel("Actual Price ($)")
ax.set_ylabel("Predicted Price ($)")
ax.set_title(f"Predicted vs Actual House Prices\nR² = {r2:.3f} | RMSE = ${rmse:,.0f}")
ax.legend()


#you can change the csv data for different plotting 
plt.tight_layout()
plt.savefig("new_predictions_vs_actual.png", dpi=150)

print("Saved plot to new_predictions_vs_actual.png")