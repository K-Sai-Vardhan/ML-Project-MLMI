#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import sklearn

print(f"Using scikit-learn: {sklearn.__version__}")

FILE = "ML_CuNi_Training_Data.txt"
if not os.path.exists(FILE):
    print("ERROR: data file not found:", FILE); sys.exit(1)

# Read raw data, ignore comment lines starting with '#'
raw = pd.read_csv(FILE, sep=r"\s+", comment="#", header=None, engine='python')
ncols = raw.shape[1]
print("Detected columns in file:", ncols)

# Handle different possible column counts robustly
if ncols == 6:
    # e.g. [Step, Temp, Press, Vol, Dens, PotEng]
    raw.columns = ["Step", "Temp", "Press", "Vol", "Dens", "PotEng"]
elif ncols == 7:
    # e.g. [TimeStep, v_step, v_temp, v_press, v_vol, v_dens, v_poteng]
    raw.columns = ["Step", "Step2", "Temp", "Press", "Vol", "Dens", "PotEng"]
    # drop duplicate step column
    raw = raw.drop(columns=["Step2"])
else:
    print("Unexpected number of columns:", ncols)
    print("Head of file (first 10 rows):\n", raw.head(10).to_string(index=False))
    sys.exit(1)

print("Final columns used:", raw.columns.tolist())
df = raw.dropna().reset_index(drop=True)
print("Loaded rows:", len(df))
print(df.head())

# Select features and target
X = df[["Temp", "Press", "Vol", "Dens"]].astype(float)
y = df["PotEng"].astype(float)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Model (Random Forest)
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
print("Training Random Forest...")
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
# RMSE in a way compatible with old sklearn
try:
    rmse = mean_squared_error(y_test, y_pred, squared=False)
except TypeError:
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n=== Metrics (test set) ===")
print(f"R^2  : {r2:.4f}")
print(f"MAE  : {mae:.6f}")
print(f"RMSE : {rmse:.6f}")

# 5-fold cross-val R^2 (lite)
print("\nRunning 5-fold CV (R^2)...")
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1)
print("CV R^2 scores:", np.round(cv_scores, 4))
print("CV mean / std:", np.round(cv_scores.mean(), 4), "/", np.round(cv_scores.std(), 4))

# Save predictions CSV
out = pd.DataFrame({"True": y_test.values, "Pred": y_pred})
out.to_csv("predictions_summary.csv", index=False)
print("Saved predictions_summary.csv")

# Plots (clean academic)
# True vs Predicted
plt.figure(figsize=(5.5,5.5))
plt.scatter(y_test, y_pred, s=18, alpha=0.6)
mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
plt.plot([mn, mx], [mn, mx], 'r--', linewidth=1)
plt.xlabel("True PotEng")
plt.ylabel("Predicted PotEng")
plt.title(f"True vs Predicted (R^2={r2:.3f})")
plt.grid(linestyle=':', linewidth=0.5)
plt.tight_layout()
plt.savefig("CuNi_ML_Performance.png", dpi=200)
plt.close()
print("Saved CuNi_ML_Performance.png")

# Feature importance
feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(6,3.5))
feat_imp.plot(kind='bar', edgecolor='k')
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=200)
plt.close()
print("Saved feature_importance.png")

# Residuals histogram
res = y_test - y_pred
plt.figure(figsize=(6,4))
plt.hist(res, bins=30, edgecolor='k')
plt.xlabel("Residual (True - Predicted)")
plt.ylabel("Counts")
plt.title("Residuals Histogram")
plt.tight_layout()
plt.savefig("residuals_hist.png", dpi=200)
plt.close()
print("Saved residuals_hist.png")

# Explanations for PPT (printed)
print("\nPlot explanations (copy-paste into PPT):")
print("- True vs Predicted: points near diagonal (y=x) show good predictions. R^2 shown in title.")
print("- Feature Importance: bars show which input variables the model relied on (higher = more important).")
print("- Residuals Histogram: centered near zero and narrow spread implies small unbiased errors.")

print("\nAll done.")
