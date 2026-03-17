import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.set_page_config(page_title="RF Visualization", layout="wide")

st.title("🌳 Random Forest Regression Visualization")

# Sidebar controls
st.sidebar.header("⚙️ Parameters")

n_estimators = st.sidebar.slider("n_estimators", 10, 300, 100)
max_depth = st.sidebar.slider("max_depth", 1, 30, 10)
noise = st.sidebar.slider("Noise", 0.0, 50.0, 10.0)
test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20)

# =========================
# Generate synthetic dataset
# =========================
X, y = make_regression(
    n_samples=300,
    n_features=1,
    noise=noise,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size/100, random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# =========================
# Show R2 Score
# =========================
st.subheader("📈 Model Performance")

r2 = r2_score(y_test, y_pred)
st.metric("R² Score", f"{r2:.2f}")

# =========================
# 📊 1. Data + Predictions
# =========================
st.subheader("📊 Regression Fit Visualization")

# Sort for smooth curve
X_grid = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
y_grid_pred = model.predict(X_grid)

fig1, ax1 = plt.subplots()
ax1.scatter(X, y)  # original data
ax1.plot(X_grid, y_grid_pred)  # model prediction curve
ax1.set_title("Random Forest Regression Fit")

st.pyplot(fig1)

# =========================
# 📊 2. Actual vs Predicted
# =========================
st.subheader("📉 Actual vs Predicted")

fig2, ax2 = plt.subplots()
ax2.scatter(y_test, y_pred)
ax2.set_xlabel("Actual")
ax2.set_ylabel("Predicted")
ax2.set_title("Actual vs Predicted")

st.pyplot(fig2)

# =========================
# 📊 3. Residual Plot
# =========================
st.subheader("📉 Residual Plot")

residuals = y_test - y_pred

fig3, ax3 = plt.subplots()
ax3.scatter(y_pred, residuals)
ax3.set_xlabel("Predicted")
ax3.set_ylabel("Residuals")
ax3.set_title("Residual Plot")

st.pyplot(fig3)