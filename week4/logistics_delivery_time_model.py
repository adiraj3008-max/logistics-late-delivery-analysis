"""
Week 4 - Predictive Modeling and Optimization in Logistics Systems
Yuva Intern | Logistics Data Analyst Intern

Predicts shipment delivery time (hours) from simulated operational data, then
uses the model to (1) shift dispatch times, (2) allocate vehicles to shipments
with a linear program, and (3) sequence a multi-stop route.

Run:  python logistics_delivery_time_model.py
Outputs: logistics_week4_dataset.csv, results.json, figures/*.png
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linprog

from sklearn.model_selection import train_test_split, KFold, cross_validate, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

SEED = 42
rng = np.random.default_rng(SEED)
RESULTS = {}

# --------------------------------------------------------------------------
# 1. DATA SIMULATION
# --------------------------------------------------------------------------
def traffic_profile(hour):
    """Average city traffic index (1-10) by hour: morning and evening peaks."""
    return 2 + 5 * np.exp(-((hour - 9) ** 2) / (2 * 1.5 ** 2)) \
             + 6 * np.exp(-((hour - 18.5) ** 2) / (2 * 1.8 ** 2))

SPEED = {"bike": 22, "van": 42, "truck": 34}          # km/h free-flow
WEATHER_MULT = {"clear": 1.00, "rain": 1.15, "storm": 1.45}

def simulate(n=5000):
    distance = np.clip(rng.lognormal(3.3, 0.8, n), 3, 400)             # km
    weight = np.clip(rng.gamma(2.0, 14.0, n), 0.5, 180)                # kg
    hour = rng.integers(6, 22, n)
    dow = rng.integers(0, 7, n)
    weekend = (dow >= 5).astype(int)
    traffic = traffic_profile(hour) * np.where(weekend == 1, 0.8, 1.0) + rng.normal(0, 0.8, n)
    traffic = np.clip(traffic, 1, 10)
    weather = rng.choice(["clear", "rain", "storm"], n, p=[0.70, 0.22, 0.08])
    stops = np.clip(rng.poisson(4, n) + 1, 1, 12)
    load = np.clip(rng.normal(70, 15, n), 30, 100)                     # warehouse load %
    exp = np.clip(rng.gamma(2.5, 2.0, n), 0, 15)                       # driver years

    # Vehicle assigned by current rule, with 15% deviations (real-world noise)
    vehicle = np.where((weight <= 15) & (distance <= 50), "bike",
               np.where(weight > 70, "truck", "van")).astype(object)
    flip = rng.random(n) < 0.15
    vehicle[flip & (vehicle == "bike")] = "van"
    vehicle[flip & (vehicle == "van")] = rng.choice(["van", "truck"], (flip & (vehicle == "van")).sum())

    df = pd.DataFrame(dict(distance_km=distance, weight_kg=weight, hour_of_day=hour,
        day_of_week=dow, traffic_index=traffic, weather=weather, vehicle_type=vehicle,
        num_stops=stops, warehouse_load_pct=load, driver_exp_years=exp))
    df["delivery_time_hrs"] = true_time(df)
    return df

def true_time(df):
    """Ground-truth generating process (non-linear, with interactions) + noise."""
    speed = df.vehicle_type.map(SPEED)
    wmult = df.weather.map(WEATHER_MULT)
    travel = df.distance_km / speed * (1 + 0.07 * df.traffic_index) * wmult
    travel = travel * (1 - 0.015 * df.driver_exp_years.clip(upper=10))
    handling = (0.15 + 0.012 * df.weight_kg) * (1 + df.warehouse_load_pct / 150)
    stop_time = df.num_stops * np.where(df.vehicle_type == "truck", 0.30, 0.22)
    total = travel + handling + stop_time
    return total * rng.lognormal(0, 0.08, len(df)) if "delivery_time_hrs" not in df else total

df = simulate()
df.to_csv("logistics_week4_dataset.csv", index=False)
RESULTS["n_rows"] = len(df)
RESULTS["describe"] = df.describe().round(2).to_dict()
RESULTS["weather_counts"] = df.weather.value_counts().to_dict()
RESULTS["vehicle_counts"] = df.vehicle_type.value_counts().to_dict()
RESULTS["missing"] = int(df.isna().sum().sum())

# --------------------------------------------------------------------------
# 2. FEATURE ENGINEERING + PREPROCESSING
# --------------------------------------------------------------------------
def add_features(d):
    d = d.copy()
    d["is_peak"] = (d.hour_of_day.between(8, 10) | d.hour_of_day.between(17, 20)).astype(int)
    d["is_weekend"] = (d.day_of_week >= 5).astype(int)
    return d

FEATURES_NUM = ["distance_km", "weight_kg", "hour_of_day", "traffic_index", "num_stops",
                "warehouse_load_pct", "driver_exp_years", "is_peak", "is_weekend"]
FEATURES_CAT = ["weather", "vehicle_type"]
TARGET = "delivery_time_hrs"

X = add_features(df)[FEATURES_NUM + FEATURES_CAT]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED)

def make_pre(scale):
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([("num", num, FEATURES_NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT)])

# --------------------------------------------------------------------------
# 3. MODEL COMPARISON WITH 5-FOLD CROSS-VALIDATION
# --------------------------------------------------------------------------
models = {
    "Linear Regression": Pipeline([("pre", make_pre(True)), ("m", LinearRegression())]),
    "Ridge (alpha=1)": Pipeline([("pre", make_pre(True)), ("m", Ridge(alpha=1.0))]),
    "Decision Tree": Pipeline([("pre", make_pre(False)), ("m", DecisionTreeRegressor(max_depth=8, random_state=SEED))]),
    "Random Forest": Pipeline([("pre", make_pre(False)), ("m", RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1))]),
    "Gradient Boosting": Pipeline([("pre", make_pre(False)), ("m", GradientBoostingRegressor(random_state=SEED))]),
}
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
cv_rows = []
for name, pipe in models.items():
    cv = cross_validate(pipe, X_train, y_train, cv=kf,
                        scoring=("neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"))
    cv_rows.append(dict(model=name,
        cv_rmse=-cv["test_neg_root_mean_squared_error"].mean(),
        cv_rmse_std=cv["test_neg_root_mean_squared_error"].std(),
        cv_mae=-cv["test_neg_mean_absolute_error"].mean(),
        cv_r2=cv["test_r2"].mean()))
cv_df = pd.DataFrame(cv_rows).round(3)
RESULTS["cv"] = cv_df.to_dict("records")
print(cv_df.to_string(index=False))

# --------------------------------------------------------------------------
# 4. HYPERPARAMETER TUNING (RandomizedSearchCV on Gradient Boosting)
# --------------------------------------------------------------------------
param_dist = {"m__n_estimators": [200, 300, 400, 600],
              "m__learning_rate": [0.03, 0.05, 0.08, 0.1],
              "m__max_depth": [2, 3, 4, 5],
              "m__subsample": [0.7, 0.85, 1.0],
              "m__min_samples_leaf": [1, 5, 10, 20]}
search = RandomizedSearchCV(models["Gradient Boosting"], param_dist, n_iter=25, cv=kf,
                            scoring="neg_root_mean_squared_error", random_state=SEED, n_jobs=-1)
search.fit(X_train, y_train)
best = search.best_estimator_
RESULTS["best_params"] = {k.replace("m__", ""): v for k, v in search.best_params_.items()}
RESULTS["best_cv_rmse"] = round(-search.best_score_, 3)
print("Best params:", RESULTS["best_params"], "CV RMSE:", RESULTS["best_cv_rmse"])

# --------------------------------------------------------------------------
# 5. FINAL TEST-SET EVALUATION (all models refit on full training set)
# --------------------------------------------------------------------------
test_rows = []
fitted = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    fitted[name] = pipe
    p = pipe.predict(X_test)
    test_rows.append(dict(model=name, rmse=root_mean_squared_error(y_test, p),
                          mae=mean_absolute_error(y_test, p), r2=r2_score(y_test, p)))
p = best.predict(X_test)
test_rows.append(dict(model="Gradient Boosting (tuned)", rmse=root_mean_squared_error(y_test, p),
                      mae=mean_absolute_error(y_test, p), r2=r2_score(y_test, p)))
test_df = pd.DataFrame(test_rows).round(3)
RESULTS["test"] = test_df.to_dict("records")
print(test_df.to_string(index=False))
pred_test = best.predict(X_test)
resid = y_test.values - pred_test
RESULTS["resid_mean"] = round(float(resid.mean()), 4)
RESULTS["resid_std"] = round(float(resid.std()), 3)
RESULTS["within_1hr_pct"] = round(float((np.abs(resid) <= 1).mean() * 100), 1)
RESULTS["mape_pct"] = round(float((np.abs(resid) / y_test.values).mean() * 100), 1)
RESULTS["target_mean"] = round(float(y.mean()), 2)

# Permutation importance on the tuned model
pi = permutation_importance(best, X_test, y_test, n_repeats=10, random_state=SEED, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=X_test.columns).sort_values(ascending=False)
RESULTS["importance"] = imp.round(4).to_dict()

# --------------------------------------------------------------------------
# 6. OPTIMIZATION
# --------------------------------------------------------------------------
# 6a. Dispatch-time shifting (+/-2 hours) for the test shipments
def predict_at_hour(d, new_hour):
    d2 = d.copy()
    shift = traffic_profile(new_hour) * np.where(d2.is_weekend == 1, 0.8, 1.0) \
          - traffic_profile(d.hour_of_day.values) * np.where(d2.is_weekend == 1, 0.8, 1.0)
    d2["traffic_index"] = np.clip(d2.traffic_index + shift, 1, 10)
    d2["hour_of_day"] = new_hour
    d2["is_peak"] = (pd.Series(new_hour, index=d2.index).between(8, 10) |
                     pd.Series(new_hour, index=d2.index).between(17, 20)).astype(int)
    return best.predict(d2)

base_pred = best.predict(X_test)
cands = []
for s in (-2, -1, 0, 1, 2):
    nh = np.clip(X_test.hour_of_day.values + s, 6, 21)
    cands.append(predict_at_hour(X_test, nh))
cands = np.array(cands)
shifted = cands.min(axis=0)
peak_mask = X_test.is_peak.values == 1
RESULTS["shift"] = dict(
    mean_before=round(float(base_pred.mean()), 3), mean_after=round(float(shifted.mean()), 3),
    pct_all=round(float((1 - shifted.mean() / base_pred.mean()) * 100), 1),
    peak_before=round(float(base_pred[peak_mask].mean()), 3),
    peak_after=round(float(shifted[peak_mask].mean()), 3),
    pct_peak=round(float((1 - shifted[peak_mask].mean() / base_pred[peak_mask].mean()) * 100), 1),
    n_peak=int(peak_mask.sum()), n_total=len(base_pred),
    n_shifted=int((cands.argmin(axis=0) != 2).sum()))

# 6b. Vehicle allocation as a linear program (one batch of 150 shipments)
VEH = ["bike", "van", "truck"]
HOURLY = {"bike": 150, "van": 350, "truck": 600}     # INR per hour
PERKM = {"bike": 3, "van": 12, "truck": 22}          # INR per km
LATE_PENALTY = 400                                   # INR per hour beyond SLA
batch = X_test.sample(150, random_state=SEED).copy()
batch["sla_hrs"] = 3.0 + batch.distance_km / 30.0

def eligible(row, v):
    if v == "bike":  return row.weight_kg <= 15 and row.distance_km <= 50
    if v == "van":   return row.weight_kg <= 120
    return True

cost = np.full((len(batch), 3), 1e9)
pred_time = np.zeros((len(batch), 3))
for j, v in enumerate(VEH):
    b2 = batch[FEATURES_NUM + FEATURES_CAT].copy()
    b2["vehicle_type"] = v
    pt = best.predict(b2)
    pred_time[:, j] = pt
    for i, (_, row) in enumerate(batch.iterrows()):
        if eligible(row, v):
            late = max(0.0, pt[i] - row.sla_hrs)
            cost[i, j] = pt[i] * HOURLY[v] + row.distance_km * PERKM[v] + late * LATE_PENALTY

# Current rule (baseline): weight/distance thresholds
rule = np.where((batch.weight_kg <= 15) & (batch.distance_km <= 50), 0,
        np.where(batch.weight_kg > 70, 2, 1))
base_counts = np.bincount(rule, minlength=3)
capacity = np.ceil(base_counts * 1.25).astype(int)    # fleet sized for rule + 25% buffer
n, m = cost.shape
c = cost.flatten()
A_eq = np.zeros((n, n * m)); 
for i in range(n): A_eq[i, i * m:(i + 1) * m] = 1
A_ub = np.zeros((m, n * m))
for j in range(m): A_ub[j, j::m] = 1
res = linprog(c, A_ub=A_ub, b_ub=capacity, A_eq=A_eq, b_eq=np.ones(n), bounds=(0, 1), method="highs")
assign = res.x.reshape(n, m).argmax(axis=1)
def totals(a):
    idx = np.arange(n)
    t = pred_time[idx, a]
    late = np.maximum(0, t - batch.sla_hrs.values)
    return dict(cost=float(cost[idx, a].sum()), hours=float(t.sum()),
                late=int((late > 0).sum()), counts=np.bincount(a, minlength=3).tolist())
tb, to = totals(rule), totals(assign)
RESULTS["alloc"] = dict(
    capacity=capacity.tolist(), baseline=tb, optimized=to,
    cost_saving_pct=round((1 - to["cost"] / tb["cost"]) * 100, 1),
    cost_saving_inr=round(tb["cost"] - to["cost"]),
    hours_saving_pct=round((1 - to["hours"] / tb["hours"]) * 100, 1),
    changed=int((assign != rule).sum()))

# 6c. Route sequencing for one van (15 stops): nearest neighbour + 2-opt
r_rng = np.random.default_rng(7)
pts = np.vstack([[0, 0], r_rng.uniform(0, 30, (15, 2))])        # depot at origin, km grid
D = np.linalg.norm(pts[:, None] - pts[None], axis=2)
def rlen(order): return sum(D[order[i], order[i + 1]] for i in range(len(order) - 1))
naive = list(range(16)) + [0]
nn, left = [0], set(range(1, 16))
while left:
    k = min(left, key=lambda q: D[nn[-1], q]); nn.append(k); left.remove(k)
nn.append(0)
best_r, improved = nn[:], True
while improved:
    improved = False
    for i in range(1, len(best_r) - 2):
        for k in range(i + 1, len(best_r) - 1):
            cand = best_r[:i] + best_r[i:k + 1][::-1] + best_r[k + 1:]
            if rlen(cand) < rlen(best_r) - 1e-9:
                best_r, improved = cand, True
RESULTS["route"] = dict(naive_km=round(rlen(naive), 1), nn_km=round(rlen(nn), 1),
                        opt_km=round(rlen(best_r), 1),
                        saving_pct=round((1 - rlen(best_r) / rlen(naive)) * 100, 1))

# --------------------------------------------------------------------------
# 7. FIGURES
# --------------------------------------------------------------------------
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
NAVY, ORANGE, GREY = "#1F3A5F", "#E8630A", "#8A94A6"

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].hist(df[TARGET], bins=50, color=NAVY); ax[0].set_xlabel("Delivery time (hours)"); ax[0].set_ylabel("Shipments"); ax[0].set_title("Distribution of target variable")
df.boxplot(column=TARGET, by="hour_of_day", ax=ax[1], grid=False, showfliers=False, boxprops=dict(color=NAVY), medianprops=dict(color=ORANGE))
ax[1].set_title("Delivery time by dispatch hour"); ax[1].set_xlabel("Hour of day"); ax[1].set_ylabel("Hours"); plt.suptitle("")
plt.tight_layout(); plt.savefig("figures/fig1_eda.png", dpi=170); plt.close()

fig, ax = plt.subplots(figsize=(6.4, 5))
cols = ["distance_km", "weight_kg", "traffic_index", "num_stops", "warehouse_load_pct", "driver_exp_years", TARGET]
corr = df[cols].corr()
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right")
ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
for i in range(len(cols)):
    for j in range(len(cols)): ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
plt.colorbar(im, fraction=0.046); ax.set_title("Correlation matrix")
plt.tight_layout(); plt.savefig("figures/fig2_corr.png", dpi=170); plt.close()

fig, ax = plt.subplots(figsize=(7.5, 3.8))
names = test_df.model.tolist(); x = np.arange(len(names))
ax.bar(x - 0.2, test_df.rmse, 0.4, label="RMSE", color=NAVY)
ax.bar(x + 0.2, test_df.mae, 0.4, label="MAE", color=ORANGE)
ax.set_xticks(x); ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=8)
ax.set_ylabel("Error (hours)"); ax.set_title("Test-set error by model (lower is better)"); ax.legend()
plt.tight_layout(); plt.savefig("figures/fig3_models.png", dpi=170); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].scatter(y_test, pred_test, s=6, alpha=0.4, color=NAVY)
lim = [0, max(y_test.max(), pred_test.max())]; ax[0].plot(lim, lim, color=ORANGE, lw=1.5)
ax[0].set_xlabel("Actual (hours)"); ax[0].set_ylabel("Predicted (hours)"); ax[0].set_title("Predicted vs actual (tuned GB)")
ax[1].scatter(pred_test, resid, s=6, alpha=0.4, color=NAVY); ax[1].axhline(0, color=ORANGE)
ax[1].set_xlabel("Predicted (hours)"); ax[1].set_ylabel("Residual (hours)"); ax[1].set_title("Residuals")
plt.tight_layout(); plt.savefig("figures/fig4_fit.png", dpi=170); plt.close()

fig, ax = plt.subplots(figsize=(7, 3.8))
imp.sort_values().plot.barh(ax=ax, color=NAVY); ax.set_xlabel("Drop in R² when feature is shuffled"); ax.set_title("Permutation importance (test set)")
plt.tight_layout(); plt.savefig("figures/fig5_importance.png", dpi=170); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].bar(["Current rule", "LP-optimised"], [tb["cost"] / 1000, to["cost"] / 1000], color=[GREY, ORANGE])
ax[0].set_ylabel("Total cost (₹ thousand)"); ax[0].set_title("Vehicle allocation: 150-shipment batch")
ax[1].bar(["As dispatched", "Best ±2 h slot"], [RESULTS["shift"]["peak_before"], RESULTS["shift"]["peak_after"]], color=[GREY, ORANGE])
ax[1].set_ylabel("Mean predicted time (h)"); ax[1].set_title("Peak-hour shipments: dispatch shifting")
plt.tight_layout(); plt.savefig("figures/fig6_optim.png", dpi=170); plt.close()

with open("results.json", "w") as f: json.dump(RESULTS, f, indent=2, default=float)
print(json.dumps({k: RESULTS[k] for k in ["shift", "alloc", "route", "importance", "resid_std", "within_1hr_pct", "mape_pct"]}, indent=1, default=float))
