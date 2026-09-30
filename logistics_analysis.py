"""Late-delivery analysis on the DataCo Smart Supply Chain dataset.
Place DataCoSupplyChainDataset.csv in the data/ folder before running.
Column names should be checked against the downloaded file."""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load
df = pd.read_csv("data/DataCoSupplyChainDataset.csv", encoding="latin-1")
print(df.shape)

# 2. Clean
df = df.drop_duplicates()
drop_cols = ["Customer Email", "Customer Password", "Customer Fname",
             "Customer Lname", "Product Image", "Product Description"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])
df["order_date"] = pd.to_datetime(df["order date (DateOrders)"])
df["order_month"] = df["order_date"].dt.month
df["order_weekday"] = df["order_date"].dt.dayofweek

# 3. KPIs and EDA
late_rate = df["Late_delivery_risk"].mean() * 100
avg_delay = (df["Days for shipping (real)"] - df["Days for shipment (scheduled)"]).mean()
print(f"Late delivery rate: {late_rate:.1f}%  |  Average delay: {avg_delay:.2f} days")
mode_late = df.groupby("Shipping Mode")["Late_delivery_risk"].mean().sort_values(ascending=False) * 100
mode_late.plot(kind="bar", title="Late Delivery Rate by Shipping Mode (%)")
plt.tight_layout(); plt.savefig("late_rate_by_mode.png"); plt.close()

# 4. Features (no leakage columns)
features = ["Shipping Mode", "Order Region", "Market", "Category Name",
            "Customer Segment", "Order Item Quantity", "Sales",
            "Order Item Discount Rate", "Days for shipment (scheduled)",
            "order_month", "order_weekday"]
X = pd.get_dummies(df[features], drop_first=True)
y = df["Late_delivery_risk"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Model
model = RandomForestClassifier(n_estimators=200, max_depth=12,
                               class_weight="balanced", random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))
print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
print(pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10))

# 6. Region clustering
region_stats = df.groupby("Order Region").agg(
    late_rate=("Late_delivery_risk", "mean"),
    avg_sales=("Sales", "mean"),
    avg_profit=("Order Profit Per Order", "mean"),
    orders=("Late_delivery_risk", "count"))
region_stats["cluster"] = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(
    StandardScaler().fit_transform(region_stats))
print(region_stats.sort_values("cluster"))
