# Week 4 – Predictive Modeling and Optimization in Logistics Systems

Yuva Intern – Logistics Data Analyst Intern.

Predicts shipment delivery time (hours) on a simulated 5,000-shipment dataset and uses the model for dispatch-time shifting, LP-based vehicle allocation and route sequencing.

## Run
```
pip install -r requirements.txt
python logistics_delivery_time_model.py
```
Outputs: `logistics_week4_dataset.csv`, `results.json`, `figures/*.png` (seed = 42).

## Models
Linear Regression, Ridge, Decision Tree, Random Forest, Gradient Boosting (tuned with RandomizedSearchCV, 5-fold CV).

## Result (held-out test set)
Tuned Gradient Boosting: RMSE 0.335 h, MAE 0.230 h, R² 0.940.
