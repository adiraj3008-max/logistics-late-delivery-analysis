# Logistics Late-Delivery Prediction and Shipping Optimisation

Week 1 project for the **Logistics Data Analyst Intern** program (Yuva Intern):
*Strategic Planning and Data Exploration in Logistics*.

## Overview
This project plans a data science solution for a global e-commerce supply chain that suffers from late deliveries. It defines the business scenario, selects KPIs, and provides a Python roadmap covering data cleaning, exploratory analysis, prediction of late orders, region clustering and shipping-mode optimisation.

## Dataset
DataCo Smart Supply Chain for Big Data Analysis (Kaggle):
https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis

Download `DataCoSupplyChainDataset.csv` and place it in the `data/` folder (not uploaded to this repo because of file size).

## KPIs
- On-Time Delivery Rate (%)
- Late Delivery Rate (%)
- Average Shipping Delay (days)
- Average Fulfilment Time (days)
- Profit per Order
- Model Recall / F1 for late orders

## Roadmap
1. Data collection
2. Data cleaning
3. Exploratory data analysis and KPI calculation
4. Feature engineering
5. Predictive modelling (Random Forest, regression)
6. Evaluation and K-Means segmentation of regions
7. Shipping-mode optimisation and reporting

## Project Structure
```
logistics-week1/
├── README.md
├── requirements.txt
├── data/                  # place the Kaggle CSV here
├── logistics_analysis.py  # end-to-end analysis script
└── Week1_Logistics_Strategic_Planning_Report.docx
```

## How to Run
```
pip install -r requirements.txt
python logistics_analysis.py
```

## Tools
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn.
