**# EE Telecom Sales Analytics**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://telecom-sales-analytics.streamlit.app/)

> While working at EE, I created this end-to-end data analytics and machine learning project for real operational data to improve efficiency and business strategies across 10 sales teams with over 120 advisors, which is reproduced here with synthetic data.

- **--**

**## Live Dashboard**

- ***[View the live dashboard here](**https://telecom-sales-analytics.streamlit.app/**)****

This project was built with Streamlit. You can interactively explore team performance, daily trends, target tracking, and ML-based team profiling.

- **--**

**## Project Overview**

This project summarises the full analytics workflow for a telecoms call centre with 8 sales teams across a 28-day reporting period. Each team sells broadband, mobile, and TV products, and performance is tracked against daily and monthly targets.

The analysis was originally conducted on real operational data at EE. The monthly dataset of the data from each team was created every month using KPI trackers I made in Excel. All data presented here is synthetically generated to reflect realistic call centre sales patterns while preserving no real employee or business information.

- ***Business questions:****
- Which teams are performing above or below target, and on which metrics?
- Do teams cluster into meaningful performance profiles based on how they sell?
- What is the relationship between call volume, conversion rate, and household value?
- **--**

**## Dashboard Pages**

- ***Overview****: Centre-wide KPIs, product mix by team, daily HH value heatmap
- ***Leaderboard****: Teams ranked by any metric with interactive selector
- ***Daily Trends****: Product sales and HH value over time, filterable by team
- ***Target Performance****: Heatmap of each team's variance vs targets across all metrics
- ***Team Deep Dive****: Per-team product trend lines, mix breakdown, and daily HH value
- ***ML Insights****: K-means clustering with scatter plot and cluster profile cards
- **--**

**## Machine Learning**

**### K-Means Clustering**

The teams were clustered by performance profiles rather than raw volume, using six features: broadband conversion rate, mobile conversion rate, TV conversion rate, average household value, products per household, and value per call.

The K was selected using a combination of the elbow method and silhouette scoring (k=3, silhouette=0.381), chosen for interpretability with 8 teams for the synthetic data.

Three clusters were identified:

- ***Efficient Converters****: highest value per call, strong across multiple conversion metrics
- ***Core Performers****: solid centre baseline, largest cluster
- ***High Bundlers****: exceptional products per household, maximise value per household interaction

**## Tech Stack**

- ***Data processing****: Python, Pandas, NumPy
- ***Visualisation****: Plotly, Matplotlib, Seaborn
- ***Machine learning****: Scikit-learn (KMeans, StandardScaler)
- ***Dashboard****: Streamlit
- ***Environment****: VSCode, Jupyter Notebooks
- **--**

**## Key Outcomes**

- This dashboard was updated using monthly data from Excel KPI trackers I designed for the sales centre, which improved efficiencies for 10 teams with over 120 advisors
- Each month, the sales manager had access to the updated dashboard to provide critical coaching insights for the team leaders and helped steer important focus areas for each team and business strategies going forward
- The three clusters were the most useful to show how each team sold products and what impact it had on the centre overall
- **--**

**## Data Notice**

All of the data in this repository is synthetically generated with the help of LLM Claude, using NumPy's random number generation with parameters based on realistic benchmarks. No real employee names, sales figures, or business data are present. The original analysis was conducted on real operational data held privately.

- **--**

**## About Me!**

- ***Isaac Samuel****

[Portfolio](https://isamuel.framer.website) · [LinkedIn](https://www.linkedin.com/in/isamuel/) · [GitHub](https://github.com/mushroomcodes)