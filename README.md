# FinanceRAG-DataBot

FinanceRAG-DataBot is a **local, privacy-preserving financial analysis assistant** that lets you query, analyze, and visualize financial data — **entirely on your machine**.  
No external APIs, cloud services, or hosted models are used, ensuring your financial data remains secure.

---

## Demo : https://drive.google.com/file/d/1RLTjk4FheiNYqmQEejk4v9js2bbRGQXh/view?usp=sharing

## 🚀 Key Features

- **Private & Offline**  
  Runs entirely on your local system. All data stays on your machine, guaranteeing full control and compliance with sensitive data policies.

- **Financial Data Querying**  
  Interactively explore structured CSV data, such as:
  - Company quarterly revenue by product category
  - Expense breakdowns
  - Key financial ratios
  

- **Dynamic Plotting**  
  Generate publication-quality plots on demand from natural language instructions, including:
  - 📊 **Bar charts** — e.g., quarterly iPhone revenue
  - 📈 **Line charts** — e.g., trend of Services revenue over time
  - 🥧 **Pie charts** — e.g., product contribution share for a specific quarter
  - 📉 **Comparative charts** — e.g., Mac vs. iPad sales

- **Schema Awareness**  
  Automatically detects and uses dataset column names and types for accurate plotting.

- **No Internet Required**  
  All computations, plotting, and data retrieval are handled with **local Python libraries**.

---

## 🛠️ How It Works

1. **Data Loading** — CSV files for selected tickers and table types are loaded from local folders.  
2. **Natural Language Commands** — You describe the analysis or chart you want.  
3. **Secure Execution** — The system executes plotting/analysis code **in a sandboxed environment** using the in-memory `df`.  
4. **Instant Visual Output** — Figures are generated using Matplotlib and displayed in the local UI.

---

## 📌 Example Use Cases

- **Quarterly Performance Tracking**  
  > *"Show a bar chart of iPhone revenue for the last 8 quarters."*

- **Category Share Analysis**  
  > *"Plot a pie chart of product revenue for Q4 2023."*

- **Trend Analysis**  
  > *"Line chart of Services revenue from 2020 to 2024."*

- **Comparative Analysis**  
  > *"Plot Mac and iPad revenues together to compare growth rates."*

---

## 📊 Example Plots

| Chart Type | Description |
|------------|-------------|
| **Bar Chart** | Quarterly product revenues |
| **Line Chart** | Trend analysis over time |
| **Pie Chart** | Product category distribution in a given quarter |
| **Stacked Bar** | Multiple products' revenue in a single chart |

---


## 🔮 Future Work

- **Predictive Analytics**  
  Use **scikit-learn**, **XGBoost**, or other ML libraries to forecast:
  - Future quarterly revenue
  - Product category growth trends
  - Market share changes

- **Data Pre-Processing Pipeline**  
  Automate:
  - Cleaning raw CSVs
  - Handling missing values
  - Normalizing columns for machine learning

- **Advanced Visualizations**  
  - Interactive charts with Plotly/Bokeh
  - Correlation heatmaps
  - Multi-axis comparison plots

- **Extended Data Sources**  
  Support more financial statements (Balance Sheet, Cash Flow) — still processed fully offline.

---

## 🖥️ Local Privacy Advantage

Unlike cloud-based finance tools, FinanceRAG-DataBot **never** sends your data to an external server or model.  
This makes it suitable for:
- Corporate environments with strict compliance rules
- Financial analysts handling confidential client data
- Personal use without privacy trade-offs

---
