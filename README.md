## 💻 Repository Contents & Data Pipeline

The codebase is structured as a modular data pipeline, separating data ingestion, feature engineering, and predictive modeling into distinct components for scalability and maintainability.

### 1. Data Ingestion & Extraction 
* **`oil_data_10y.py` & `oil_price_10y.py`**: Automated scripts designed to extract, parse, and validate a decade of historical WTI crude oil data.
* **`rubber_price.py` & `get_price.py`**: Web scraping and API integration scripts for continuous ingestion of natural rubber pricing metrics.

### 2. Feature Engineering & Processing
* **`compare price & stock.py`**: Core feature engineering script. Cross-references historical price actions with inventory/stock deviations to identify fundamental market squeezes.
* **Processed Datasets** (`master_oil_rubber.csv`, `rubber_cleaned.csv`, `final_research_data.csv`): Aggregated and cleaned datasets ready for machine learning ingestion.

### 3. Predictive Modeling (Core Engine)
* **`oil_prediction_comments_english.ipynb`**: The central machine learning notebook. Contains comprehensive Exploratory Data Analysis (EDA), model training, hyperparameter tuning, backtesting, and forward-looking scenario generation.

### 4. Visualization & Reporting
* **`chart for oil.py`**: Generates production-ready financial charts, mapping price movements against major macroeconomic and geopolitical events.
* **`Macro_Comovement_Chart.png`**: Rendered visual outputs demonstrating the asymmetric shocks of global events on commodity prices.

---
*Note: The virtual environment directory (`thesis_env`) has been omitted from this repository to adhere to deployment best practices.*
