# 🏡 HomeLens — House Price Dashboard

A modern Streamlit dashboard for exploring California housing data, comparing four regression models and predicting house prices.

## Files
- `app.py` — Streamlit dashboard
- `house_prices_cleaned.csv` — dataset required by the app
- `requirements.txt` — Python dependencies
- `.streamlit/config.toml` — theme (cream, teal and terracotta)

## Use your own data
Upload **any CSV** from the sidebar (**Your data**). The app adapts to what it finds:

- **Reading:** comma, semicolon or tab separated files, UTF-8 or Latin-1 text. Values like `$1,200` or `15%` become numbers, and date columns are detected.
- **Column types:** every column is classified as numeric, categorical, date or free text.
- **Target:** the app picks a column to predict (you can change it, or choose *No target* to only explore). A numeric target runs **regression**; a category such as species, status or survived runs **classification**.
- **Features:** chosen automatically (IDs, dates, free text and columns that leak the answer are skipped). Add or remove features in the sidebar.
- **Cleaning:** missing values are filled, text columns are one-hot encoded, and four models are trained.
- **No suitable target?** You still get the full data profile and explorer.

Every page adapts, including the predictor form. The map appears only if the file has latitude and longitude columns.

## Pages
- **Data profile** (always): row and column counts, column types, missing values, per-column charts, correlation heatmap
- **Overview**: KPIs and charts for the chosen target (distribution and correlations, or class balance and feature comparison)
- **Model performance**: leaderboard, actual vs predicted or confusion matrix, feature importance
- **Predictor**: enter feature values and get a prediction (with class probabilities for classification)
- **Statistics**: t-test, Pearson, Spearman, ANOVA, chi-square, confidence interval and Central Limit Theorem
- **Explore data**: filter, view and download

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Upload every file in this folder (including the `.streamlit` folder) to a GitHub repository.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Create an app from your repository and set the main file to `app.py`.
4. Deploy.

The bundled sample (California housing) uses `Price` and these features: `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`. Your own file can use any columns.
