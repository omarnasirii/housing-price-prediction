import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV

# Streamlit page configuration
st.set_page_config(page_title="Housing Market Prediction", layout="wide")

# Helper function to validate and load data
def load_data(uploaded_file, parse_dates=False, index_col=None):
    try:
        return pd.read_csv(uploaded_file, parse_dates=parse_dates, index_col=index_col)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# Sidebar for file uploads
st.sidebar.header("Upload Data Files")
uploaded_fed_files = [
    st.sidebar.file_uploader(f"Upload {name}", type=["csv"]) for name in ["Interest Rates", "Vacancy Rates", "CPI Data"]
]
uploaded_zillow_files = [
    st.sidebar.file_uploader(f"Upload {name}", type=["csv"]) for name in ["Median Sale Price", "Home Value Index"]
]

# Load and process data if files are provided
if all(uploaded_fed_files) and all(uploaded_zillow_files):
    st.sidebar.success("All required files uploaded. Processing data...")

    # Load Federal Reserve data
    fed_data_frames = [
        load_data(f, parse_dates=True, index_col=0) for f in uploaded_fed_files
    ]
    if any(df is None for df in fed_data_frames):
        st.error("Please check the Federal Reserve files and re-upload them.")
        st.stop()
    fed_data = pd.concat(fed_data_frames, axis=1).ffill().dropna()

    # Load Zillow data
    zillow_data_frames = [load_data(f) for f in uploaded_zillow_files]
    if any(df is None for df in zillow_data_frames):
        st.error("Please check the Zillow files and re-upload them.")
        st.stop()

    # Process Zillow data
    try:
        zillow_data_frames = [pd.DataFrame(df.iloc[0, 5:]) for df in zillow_data_frames]
        for df in zillow_data_frames:
            df.index = pd.to_datetime(df.index)
            df["month"] = df.index.to_period("M")
        price_data = zillow_data_frames[0].merge(zillow_data_frames[1], on="month")
        price_data.index = zillow_data_frames[0].index
        del price_data["month"]
        price_data.columns = ["price", "value"]
    except Exception as e:
        st.error(f"Error processing Zillow data: {e}")
        st.stop()

    # Merge datasets and prepare features
    fed_data.index += timedelta(days=2)
    price_data = fed_data.merge(price_data, left_index=True, right_index=True)
    price_data.columns = ["interest", "vacancy", "cpi", "price", "value"]
    price_data["adj_price"] = price_data["price"] / price_data["cpi"] * 100
    price_data["adj_value"] = price_data["value"] / price_data["cpi"] * 100
    price_data["next_quarter"] = price_data["adj_price"].shift(-13)
    price_data.dropna(inplace=True)
    price_data["change"] = (price_data["next_quarter"] > price_data["adj_price"]).astype(int)

    # Display processed data
    st.subheader("Processed Data")
    st.dataframe(price_data)

    # Define predictors and target
    predictors = ["interest", "vacancy", "adj_price", "adj_value"]
    target = "change"

    # Backtesting function with yearly evaluation
    def predict(train, test, predictors, target):
        rf = RandomForestClassifier(min_samples_split=10, random_state=1)
        rf.fit(train[predictors], train[target])
        return rf.predict(test[predictors])

    def backtest(data, predictors, target, start=260, step=52):
        all_preds = []
        for i in range(start, data.shape[0], step):
            train = data.iloc[:i]
            test = data.iloc[i:(i + step)]
            all_preds.append(predict(train, test, predictors, target))
        preds = np.concatenate(all_preds)
        return preds, accuracy_score(data.iloc[start:][target], preds)

    # Run initial backtest
    preds, accuracy = backtest(price_data, predictors, target)
    st.subheader(f"Backtest Accuracy: {accuracy:.2f}")

    # Add yearly rolling averages and moving averages
    yearly = price_data.rolling(52, min_periods=1).mean()  # 1-year (52 weeks) moving average
    yearly_ratios = [p + "_year" for p in predictors]
    price_data[yearly_ratios] = price_data[predictors] / yearly[predictors]

    # Enhancing features with moving averages (longer-term trends)
    price_data["ma_3y_adj_price"] = price_data["adj_price"].rolling(156, min_periods=1).mean()
    price_data["ma_3y_adj_value"] = price_data["adj_value"].rolling(156, min_periods=1).mean()

    # Update predictors
    updated_predictors = predictors + yearly_ratios + ["ma_3y_adj_price", "ma_3y_adj_value"]

    # Hyperparameter tuning for RandomForest
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    rf = RandomForestClassifier(random_state=1)
    grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(price_data[updated_predictors], price_data[target])
    best_rf = grid_search.best_estimator_

    # Run updated backtest with yearly ratios
    preds, accuracy = backtest(price_data, updated_predictors, target)
    st.subheader(f"Updated Backtest Accuracy with Yearly Ratios and MA: {accuracy:.2f}")

    # Visualization
    plot_data = price_data.iloc[260:].copy()
    pred_match = (preds == plot_data[target])
    plot_data["color"] = ["green" if match else "red" for match in pred_match]

    st.subheader("Prediction Visualization")
    fig, ax = plt.subplots()
    plot_data.reset_index().plot.scatter(x="index", y="adj_price", c=plot_data["color"], ax=ax)
    st.pyplot(fig)

    # Feature importance using best RandomForest model
    result = permutation_importance(best_rf, price_data[updated_predictors], price_data[target], n_repeats=10, random_state=1)
    st.subheader("Feature Importances")
    st.bar_chart(pd.Series(result["importances_mean"], index=updated_predictors))

else:
    st.sidebar.warning("Please upload all required files.")
