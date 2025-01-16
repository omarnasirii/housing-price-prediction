# Housing Market Prediction Model
This project is designed to predict housing price trends (whether they will go up or down) in the next quarter using machine learning. The model employs a Random Forest Classifier trained on economic indicators such as interest rates, vacancy rates, CPI (Consumer Price Index), and historical housing prices. It also leverages backtesting to evaluate the model’s accuracy over time.

## Table of Contents
Overview
Installation
Usage
Data Sources
Model Explanation
Contributing
License

### Overview
The Housing Market Prediction Model predicts whether housing prices will increase or decrease based on a variety of economic factors. It uses a Random Forest Classifier to make these predictions, which can help investors, real estate professionals, and economists forecast trends in the housing market.

### Key Features:
Backtesting: Evaluates model performance over time by training on past data and testing on unseen future data.
Random Forest Classifier: Classifies whether housing prices will rise or fall.
Feature Engineering: Incorporates economic indicators like interest rates, vacancy rates, and CPI data.
Yearly Ratio: Includes yearly rolling averages to capture long-term trends and seasonality.
Visualization: Displays prediction results using scatter plots.

### Installation
To run this project locally, follow the steps below.

#### Prerequisites
Python 3.7+ is required.
You should have Streamlit installed to run the web interface.

#### Steps to Set Up:

##### Clone the repository:
git clone https://github.com/yourusername/housing-market-prediction.git
cd housing-market-prediction

##### Create a virtual environment (if not already created):
python3 -m venv venv
Activate the virtual environment:

On macOS/Linux:
source venv/bin/activate

On Windows:
venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt
Upload the required data (see Data Sources below for instructions) into the data/ folder.

### Usage
Once you've set up the environment and added your data, you can run the application.

Run the Streamlit app:
streamlit run housing_model.py

The app will open in your default web browser. From there, you can:
 - Upload your CSV data files via the sidebar.
 - View processed data, prediction results, and backtest accuracy.
 - Visualize predictions with a scatter plot of predicted vs actual housing prices.

### Data Sources
To make predictions using the model, you will need to upload data files in CSV format. Here are suggested sources for obtaining the required datasets:

#### Interest Rates:
FRED: Interest Rates
Federal Reserve Economic Data

#### Vacancy Rates:
U.S. Census Bureau: Housing Vacancy Rates
Bureau of Labor Statistics - Vacancy Rates

#### CPI (Consumer Price Index):
Bureau of Labor Statistics: CPI Data

#### Median Sale Price:
Zillow Research Data

#### Home Value Index:
Zillow Home Value Index Data

You can download the data from these links, and then upload them into the data/ folder in your project directory. I also have sample data available for use.

### Model Explanation
The Random Forest Classifier is used to predict whether housing prices will rise or fall in the next quarter based on various economic indicators. Here's a brief explanation of the process:

Data Preprocessing:

The uploaded CSV data files are cleaned and merged into a single dataset.
Adjusted prices are calculated by dividing housing prices by the CPI data to normalize for inflation.
Feature Engineering:

Features used for prediction include interest rates, vacancy rates, CPI, and adjusted prices.
Yearly rolling averages for each feature are calculated to capture long-term trends and seasonality.
Model Training and Prediction:

A Random Forest Classifier is trained on the historical data to predict the future movement of housing prices (up or down).
The model is evaluated using backtesting to determine its accuracy over time.
Evaluation:

The accuracy of the model is calculated using accuracy score (percentage of correct predictions).
Feature importance is also visualized to understand which factors are most influential in making predictions.
Contributing
We welcome contributions to improve the model and the code. If you'd like to contribute, follow these steps:

### Steps to Contribute:
 - Fork the repository on GitHub.
 - Create a new branch for your feature (git checkout -b feature-name).
 - Commit your changes (git commit -am 'Add new feature').
 - Push your changes (git push origin feature-name).
 - Open a pull request.

### License
This project is licensed under the MIT License - see the LICENSE file for details.

