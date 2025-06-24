# MajorProject
My project proves you can predict Bitcoin's price more accurately by including economic data like interest rates and recession risk. 
I tested several models and found that this extra data significantly improves the forecast, with a deep learning model called LSTM performing the best.

1. Setup and Imports 
• Install Packages: The code starts by installing necessary Python libraries like arch, 
pmdarima, pymannkendall, matplotlib, seaborn, tensorflow, etc. These libraries are 
used for time series analysis, statistical tests, data visualization, and building neural 
networks. 
• Import Libraries: Imports all of the libraries which installed above. 
2. Data Loading and Preparation 
• Data Loading: 
o Reads CSV files for various cryptocurrencies (BTC, ETH, SOL, HEX, XRP, DOT1, 
BNB, ADA, USDT, USDC, DOGE) from a specified directory 
("E:\MajorProject\Data\Dataset"). It extracts the "Adj Close" (Adjusted 
Closing Price) column from each CSV and assigns appropriate names. 
o Reads the "Date" column from "BTC-USD.csv" and converts it to datetime 
objects. 
• Data Combination: 
o Combines the adjusted closing prices of all cryptocurrencies into a single 
DataFrame called data. The date is set as the index. 
• Data Cleaning: 
o Displays information about the DataFrame (data.info()). 
o Checks for missing values (data.isnull().sum()). 
o Removes columns with a high number of missing values (more than 4). 
o Fills in any remaining missing values using forward fill (ffill) -- this propagates 
the last valid observation forward. 
3. Exploratory Data Analysis (EDA) 
• Descriptive Statistics: Provides a summary of the central tendency, dispersion and 
shape of a dataset’s distribution, excluding NaN values. 
• Visualizations: Creates line plots of each cryptocurrency's adjusted closing price over 
t
 ime. This helps to visualize trends and patterns. 
4. Trend Analysis and Stationarity Checks 
• Mann-Kendall Trend Test: Performs the Mann-Kendall test on the trend component 
of each time series to determine if there's a statistically significant monotonic trend 
(increasing or decreasing). 
• Autocorrelation Plots (ACF): Generates ACF plots for each cryptocurrency's price 
data. ACF plots show the correlation of a time series with its lagged values, helping to 
identify potential seasonality or autoregressive components. 
• Unit Root Tests (ADF and KPSS): Performs Augmented Dickey-Fuller (ADF) and 
Kwiatkowski-Phillips-Schmidt-Shin (KPSS) tests to check for stationarity in the time 
series. Stationarity is an important assumption for many time series models. The ADF 
test checks for the presence of a unit root (non-stationarity), while the KPSS test 
checks for trend stationarity. 
5. Vector Autoregression (VAR) Modeling 
• Subset Selection: Creates a subset of the data (data2) containing only "DOGE," 
"ETH," and "BTC" prices. 
• Stationarity Verification: Re-runs ADF tests on the selected cryptocurrencies. 
• Differencing: Applies differencing to make the time series stationary (removes 
trends). Then, performs ADF tests again on the differenced data. 
• VAR Model Fitting: 
o Creates a VAR model using statsmodels. 
o Selects the optimal lag order for the VAR model using information criteria 
(AIC, BIC, FPE, HQIC). 
• Cointegration Test: Performs the Johansen cointegration test to determine if there 
are long-run equilibrium relationships between the selected cryptocurrencies. 
• Impulse Response Analysis: Conducts impulse response analysis to examine the 
effects of shocks to one cryptocurrency's price on the other cryptocurrencies. 
6. Univariate Time Series Forecasting (BTC) 
• Data Splitting: 
o Splits the data into "front" (everything except the last 30 days) and "end" (last 
30 days) sets. 
o Further splits the "front" data into training and testing sets. 
• Log Transformation: Applies a logarithmic transformation (using boxcox with 
lambda=0, which is equivalent to a natural logarithm) to the "DOGE," "ETH," and 
"BTC" prices. Log transformations can help stabilize the variance and make the data 
more normally distributed. 
• Visualization: Plots the original BTC price and the log-transformed BTC price for the 
training set. 
• Time Series Cross-Validation: Uses TimeSeriesSplit to create multiple train-test splits 
for cross-validation. 
• Model Evaluation Function: Defines a function error to calculate Mean Squared Error 
(MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE). 
• Model Training and Evaluation (Cross-Validation): 
o Holt's Linear Exponential Smoothing: Trains a Holt's model using cross
validation and calculates MSE, RMSE, and MAE. 
o Holt-Winters' Seasonal Exponential Smoothing: Trains a Holt-Winters' model 
using cross-validation and calculates MSE, RMSE, and MAE. 
o ARIMA: Trains an ARIMA model using auto_arima (which automatically finds 
the best ARIMA parameters) and calculates MSE, RMSE, and MAE. 
o SARIMA: Trains a SARIMA model (Seasonal ARIMA) using auto_arima with a 
specified seasonal period (m=30) and calculates MSE, RMSE, and MAE. 
o ARIMAX: Trains ARIMAX models (ARIMA with exogenous variables) using 
"DOGE" and "ETH" prices as independent variables. Calculates MSE, RMSE, 
and MAE. 
o SARIMAX: Trains SARIMAX models (Seasonal ARIMAX) using "DOGE" and 
"ETH" prices as independent variables and a seasonal period of 30. Calculates 
MSE, RMSE, and MAE. 
• Model Comparison: Compares the performance of all the models based on their 
RMSE values and presents the results in a DataFrame. 
• AIC Tuning: The code then proceeds to test different seasonal period values to see 
what works best. 
• Summary of best Models: Model 1 is non seasonal ARIMAX Model and Model2 is 
seasonal Arimax model and their summaries printed. 
• Residual Analysis: Residual analysis is done to look to check for white noise. 
• Plots: Plots are provided to visualize the result. 
• Model Analysis Function: Defines error 2 for model analysis. 
7. LSTM Neural Network Modeling 
• Data Scaling: Scales the "BTC" price data using MinMaxScaler to a range between 0 
and 1. This is often beneficial for neural networks. 
• Train-Test Split: Splits the scaled data into training and testing sets. 
• Time Series Data Preparation: Defines a function ts to create time series data 
suitable for LSTM models. This function takes the data and a timestep parameter and 
creates sequences of data that the LSTM can use to learn from past values. 
• LSTM Model Training and Evaluation: 
o Lag Selection: A loop iterates through different lag values (5, 10, 15, 20) to 
f
 ind the best lag for the LSTM model. For each lag: 
▪ Prepares the data using the ts function. 
▪ Reshapes the data for the LSTM input. 
▪ Creates an LSTM model with two LSTM layers, dropout layers, and a 
dense output layer. 
▪ Compiles the model with the Adam optimizer and mean squared error 
loss. 
▪ Trains the model for 100 epochs. 
▪ Makes predictions on the training and testing sets. 
▪ Inverse transforms the predictions and actual values back to the 
original scale. 
▪ Calculates the RMSE for the training and testing sets. 
▪ Stores the results in a DataFrame. 
o LSTM Model Implementation: The LSTM model is then implemented. 
o Predictions: Makes the prediction of the dataset. 
• Future Predictions: 
o Scales the last 5 days of the original data. 
o Predicts the next value using the trained model. 
o Predicts for the next 30 days and plots the predictions.
