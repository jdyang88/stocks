# import streamlit as st
# import FinanceDataReader as fdr
# import datetime
# import matplotlib.pyplot as plt
# from keras.models import Sequential
# from keras.layers import Dense, LSTM
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# import numpy as np
# import pandas as pd
# from prophet import Prophet
# from prophet.plot import add_changepoints_to_plot

# # Streamlit app title
# st.title('Stock Price Prediction Dashboard')

# # Company and their stock codes
# companies = {
#     'SK Innovation (096770)': '096770',
#     'Samsung Electronics (005930)': '005930',
#     'SK Telecom (017670)': '017670',
#     'LG Chemicals (051910)': '051910',
#     'SK (034730)': '034730',
#     'Celtrion (068270)': '068270',
#     'Samsung Biologics (207940)': '207940',
#     'SK Hynix (000660)': '000660',
#     'Korean Air (003490)': '003490',
#     'Hyundae Motor (005380)': '005380',
#     'Posco Holdings (005490)': '005490',
#     # Add more companies and their codes as needed
# }

# # Company selection with ticker numbers
# selected_company = st.selectbox('Select a company:', list(companies.keys()))
# stock_code = companies[selected_company]

# # Displaying the selected company and stock code
# col1, col2 = st.columns(2)
# with col1:
#     st.write(f"Selected Company: {selected_company.split(' (')[0]}")
# with col2:    
#     st.write(f"Ticker Number: {stock_code}")

# # Initialize session state for start and end dates if not already set
# if 'start_date' not in st.session_state:
#     st.session_state['start_date'] = datetime.date(2000, 1, 1)
# if 'end_date' not in st.session_state:
#     st.session_state['end_date'] = datetime.datetime.today().date()

# # Function to update date range slider based on start/end date inputs
# def update_date_range():
#     st.session_state['date_range'] = (st.session_state['start_date'], st.session_state['end_date'])

# # Function to update start/end date inputs based on date range slider
# def update_start_end_dates():
#     st.session_state['start_date'], st.session_state['end_date'] = st.session_state['date_range']

# # Layout for start and end date inputs with callbacks
# col1, col2 = st.columns(2)
# with col1:
#     start_date = st.date_input('Start Date', st.session_state['start_date'], on_change=update_date_range, key='start_date')
# with col2:
#     end_date = st.date_input('End Date', st.session_state['end_date'], on_change=update_date_range, key='end_date')

# # Range slider for date selection
# date_range = st.slider(
#     "Select Date Range",
#     min_value=datetime.date(2000, 1, 1),
#     max_value=datetime.datetime.today().date(),
#     value=(st.session_state['start_date'], st.session_state['end_date']),
#     on_change=update_start_end_dates,
#     key='date_range'
# )

# # Updating the start and end dates based on the range slider
# start_date, end_date = date_range

# # FinanceDataReader to fetch stock data
# df = fdr.DataReader(stock_code, start_date, end_date)

# # Display stock price graph
# st.line_chart(df['Close'])

# # Display the source of stock data
# st.write("Data Source: [FinanceDataReader](https://financedata.github.io/finance_data_reader/)")

# # Checkbox to toggle the display of recent 10 days of closing prices
# if st.toggle('Show Recent 10 Days of Closing Prices'):
#     # Filter the DataFrame to get the last 10 days of data
#     last_10_days_df = df.tail(10)

#     # Display the last 10 days of data
#     st.write("Recent 10 days of Closing Prices:")
#     st.dataframe(last_10_days_df)

# # Initialize variables with default values
# trainScore = testScore = next_day_prediction = None
# future_prices = {}
# future_days = [10, 20, 30, 60]

# # LSTM Model Execution Button
# if st.button('Run LSTM Model to predict future price'):
#     st.write("*Modeling in progress. Please wait...*")
#     # Initialize a progress bar
#     progress_bar = st.progress(0)

#     # FinanceDataReader to fetch stock data
#     df = fdr.DataReader(stock_code, start_date, end_date)
#     dataset = df['Close'].values
#     dataset = dataset.astype('float32')
#     dataset = np.reshape(dataset, (-1, 1))
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     dataset = scaler.fit_transform(dataset)

#     # Update progress after data preparation
#     progress_bar.progress(10)

#     train_size = int(len(dataset) * 0.80)
#     test_size = len(dataset) - train_size
#     train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

#     look_back = 1

#     def create_dataset(dataset, look_back=1):
#         dataX, dataY = [], []
#         for i in range(len(dataset) - look_back - 1):
#             a = dataset[i:(i + look_back), 0]
#             dataX.append(a)
#             dataY.append(dataset[i + look_back, 0])
#         return np.array(dataX), np.array(dataY)

#     trainX, trainY = create_dataset(train, look_back)
#     testX, testY = create_dataset(test, look_back)

#     # Reshape input to be [samples, time steps, features]
#     trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
#     testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

#     # Create and fit the LSTM network
#     model = Sequential()
#     model.add(LSTM(50, input_shape=(1, look_back)))
#     model.add(Dense(1))
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2)


#     # Make predictions
#     trainPredict = model.predict(trainX)
#     testPredict = model.predict(testX)

#     # Expand the dimensions of trainY and testY for inverse transformation
#     trainY_expanded = np.expand_dims(trainY, axis=1)
#     testY_expanded = np.expand_dims(testY, axis=1)

#     # Invert predictions
#     trainPredict = scaler.inverse_transform(trainPredict)
#     trainY = scaler.inverse_transform(trainY_expanded)
#     testPredict = scaler.inverse_transform(testPredict)
#     testY = scaler.inverse_transform(testY_expanded)

#     # Calculate root mean squared error
#     trainScore = np.sqrt(mean_squared_error(trainY, trainPredict))
#     testScore = np.sqrt(mean_squared_error(testY, testPredict))

#     # Calculate average values of the training and test datasets
#     train_avg = np.average(trainY)
#     test_avg = np.average(testY)

#     # Calculate accuracy percentage
#     train_accuracy = (1 - trainScore / train_avg) * 100
#     test_accuracy = (1 - testScore / test_avg) * 100

#     # Display Model Outputs and Accuracy
#     if trainScore is not None and train_accuracy is not None:
#         st.write(f"Train RMSE: {trainScore:.2f}")
#         # st.write(f"Train Predicted Accuracy: {train_accuracy:.2f}%")

#     if testScore is not None and test_accuracy is not None:
#         st.write(f"Test RMSE: {testScore:.2f}")
#         # st.write(f"Test Predicted Accuracy: {test_accuracy:.2f}%")

#     last_price = dataset[-1]
#     last_price = np.reshape(last_price, (1, 1, 1))
#     next_day_prediction = model.predict(last_price)
#     next_day_prediction = scaler.inverse_transform(next_day_prediction)

#     if next_day_prediction is not None:
#         st.write(f"Tomorrow's {selected_company.split(' (')[0]} Stock Price Prediction: {next_day_prediction[0, 0]:.2f}")

#     # Future Price Predictions
#     def predict_future_price(model, last_price, days):
#         future_prices = []
#         for _ in range(days):
#             next_day_prediction = model.predict(last_price)
#             future_prices.append(scaler.inverse_transform(next_day_prediction)[0])
#             last_price = np.reshape(next_day_prediction, (1, 1, 1))
#         return future_prices

#     future_prices = predict_future_price(model, last_price, max(future_days))

#     for days in future_days:
#         if future_prices[days - 1] is not None:
#             st.write(f"{days} days later {selected_company.split(' (')[0]} Stock Price Prediction: {future_prices[days - 1][0]:.2f}")

#     # Display Prediction Graphs
#     plt.figure(figsize=(14, 7))
#     plt.plot(scaler.inverse_transform(dataset), label='Actual Stock Price')

#     # Plotting train predictions
#     trainPredictPlot = np.empty_like(dataset)
#     trainPredictPlot[:, :] = np.nan
#     trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
#     plt.plot(trainPredictPlot, label='Train Predictions')

#     # Plotting test predictions
#     testPredictPlot = np.empty_like(dataset)
#     testPredictPlot[:, :] = np.nan
#     testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
#     plt.plot(testPredictPlot, label='Test Predictions')

#     # Finalize the progress bar
#     progress_bar.progress(100)

#     # Plotting future predictions
#     for days in future_days:
#         plt.axvline(x=len(dataset) - 1 + days, color='r', linestyle='--', alpha=0.3)
#         plt.text(len(dataset) - 1 + days, future_prices[days - 1][0], f"{days} days: {future_prices[days - 1][0]:.2f}", fontsize=9)

#     plt.xlabel('Time', fontsize=14)
#     plt.ylabel('Stock Price', fontsize=14)
#     plt.title('Stock Price Prediction', fontsize=16)
#     plt.legend()
#     plt.grid()
#     st.pyplot(plt)

#     # LSTM Model Explanation
#     st.write("""
#     **LSTM (Long Short-Term Memory) Model:**
#     LSTM is a type of recurrent neural network (RNN) that is well-suited to predict time series data. 
#     It uses memory cells to store information over long periods of time, making it effective for capturing patterns in sequential data.
#     In this application, the LSTM model is used to predict future stock prices based on historical closing prices.
#     """)



# # Prophet Model Execution Button
# if st.button('Run Prophet Model to predict future price'):
#     st.write("*Prophet Modeling in progress. Please wait...*")
    
#     # Initialize a progress bar for Prophet
#     prophet_progress_bar = st.progress(0)

#     # FinanceDataReader to fetch stock data
#     df = fdr.DataReader(stock_code, start_date, end_date)
#     df = df.reset_index()
#     df.rename(columns={'Date':'ds', 'Close':'y'}, inplace=True)

#     # Update progress after data preparation
#     prophet_progress_bar.progress(10)

#     # Create and fit the Prophet model
#     prophet_model = Prophet(daily_seasonality=True)
#     prophet_model.fit(df)

#     # Create future dataframe for predictions
#     future = prophet_model.make_future_dataframe(periods=max(future_days))
#     forecast = prophet_model.predict(future)

#     # Update progress
#     prophet_progress_bar.progress(50)

#     # Customizing the forecast plot using Matplotlib
#     fig1, ax1 = plt.subplots(figsize=(10, 6))
#     prophet_model.plot(forecast, ax=ax1)
#     ax1.set_ylabel('Stock Price')  # Set Y-axis label
#     ax1.set_title(f'Forecast for {selected_company.split(" (")[0]}')  # Set Title
#     add_changepoints_to_plot(ax1, prophet_model, forecast)  # Optional: to add changepoints

#     st.pyplot(fig1)

#     # Displaying the forecast components
#     fig2 = prophet_model.plot_components(forecast)
#     st.pyplot(fig2)

#     # Update the progress bar to complete
#     prophet_progress_bar.progress(100)

#     # Display future price predictions
#     for days in future_days:
#         predicted_price = forecast.iloc[-days]['yhat']
#         st.write(f"{days} days later {selected_company.split(' (')[0]} Stock Price Prediction: {predicted_price:.2f}")

#     # Prophet Model Explanation
#     st.write("""
#     **Prophet Model:**
#     Prophet is a forecasting tool developed by Facebook, designed for time series data. 
#     It is particularly effective for time series with strong seasonal effects and missing data. 
#     The model decomposes the time series into trend, seasonality, and holiday effects, making it robust for various business time series predictions.
#     In this application, the Prophet model is used to predict future stock prices based on historical closing prices.
#     """)



# stocks.py

import streamlit as st
import FinanceDataReader as fdr
import datetime
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import os
from newsapi import NewsApiClient
from textblob import TextBlob
import nltk
nltk.download('punkt')

# Streamlit app title
st.title('Enhanced Stock Price Prediction Dashboard with Sentiment Analysis')

# Company and their stock codes
companies = {
    'SK Innovation (096770)': '096770',
    'Samsung Electronics (005930)': '005930',
    'SK Telecom (017670)': '017670',
    'LG Chemicals (051910)': '051910',
    'SK (034730)': '034730',
    'Celtrion (068270)': '068270',
    'Samsung Biologics (207940)': '207940',
    'SK Hynix (000660)': '000660',
    'Korean Air (003490)': '003490',
    'Hyundae Motor (005380)': '005380',
    'Posco Holdings (005490)': '005490',
    # Add more companies and their codes as needed
}

# Company selection with ticker numbers
selected_company = st.selectbox('Select a company:', list(companies.keys()))
stock_code = companies[selected_company]
company_name = selected_company.split(' (')[0]

# Displaying the selected company and stock code
col1, col2 = st.columns(2)
with col1:
    st.write(f"Selected Company: {company_name}")
with col2:
    st.write(f"Ticker Number: {stock_code}")

# Initialize session state for start and end dates if not already set
if 'start_date' not in st.session_state:
    st.session_state['start_date'] = datetime.date(2000, 1, 1)
if 'end_date' not in st.session_state:
    st.session_state['end_date'] = datetime.datetime.today().date()

# Function to update date range slider based on start/end date inputs
def update_date_range():
    st.session_state['date_range'] = (st.session_state['start_date'], st.session_state['end_date'])

# Function to update start/end date inputs based on date range slider
def update_start_end_dates():
    st.session_state['start_date'], st.session_state['end_date'] = st.session_state['date_range']

# Layout for start and end date inputs with callbacks
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input('Start Date', st.session_state['start_date'], on_change=update_date_range, key='start_date')
with col2:
    end_date = st.date_input('End Date', st.session_state['end_date'], on_change=update_date_range, key='end_date')

# Range slider for date selection
date_range = st.slider(
    "Select Date Range",
    min_value=datetime.date(2000, 1, 1),
    max_value=datetime.datetime.today().date(),
    value=(st.session_state['start_date'], st.session_state['end_date']),
    on_change=update_start_end_dates,
    key='date_range'
)

# Updating the start and end dates based on the range slider
start_date, end_date = date_range

# FinanceDataReader to fetch stock data
df = fdr.DataReader(stock_code, start_date, end_date)

# Display stock price graph
st.line_chart(df['Close'])

# Display the source of stock data
st.write("Data Source: [FinanceDataReader](https://financedata.github.io/finance_data_reader/)")

# Checkbox to toggle the display of recent 10 days of closing prices
if st.checkbox('Show Recent 10 Days of Closing Prices'):
    # Filter the DataFrame to get the last 10 days of data
    last_10_days_df = df.tail(10)

    # Display the last 10 days of data
    st.write("Recent 10 days of Closing Prices:")
    st.dataframe(last_10_days_df)

# Initialize variables with default values
trainScore = testScore = next_day_prediction = None
future_prices = {}
future_days = [10, 20, 30, 60]

# LSTM Model Execution Button
if st.button('Run Enhanced LSTM Model to Predict Future Price'):
    st.write("*Modeling in progress. Please wait...*")
    # Initialize a progress bar
    progress_bar = st.progress(0)

    # FinanceDataReader to fetch stock data
    df = fdr.DataReader(stock_code, start_date, end_date)
    dataset = df['Close'].values
    dataset = dataset.astype('float32')
    dataset = np.reshape(dataset, (-1, 1))

    # **Sentiment Analysis Integration**
    # Fetch recent news articles
    st.write("Fetching recent news articles for sentiment analysis...")
    # Get the API key from environment variable
    news_api_key = os.environ.get('c4b101fd9551431cadbae50314fba17c')
    if not news_api_key:
        st.error("News API Key is not set. Please set the NEWS_API_KEY environment variable.")
        st.stop()

    newsapi = NewsApiClient(api_key=news_api_key)
    try:
        articles = newsapi.get_everything(
            q=company_name,
            language='en',
            sort_by='relevancy',
            page_size=100
        )
    except Exception as e:
        st.error(f"Error fetching news articles: {e}")
        st.stop()

    # Perform sentiment analysis on the articles
    sentiments = []
    for article in articles['articles']:
        description = article['description']
        if description:
            analysis = TextBlob(description)
            sentiments.append(analysis.sentiment.polarity)

    # Calculate average sentiment score
    if sentiments:
        avg_sentiment = np.mean(sentiments)
    else:
        avg_sentiment = 0  # Neutral sentiment if no articles are found

    # Update progress after sentiment analysis
    progress_bar.progress(20)

    # Incorporate sentiment score into the dataset
    sentiment_feature = np.full((dataset.shape[0], 1), avg_sentiment)
    dataset_with_sentiment = np.hstack((dataset, sentiment_feature))

    # Scaling the features
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_scaled = scaler.fit_transform(dataset_with_sentiment)

    # Update progress after data preparation
    progress_bar.progress(30)

    train_size = int(len(dataset_scaled) * 0.80)
    test_size = len(dataset_scaled) - train_size
    train, test = dataset_scaled[0:train_size, :], dataset_scaled[train_size:len(dataset_scaled), :]

    look_back = 1

    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), :]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # Reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], look_back, trainX.shape[2]))
    testX = np.reshape(testX, (testX.shape[0], look_back, testX.shape[2]))

    # Create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, trainX.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2)

    # Update progress after model training
    progress_bar.progress(70)

    # Make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # Invert predictions and actual values for RMSE calculation
    trainPredict_inverted = scaler.inverse_transform(np.hstack((trainPredict, np.full((trainPredict.shape[0], 1), avg_sentiment))))
    trainY_inverted = scaler.inverse_transform(np.hstack((trainY.reshape(-1, 1), np.full((trainY.shape[0], 1), avg_sentiment))))
    testPredict_inverted = scaler.inverse_transform(np.hstack((testPredict, np.full((testPredict.shape[0], 1), avg_sentiment))))
    testY_inverted = scaler.inverse_transform(np.hstack((testY.reshape(-1, 1), np.full((testY.shape[0], 1), avg_sentiment))))

    # Calculate root mean squared error
    trainScore = np.sqrt(mean_squared_error(trainY_inverted[:,0], trainPredict_inverted[:,0]))
    testScore = np.sqrt(mean_squared_error(testY_inverted[:,0], testPredict_inverted[:,0]))

    # Calculate average values of the training and test datasets
    train_avg = np.average(trainY_inverted[:,0])
    test_avg = np.average(testY_inverted[:,0])

    # Calculate accuracy percentage
    train_accuracy = (1 - trainScore / train_avg) * 100
    test_accuracy = (1 - testScore / test_avg) * 100

    # Display Model Outputs and Accuracy
    if trainScore is not None and train_accuracy is not None:
        st.write(f"Train RMSE: {trainScore:.2f}")
        st.write(f"Train Accuracy: {train_accuracy:.2f}%")

    if testScore is not None and test_accuracy is not None:
        st.write(f"Test RMSE: {testScore:.2f}")
        st.write(f"Test Accuracy: {test_accuracy:.2f}%")

    # Predicting future prices
    last_values = dataset_scaled[-look_back:]
    future_prices = []

    for day_ahead in range(1, max(future_days) + 1):
        prediction = model.predict(last_values.reshape(1, look_back, trainX.shape[2]))
        inverse_prediction = scaler.inverse_transform(np.hstack((prediction, [[avg_sentiment]])))
        future_prices.append(inverse_prediction[0, 0])
        # Append the prediction to last_values for the next prediction
        next_input = np.hstack((prediction, [[avg_sentiment]]))
        last_values = np.vstack((last_values[1:], next_input))

    # Display future price predictions
    for days in future_days:
        st.write(f"{days} days later {company_name} Stock Price Prediction: {future_prices[days - 1]:.2f}")

    # Update progress after predictions
    progress_bar.progress(90)

    # Display Prediction Graphs
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, dataset[:,0], label='Actual Stock Price')

    # Plotting train predictions
    trainPredictPlot = np.empty_like(dataset[:,0])
    trainPredictPlot[:] = np.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back] = trainPredict_inverted[:,0]
    plt.plot(df.index, trainPredictPlot, label='Train Predictions')

    # Plotting test predictions
    testPredictPlot = np.empty_like(dataset[:,0])
    testPredictPlot[:] = np.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1] = testPredict_inverted[:,0]
    plt.plot(df.index, testPredictPlot, label='Test Predictions')

    # Plotting future predictions
    future_dates = pd.date_range(start=df.index[-1], periods=max(future_days)+1, closed='right')
    plt.plot(future_dates, future_prices, marker='o', linestyle='--', label='Future Predictions')

    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Stock Price', fontsize=14)
    plt.title('Enhanced Stock Price Prediction with Sentiment Analysis', fontsize=16)
    plt.legend()
    plt.grid()
    st.pyplot(plt)

    # Finalize the progress bar
    progress_bar.progress(100)

    # LSTM Model Explanation
    st.write("""
    **Enhanced LSTM Model with Sentiment Analysis:**

    This model incorporates sentiment analysis of recent news articles related to the selected company. By analyzing the sentiment (positive, negative, neutral) expressed in the news, we can gain insights into the public perception of the company, which often impacts stock prices.

    The sentiment score is integrated as an additional feature in the LSTM model, enabling it to learn from both historical stock prices and current public sentiment.

    **Note**: The sentiment score is calculated using the TextBlob library, which analyzes the polarity of the news article descriptions.
    """)
