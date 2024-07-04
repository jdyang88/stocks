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
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot

# Streamlit app title
st.title('Stock Price Prediction Dashboard')

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

# Displaying the selected company and stock code
col1, col2 = st.columns(2)
with col1:
    st.write(f"Selected Company: {selected_company.split(' (')[0]}")
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
if st.button('Run LSTM Model to predict future price'):
    st.write("*Modeling in progress. Please wait...*")
    # Initialize a progress bar
    progress_bar = st.progress(0)

    # FinanceDataReader to fetch stock data
    df = fdr.DataReader(stock_code, start_date, end_date)
    dataset = df['Close'].values
    dataset = dataset.astype('float32')
    dataset = np.reshape(dataset, (-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # Update progress after data preparation
    progress_bar.progress(10)

    train_size = int(len(dataset) * 0.80)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    look_back = 1

    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # Reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # Create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(50, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2)

    # Make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # Expand the dimensions of trainY and testY for inverse transformation
    trainY_expanded = np.expand_dims(trainY, axis=1)
    testY_expanded = np.expand_dims(testY, axis=1)

    # Invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform(trainY_expanded)
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform(testY_expanded)

    # Calculate root mean squared error
    trainScore = np.sqrt(mean_squared_error(trainY, trainPredict))
    testScore = np.sqrt(mean_squared_error(testY, testPredict))

    # Calculate average values of the training and test datasets
    train_avg = np.average(trainY)
    test_avg = np.average(testY)

    # Calculate accuracy percentage
    train_accuracy = (1 - trainScore / train_avg) * 100
    test_accuracy = (1 - testScore / test_avg) * 100

    # Display Model Outputs and Accuracy
    if trainScore is not None and train_accuracy is not None:
        st.write(f"Train RMSE: {trainScore:.2f}")
        # st.write(f"Train Predicted Accuracy: {train_accuracy:.2f}%")

    if testScore is not None and test_accuracy is not None:
        st.write(f"Test RMSE: {testScore:.2f}")
        # st.write(f"Test Predicted Accuracy: {test_accuracy:.2f}%")

    last_price = dataset[-1]
    last_price = np.reshape(last_price, (1, 1, 1))
    next_day_prediction = model.predict(last_price)
    next_day_prediction = scaler.inverse_transform(next_day_prediction)

    if next_day_prediction is not None:
        st.write(f"Tomorrow's {selected_company.split(' (')[0]} Stock Price Prediction: {next_day_prediction[0, 0]:.2f}")

    # Future Price Predictions
    def predict_future_price(model, last_price, days):
        future_prices = []
        for _ in range(days):
            next_day_prediction = model.predict(last_price)
            future_prices.append(scaler.inverse_transform(next_day_prediction)[0])
            last_price = np.reshape(next_day_prediction, (1, 1, 1))
        return future_prices

    future_prices = predict_future_price(model, last_price, max(future_days))

    for days in future_days:
        st.write(f"{days} days later {selected_company.split(' (')[0]} Stock Price Prediction: {future_prices[days - 1][0]:.2f}")

    # Display Prediction Graphs
    plt.figure(figsize=(14, 7))
    plt.plot(scaler.inverse_transform(dataset), label='Actual Stock Price')

    # Plotting train predictions
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
    plt.plot(trainPredictPlot, label='Train Predictions')

    # Plotting test predictions
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
    plt.plot(testPredictPlot, label='Test Predictions')

    # Finalize the progress bar
    progress_bar.progress(100)

    # Plotting future predictions
    for days in future_days:
        plt.axvline(x=len(dataset) - 1 + days, color='r', linestyle='--', alpha=0.3)
        plt.text(len(dataset) - 1 + days, future_prices[days - 1][0], f"{days} days: {future_prices[days - 1][0]:.2f}", fontsize=9)

    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Stock Price', fontsize=14)
    plt.title('Stock Price Prediction', fontsize=16)
    plt.legend()
    plt.grid()
    st.pyplot(plt)

# Prophet Model Execution Button
if st.button('Run Prophet Model to predict future price'):
    st.write("*Prophet Modeling in progress. Please wait...*")
    
    # Initialize a progress bar for Prophet
    prophet_progress_bar = st.progress(0)

    # FinanceDataReader to fetch stock data
    df = fdr.DataReader(stock_code, start_date, end_date)
    df = df.reset_index()
    df.rename(columns={'Date':'ds', 'Close':'y'}, inplace=True)

    # Update progress after data preparation
    prophet_progress_bar.progress(10)

    # Create and fit the Prophet model
    prophet_model = Prophet(daily_seasonality=True)
    prophet_model.fit(df)

    # Create future dataframe for predictions
    future = prophet_model.make_future_dataframe(periods=max(future_days))
    forecast = prophet_model.predict(future)

    # Update progress
    prophet_progress_bar.progress(50)

    # Customizing the forecast plot using Matplotlib
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    prophet_model.plot(forecast, ax=ax1)
    ax1.set_ylabel('Stock Price')  # Set Y-axis label
    ax1.set_title(f'Forecast for {selected_company.split(" (")[0]}')  # Set Title
    add_changepoints_to_plot(ax1, prophet_model, forecast)  # Optional: to add changepoints

    st.pyplot(fig1)

    # Displaying the forecast components
    fig2 = prophet_model.plot_components(forecast)
    st.pyplot(fig2)

    # Update the progress bar to complete
    prophet_progress_bar.progress(100)

    # Display future price predictions
    for days in future_days:
        predicted_price = forecast.iloc[-days]['yhat']
        st.write(f"{days} days later {selected_company.split(' (')[0]} Stock Price Prediction: {predicted_price:.2f}")

