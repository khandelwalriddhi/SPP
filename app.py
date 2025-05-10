import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the pre-trained model
model = load_model(r'C:\Users\a\SPP\Stock Prediction Model.keras')

# Streamlit header
st.header('Stock Market Predictor')

# List of company codes (add more as needed)
company_codes = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "INTC",
    "IBM", "ORCL", "ADBE", "AMD", "BABA", "PYPL", "DIS", "CSCO", "TWTR"
]

# Input for searching company codes
search_input = st.text_input('Search Company Code', '')
suggestions = [code for code in company_codes if code.lower().startswith(search_input.lower())]

# Dropdown menu for suggestions
if search_input:
    selected_company = st.selectbox('Suggested Company Codes', suggestions)
else:
    selected_company = st.selectbox('Suggested Company Codes', company_codes)

# Default stock symbol or user-selected symbol
stock = selected_company if selected_company else 'NVDA'

# Date range for data
start = '2012-01-01'
end = '2023-12-12'

# Fetch stock data
data = yf.download(stock, start, end)

# Display fetched stock data
st.subheader('Stock Data')
st.write(data)

# Split data into training and testing sets
data_train = pd.DataFrame(data.Close[0:int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80):len(data)])

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.fit_transform(data_test)

# Calculate moving averages
ma_50_days = data.Close.rolling(50).mean()
ma_100_days = data.Close.rolling(100).mean()
ma_200_days = data.Close.rolling(200).mean()

# Prepare data for model prediction
x = []
y = []
for i in range(100, data_test_scaled.shape[0]):
    x.append(data_test_scaled[i - 100:i])
    y.append(data_test_scaled[i, 0])

x, y = np.array(x), np.array(y)

# Predict using the model
predictions = model.predict(x)
scale = 1 / scaler.scale_
predictions = predictions * scale
y = y * scale

# Dropdown for graph selection
st.subheader("Visualization Options")
graph_type = st.selectbox(
    "Select the type of graph you want to view:",
    ["Price vs MA50", "Price vs MA50 vs MA100", "Price vs MA100 vs MA200", "Original vs Predicted", "Bar Chart", "Pie Chart"]
)

# Generate graphs based on the selected option
if graph_type == "Price vs MA50":
    st.subheader('Price vs MA50')
    fig1 = plt.figure(figsize=(8, 6))
    plt.plot(ma_50_days, 'r', label='MA50')
    plt.plot(data.Close, 'g', label='Close Price')
    plt.legend()
    plt.title('Price vs MA50')
    plt.xlabel('Date')
    plt.ylabel('Price')
    st.pyplot(fig1)

elif graph_type == "Price vs MA50 vs MA100":
    st.subheader('Price vs MA50 vs MA100')
    fig2 = plt.figure(figsize=(8, 6))
    plt.plot(ma_50_days, 'r', label='MA50')
    plt.plot(ma_100_days, 'b', label='MA100')
    plt.plot(data.Close, 'g', label='Close Price')
    plt.legend()
    plt.title('Price vs MA50 vs MA100')
    plt.xlabel('Date')
    plt.ylabel('Price')
    st.pyplot(fig2)

elif graph_type == "Price vs MA100 vs MA200":
    st.subheader('Price vs MA100 vs MA200')
    fig3 = plt.figure(figsize=(8, 6))
    plt.plot(ma_100_days, 'r', label='MA100')
    plt.plot(ma_200_days, 'b', label='MA200')
    plt.plot(data.Close, 'g', label='Close Price')
    plt.legend()
    plt.title('Price vs MA100 vs MA200')
    plt.xlabel('Date')
    plt.ylabel('Price')
    st.pyplot(fig3)

elif graph_type == "Original vs Predicted":
    st.subheader('Original Price vs Predicted Price')
    fig4 = plt.figure(figsize=(8, 6))
    plt.plot(predictions, 'r', label='Predicted Price')
    plt.plot(y, 'g', label='Original Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.title('Original Price vs Predicted Price')
    st.pyplot(fig4)

elif graph_type == "Bar Chart":
    st.subheader('Bar Chart: Closing Price by Month')
    data['Month'] = data.index.month
    monthly_avg = data.groupby('Month')['Close'].mean()
    fig5 = plt.figure(figsize=(8, 6))
    plt.bar(monthly_avg.index, monthly_avg.values, color='skyblue')
    plt.xlabel('Month')
    plt.ylabel('Average Closing Price')
    plt.title('Monthly Average Closing Price')
    st.pyplot(fig5)

elif graph_type == "Pie Chart":
    st.subheader('Pie Chart: Percentage of Data by Year')
    data['Year'] = data.index.year
    yearly_data_count = data['Year'].value_counts()
    fig6 = plt.figure(figsize=(8, 6))
    plt.pie(yearly_data_count, labels=yearly_data_count.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.tab10.colors)
    plt.title('Percentage of Data by Year')
    st.pyplot(fig6)
