import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import yfinance as yf
warnings.filterwarnings("ignore")
import streamlit as st
import datetime
import plotly.graph_objs as go
import numpy as np
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf
from keras import layers
from keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN,Layer #type:ignore
from keras.optimizers import SGD #type:ignore
from keras.models import Sequential #type:ignore
import io


# Initialize session state variables
if "page" not in st.session_state:
    st.session_state["page"] = "About"
if "stock_data" not in st.session_state:
    st.session_state["stock_data"] = None
if "data_scaled" not in st.session_state:
    st.session_state["data_scaled"] = None
if "scaler" not in st.session_state:
    st.session_state["scaler"] = None

# Function to fetch stock data
def fetch_meta_stock_data(start_date, end_date, ticker):
    try:
        # Fetch the data
        print(f"Fetching stock data for {ticker} from {start_date} to {end_date}...")
        data = yf.download(ticker, start=start_date, end=end_date)

        # Check if data was retrieved
        if data.empty:
            print("No data was retrieved. Please check the ticker or date range.")
            return pd.DataFrame()  # Return an empty DataFrame

        # Reset the index to make 'Date' a column
        data = data.reset_index()

        # Rename the 'Date' column if necessary
        if 'Date' not in data.columns and 'date' in data.columns:
            data = data.rename(columns={'date': 'Date'})

        # Ensure 'Date' column is datetime
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None) 

        print(f"Data successfully fetched for {ticker}.")
        return data

    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of an error

# Function to preprocess stock data
def preprocess_data(data):
    # Handle missing values (forward fill)
    data = data.ffill()

    # Detect and handle outliers using z-score
    z_scores = (data[f'Close {ticker}'] - data[f'Close {ticker}'].mean()) / data[f'Close {ticker}'].std()
    data = data[(z_scores < 3) & (z_scores > -3)]

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[[f'Close {ticker}']])

    return data, data_scaled, scaler

# Function to create an interactive Plotly chart
def plot_interactive(data, ticker):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=data['Date'],
        open=data[f'Open {ticker}'],
        high=data[f'High {ticker}'],
        low=data[f'Low {ticker}'],
        close=data[f'Close {ticker}'],
        name="Candlestick"
    ))

    fig.update_layout(
        title=f"{ticker} Stock Price Data: {data['Date'].min().date()} to {data['Date'].max().date()}",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_dark"
    )

    return fig
def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    return stream.getvalue()



def build_normal_lstm_model(input_shape):
    model = Sequential()

    # Single LSTM layer with dropout
    model.add(LSTM(units=100, return_sequences=False, input_shape=input_shape))
    model.add(Dropout(0.2))  # Dropout to prevent overfitting

    # Output layer
    model.add(Dense(units=1))  # Predict next day's stock price

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    st.markdown("<h4 style='font-size: 18px;color: purple;'>Vanilla LSTM Model Summary</h4>", unsafe_allow_html=True)
    summary_text = get_model_summary(model)
    st.text(summary_text)
    return model

def build_improved_lstm_model(input_shape):
    model = Sequential()

    # First LSTM layer with dropout
    model.add(LSTM(units=120, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))  # Dropout to prevent overfitting

    # Second LSTM layer with dropout
    model.add(LSTM(units=120, return_sequences=True))
    model.add(Dropout(0.3))

    # Third LSTM layer with dropout
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(units=1))  # Predict next day's stock price

    # Compile the model with Adam optimizer
    model.compile(optimizer='adam', loss='mean_squared_error')
    st.markdown("<h4 style='font-size: 18px;color: purple;'>Stacked LSTM Model Summary</h4>", unsafe_allow_html=True)
    summary_text = get_model_summary(model)
    st.text(summary_text)    
    return model

# Function to build a single-layer GRU model
def build_normal_gru_model(input_shape):
    model = Sequential()

    # Single GRU layer with dropout
    model.add(GRU(units=100, return_sequences=False, input_shape=input_shape))
    model.add(Dropout(0.2))  # Dropout to prevent overfitting

    # Output layer
    model.add(Dense(units=1))  # Predict next day's stock price

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    st.markdown("<h4 style='font-size: 18px;color: purple;'>GRU Model Summary</h4>", unsafe_allow_html=True)
    summary_text = get_model_summary(model)
    st.text(summary_text)    
    return model

def build_momentum_rnn(input_shape):
    model = Sequential()

    # Simple RNN layer with momentum optimization
    model.add(SimpleRNN(units=100, return_sequences=False, input_shape=input_shape))
    model.add(Dropout(0.2))  # Dropout to prevent overfitting

    # Output layer
    model.add(Dense(units=1))  # For stock price prediction

    # Compile the model with Momentum-based SGD optimizer
    momentum_optimizer = SGD(learning_rate=0.01, momentum=0.9)  # Using momentum
    model.compile(optimizer=momentum_optimizer, loss='mean_squared_error')
    st.markdown("<h4 style='font-size: 18px;color: purple;'>Momentum RNN Model Summary</h4>", unsafe_allow_html=True)
    summary_text = get_model_summary(model)
    st.text(summary_text)    
    return model

class CustomMogrifierLSTM(Layer):
    def __init__(self, units, mogrifier_rounds=5, **kwargs):
        super(CustomMogrifierLSTM, self).__init__(**kwargs)
        self.units = units
        self.mogrifier_rounds = mogrifier_rounds
        self.lstm_layer = LSTM(units, return_sequences=True)

    def build(self, input_shape):
        # Create trainable weights for the Mogrifier transformation
        # The shape of Wx should be (units, units) to match h's shape
        self.Wx = self.add_weight(
            shape=(self.units, self.units),  # Changed shape to (units, units)
            initializer="random_normal",
            trainable=True,
            name="Wx"
        )
        self.Wh = self.add_weight(
            shape=(self.units, self.units),  # Changed shape to (units, units)
            initializer="random_normal",
            trainable=True,
            name="Wh"
        )
        super(CustomMogrifierLSTM, self).build(input_shape)

    def call(self, inputs):
        x = inputs
        h = self.lstm_layer(x)

        for _ in range(self.mogrifier_rounds):
            # Update x using h
            x = x * tf.sigmoid(tf.matmul(h, self.Wh))
            # Update h using x
            # Transpose Wx to align dimensions for matrix multiplication
            h = h * tf.sigmoid(tf.matmul(x, tf.transpose(self.Wx))) # Added transpose

        return h

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.units)

def build_mogrifier_lstm_model(input_shape, units=128, mogrifier_rounds=5):
    model = Sequential()
    model.add(CustomMogrifierLSTM(units, mogrifier_rounds=mogrifier_rounds, input_shape=input_shape))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    summary_text = get_model_summary(model)
    st.markdown("<h4 style='font-size: 18px;color: purple;'>Mogrifier LSTM Model Summary</h4>", unsafe_allow_html=True)
    st.text(summary_text)    
    return model

# Function to plot predictions and calculate metrics
def evaluate_and_plot(model, X_train, y_train, X_test, y_test, scaler=None):
    # Train the model
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test), verbose=1)    
    # Make predictions
    predicted_prices = model.predict(X_test)

    # Inverse transform the scaled data to get actual prices
    predicted_prices = scaler.inverse_transform(predicted_prices)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Plot actual vs predicted prices
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_actual, label='Actual Prices')
    plt.plot(predicted_prices, label='Predicted Prices')
    plt.title("Stock Price Prediction")
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    st.pyplot(plt)
    # Calculate evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_test_actual, predicted_prices))
    mae = mean_absolute_error(y_test_actual, predicted_prices)
    mse = mean_squared_error(y_test_actual, predicted_prices)
    r2 = r2_score(y_test_actual, predicted_prices)
    
    return mse, mae, rmse, r2

def compare_models(X_train, y_train, X_test, y_test, scaler=None):
    # Initialize a DataFrame to store the results
    results_list = []

    # 1. Normal LSTM
    model_lstm = build_normal_lstm_model((X_train.shape[1], 1))
    print("Training VANILLA LSTM...")
    st.write("Training VANILLA LSTM...")
    mse, mae, rmse, r2 = evaluate_and_plot(model_lstm, X_train, y_train, X_test, y_test, scaler)
    results_list.append({'Model': 'VANILLA LSTM', 'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2}) # append to list
    
    # 2. Improved LSTM
    model_improved_lstm = build_improved_lstm_model((X_train.shape[1], 1))
    print("Training STACKED LSTM...")
    st.write("Training STACKED LSTM...")
    mse, mae, rmse, r2 = evaluate_and_plot(model_improved_lstm, X_train, y_train, X_test, y_test, scaler)
    results_list.append({'Model': 'STACKED LSTM', 'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2}) # append to list
    
    # 3. Normal GRU
    model_gru = build_normal_gru_model((X_train.shape[1], 1))
    print("Training GRU...")
    st.write("Training GRU...")
    mse, mae, rmse, r2 = evaluate_and_plot(model_gru, X_train, y_train, X_test, y_test, scaler)
    results_list.append({'Model': ' GRU', 'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2}) # append to list
    
    # 4. RNN Model
    model_hybrid = build_momentum_rnn((X_train.shape[1], 1))
    print("Training MOMENTUM RNN...")
    st.write("Training MOMENTUM RNN...")
    mse, mae, rmse, r2 = evaluate_and_plot(model_hybrid, X_train, y_train, X_test, y_test, scaler)
    results_list.append({'Model': 'MOMENTUM RNN', 'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2}) # append to list

    # 5. Mogrifier LSTM Model
    model_mogrifier = build_mogrifier_lstm_model((X_train.shape[1], 1))
    print("Training MOGRIFIER LSTM...")
    st.write("Training MOGRIFIER LSTM...")
    mse, mae, rmse, r2 = evaluate_and_plot(model_mogrifier, X_train, y_train, X_test, y_test, scaler)
    results_list.append({'Model': 'MOGRIFIER LSTM', 'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2}) # append to list

    results = pd.DataFrame(results_list)

    
    # Display the comparison table
    print(results)
    return pd.DataFrame(results_list)


# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["About", "Predict"])

# About Page
if page == "About":
    st.session_state["page"] = "About"
    st.title("About the Stock Prediction App")
    st.write("""
        This application allows users to fetch and visualize stock market data. 
        In the **Predict** section, you can:
        - Fetch 5 years of historical data for any stock ticker.
        - Visualize the data interactively using candlestick charts.
    """)

# Predict Page
if page == "Predict":
    st.session_state["page"] = "Predict"

    # Sidebar for user input
    st.sidebar.title("Stock Data Fetcher")
    ticker = st.sidebar.text_input("Enter stock ticker (e.g., META):", "META")
    end_date = datetime.datetime.today().date()
    start_date = end_date - datetime.timedelta(days=5 * 365)

    if st.sidebar.button("Fetch Data"):
        with st.spinner("Getting Your Data..."):

            # Fetch stock data
            data = fetch_meta_stock_data(start_date, end_date, ticker)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [' '.join(col).strip() for col in data.columns.values]
            if not data.empty:
                # Preprocess the data
                data, data_scaled, scaler = preprocess_data(data)

                # Save to session state
                st.session_state["stock_data"] = data
                st.session_state["data_scaled"] = data_scaled
                st.session_state["scaler"] = scaler

                st.success("Data successfully fetched and preprocessed!")

    # Display the data and interactive plot if available
    if st.session_state["stock_data"] is not None:
        st.subheader(f"Stock Data for {ticker}")
        st.dataframe(st.session_state["stock_data"].tail())  # Show the last 5 rows

        st.subheader("Interactive Candlestick Chart")
        fig = plot_interactive(st.session_state["stock_data"], ticker)
        st.plotly_chart(fig, use_container_width=True)
    
    # Check if data_scaled exists in session state
    if st.session_state["data_scaled"] is not None:
        st.subheader("Preparing Data for LSTM")
        
        # Function to create sequences
        def create_sequences(data_scaled, sequence_length=60):
            X, y = [], []
            for i in range(sequence_length, len(data_scaled)):
                X.append(data_scaled[i-sequence_length:i, 0])
                y.append(data_scaled[i, 0])
            X = np.array(X)
            y = np.array(y)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM
            return X, y

        # Generate sequences
        X, y = create_sequences(st.session_state["data_scaled"])
        st.session_state["X"], st.session_state["y"] = X, y  # Save to session state

        st.write(f"Data prepared for LSTM: X shape = {X.shape}, y shape = {y.shape}")
    
    if "X" in st.session_state and "y" in st.session_state:
        st.subheader("Splitting Data into Training and Testing Sets")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            st.session_state["X"], st.session_state["y"], test_size=0.2, shuffle=False
        )

        # Save to session state
        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test
        st.session_state["y_train"] = y_train
        st.session_state["y_test"] = y_test

        st.write(f"Training set: X_train shape = {X_train.shape}, y_train shape = {y_train.shape}")
        st.write(f"Testing set: X_test shape = {X_test.shape}, y_test shape = {y_test.shape}")


    if st.sidebar.button("Run All Models"):
        st.subheader(" Model Building")

        with st.spinner("Training and evaluating all models..."):
            # Run and compare all models
            results = compare_models(
                st.session_state["X_train"], 
                st.session_state["y_train"], 
                st.session_state["X_test"], 
                st.session_state["y_test"], 
                scaler=st.session_state.get("scaler")
            )
        
        # Display the results as a dataframe in Streamlit
        st.success("All models have been evaluated!")
        
        st.subheader("Model Comparisons")
        st.dataframe(results)

        # Optional: Highlight the best-performing model based on R² or RMSE
        best_model = results.loc[results["R2"].idxmax()]
        best_model_name = best_model['Model']
        best_r2_score = best_model['R2']
        st.markdown(f"""
    <div style='text-align: center;border-radius:20px;background-color:#79e98175'>
        <h3> Best Performing Model: {best_model_name} </h3>
        <h4>R² Score: {best_r2_score:.4f} </h4>
    </div>
    """, unsafe_allow_html=True)

