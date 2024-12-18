__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import yfinance as yf
import datetime
import pandas as pd
import google.generativeai as genai
import matplotlib.pyplot as plt
from plotly import graph_objects as go
import altair as alt
import plotly.graph_objs as go
import numpy as np
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf
from keras import layers
from keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN,Layer # type: ignore
from keras.optimizers import SGD # type: ignore
from keras.models import Sequential # type: ignore
import io
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from crewai import Agent, Task, Crew, Process
from crewai_tools import YoutubeChannelSearchTool, ScrapeWebsiteTool
import warnings
from typing import List
import json
from streamlit_lottie import st_lottie


warnings.filterwarnings("ignore")
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Initialize Groq model
groq_model = ChatGroq(model="groq/llama3-8b-8192", api_key=GROQ_API_KEY)

# Set the page configuration for Streamlit
st.set_page_config(page_title="PredictorX", page_icon="📈", layout="wide")

# Function for calling local CSS file
def local_css(file_name):
    with open(file_name) as f:
        st.sidebar.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Function for sidebar navigation
def sidebar_navigation():
    menu_options = ["About", "Visualize", "Predict", "News","Ticker Guide"]
    choice = st.sidebar.radio("Navigate to", menu_options)
    return choice

# Initialize session state with keys to prevent resets
def initialize_session_state():
    # Initialize only if not already set
    if 'stock_analysis' not in st.session_state:
        st.session_state['stock_analysis'] = {
            'selected_stock': 'GOOG',
            'start_date': datetime.datetime(2020, 1, 1),
            'end_date': datetime.datetime.today(),
            'button_clicked': False,
            'show_actions': False,
            'show_financials': False,
            'show_shareholders': False,
            'show_cashflow': False,
            'show_balance_sheet': False,
            'show_analyst_rec': False
        }

# Initialize session state variables
if "page" not in st.session_state:
    st.session_state["page"] = "About"
if "stock_data" not in st.session_state:
    st.session_state["stock_data"] = None
if "data_scaled" not in st.session_state:
    st.session_state["data_scaled"] = None
if "scaler" not in st.session_state:
    st.session_state["scaler"] = None

def create_dynamic_tools(topic: str) -> List:
    """Create tools tailored to the user's stock topic"""
    try:
        youtube_tool = YoutubeChannelSearchTool(
            youtube_channel_handle='@YahooFinance',
            config={
                'llm': {
                    'provider': 'groq',
                    'config': {
                        'model': 'llama-3.1-8b-instant',
                        'api_key': GROQ_API_KEY,
                        'temperature': 0.7
                    }
                },
                'embedder': {
                    'provider': 'google',
                    'config': {
                        'model': 'models/embedding-001'
                    }
                }
            }
        )
        
        web_search_tool = ScrapeWebsiteTool(
            config={
                'llm': {
                    'provider': 'groq',
                    'config': {
                        'model': 'llama-3.1-8b-instant',
                        'api_key': GROQ_API_KEY,
                        'temperature': 0.7
                    }
                }
            }
        )
        
        return [youtube_tool, web_search_tool]
    except Exception as e:
        st.error(f"Error creating tools: {e}")
        return []

def generate_stock_insights(topic):
    """Generate  but comprehensive stock insights based on user input"""
    # Dynamic tools
    tools = create_dynamic_tools(topic)
    
    # Research Agent
    researcher_agent = Agent(
        role="Stock Market Intelligence Analyst",
        goal=f"Conduct an in-depth investigation and analysis of topic or stock ticker: {topic} in the stock market",
        backstory=f"You are a seasoned financial analyst specializing in researching and providing comprehensive insights about {topic}. "
                  "Your expertise lies in gathering the most recent and relevant information from multiple sources.",
        tools=tools,
        allow_delegation=True,
        llm=groq_model,
        verbose=True
    )
    
    # Insight Compiler Agent
    insights_agent = Agent(
        role="Financial Content Strategist",
        goal="Transform raw financial data into a , coherent, engaging narrative",
        backstory="You excel at converting complex financial information into , clear, actionable insights including tables and metrics. "
                  "Your writing style is professional yet accessible, making complex topics easy to understand.",
        llm=groq_model,
        allow_delegation=False,
        verbose=True
    )
    
    # Report Formatter Agent
    formatter_agent = Agent(
    role="Professional Report Designer",
    goal="Create a visually structured and professionally formatted  financial report with  tables, data, headings and subheadings if necessary but without placeholders, [] page breaks, and images",
    backstory="You specialize in presenting financial information in a clean, organized manner with tables, and bullets if necessary. "
              "Ensure no placeholder text or unnecesary page breaks or [] remains in the final report.",
    tools=[],
    llm=groq_model,
    allow_delegation=False,
    verbose=True,
    config={
        "remove_placeholders": True,
        "strict_formatting": True
    }
)
    
    # Research Task
    research_task = Task(
        description=f"""
        Conduct comprehensive research on {topic} which is either by a proper topic or a stock ticker:
        - Latest news and developments
        - Tables and metrics
        - Market sentiment
        - Recent performance trends
        - Key financial indicators
        - Potential future outlook
        
        Ensure the information is current, accurate, and provides meaningful insights.
        """,
        expected_output="Detailed raw research findings about the specified stock or topic",
        agent=researcher_agent,
        output_format="Markdown"
    )
    
    # Insight Compilation Task
    insights_task = Task(
        description="Transform the raw research into a  but compelling narrative. "
                    "Create a story that provides context, analysis, and actionable insights  with data, headings and subheadings including tables, bullets and metrics if necessary.",
        expected_output="A well-structured, and insightful financial narrative",
        agent=insights_agent,
        context=[research_task],
        output_format="Markdown"
    )
    
    # Formatting Task
    format_task = Task(
        description="Refine and format the financial narrative into a  professional report with tables, bullets and metrics if necessary. "
                    "Ensure crispness, , clarity, readability, and professional presentation. No placeholders or unnecessary images , brackets or page breaks should be present",
        expected_output="A polished, professionally formatted  financial insights report with data, headings and subheadings.",
        agent=formatter_agent,
        context=[insights_task],
        output_format="Markdown"
    )
    
    # Create Crew
    crew = Crew(
        agents=[researcher_agent, insights_agent, formatter_agent],
        tasks=[research_task, insights_task, format_task],
        process=Process.sequential,
        verbose=True
    )
    
    # Generate Insights
    result = crew.kickoff(inputs={"topic": topic})
    return result
   
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
def preprocess_data(data,ticker):
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
def evaluate_and_plot(model, X_train, y_train, X_test, y_test,epochs, scaler=None):
    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=64, validation_data=(X_test, y_test), verbose=1)    
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

def compare_models(X_train, y_train, X_test, y_test,epochs, scaler=None):
    # Initialize a DataFrame to store the results
    results_list = []

    # 1. Normal LSTM
    model_lstm = build_normal_lstm_model((X_train.shape[1], 1))
    print("Training VANILLA LSTM...")
    st.write("Training VANILLA LSTM...")
    mse, mae, rmse, r2 = evaluate_and_plot(model_lstm, X_train, y_train, X_test, y_test,epochs, scaler)
    results_list.append({'Model': 'VANILLA LSTM', 'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2}) # append to list
    
    # 2. Improved LSTM
    model_improved_lstm = build_improved_lstm_model((X_train.shape[1], 1))
    print("Training STACKED LSTM...")
    st.write("Training STACKED LSTM...")
    mse, mae, rmse, r2 = evaluate_and_plot(model_improved_lstm, X_train, y_train, X_test, y_test,epochs, scaler)
    results_list.append({'Model': 'STACKED LSTM', 'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2}) # append to list
    
    # 3. Normal GRU
    model_gru = build_normal_gru_model((X_train.shape[1], 1))
    print("Training GRU...")
    st.write("Training GRU...")
    mse, mae, rmse, r2 = evaluate_and_plot(model_gru, X_train, y_train, X_test, y_test,epochs, scaler)
    results_list.append({'Model': ' GRU', 'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2}) # append to list
    
    # 4. RNN Model
    model_hybrid = build_momentum_rnn((X_train.shape[1], 1))
    print("Training MOMENTUM RNN...")
    st.write("Training MOMENTUM RNN...")
    mse, mae, rmse, r2 = evaluate_and_plot(model_hybrid, X_train, y_train, X_test, y_test,epochs, scaler)
    results_list.append({'Model': 'MOMENTUM RNN', 'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2}) # append to list

    # 5. Mogrifier LSTM Model
    model_mogrifier = build_mogrifier_lstm_model((X_train.shape[1], 1))
    print("Training MOGRIFIER LSTM...")
    st.write("Training MOGRIFIER LSTM...")
    mse, mae, rmse, r2 = evaluate_and_plot(model_mogrifier, X_train, y_train, X_test, y_test,epochs, scaler)
    results_list.append({'Model': 'MOGRIFIER LSTM', 'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2}) # append to list

    results = pd.DataFrame(results_list)

    
    # Display the comparison table
    print(results)
    return pd.DataFrame(results_list)
       
# Visualize function where the actual chart is displayed
def show_visualize():
    # Ensure session state is initialized
    initialize_session_state()
    
    
    # Use session state to persist stock selection
    selected_stock = st.sidebar.text_input(
        "Enter a valid stock ticker...", 
        st.session_state['stock_analysis']['selected_stock'],
        help="Provide a specific stock ticker. Refer to our Ticker Guide if needed."
    )
    
    # Update selected stock in session state
    st.session_state['stock_analysis']['selected_stock'] = selected_stock

    # Button to trigger data fetch
    button_clicked = st.sidebar.button("🏃‍♀️‍➡️ GO")
    
    # Date range selection
    st.sidebar.subheader("Select Date Range")
    start_date = st.sidebar.date_input(
        "Start Date", 
        st.session_state['stock_analysis']['start_date']
    )
    end_date = st.sidebar.date_input(
        "End Date", 
        st.session_state['stock_analysis']['end_date']
    )

    # Update dates in session state
    st.session_state['stock_analysis']['start_date'] = start_date
    st.session_state['stock_analysis']['end_date'] = end_date

    # Visualization section
    st.subheader("Choose MORE")

    # Create columns for checkboxes
    col1, col2, col3 = st.columns(3)

    with col1:
        actions = st.checkbox(
            "Stock Actions", 
            key='show_actions', 
            value=st.session_state['stock_analysis']['show_actions']
        )
        st.session_state['stock_analysis']['show_actions'] = actions
        
    with col2:
        financials = st.checkbox(
            "Quarterly Financials", 
            key='show_financials', 
            value=st.session_state['stock_analysis']['show_financials']
        )
        st.session_state['stock_analysis']['show_financials'] = financials

    with col3:
        major_shareholders = st.checkbox(
            "Institutional Shareholders", 
            key='show_shareholders', 
            value=st.session_state['stock_analysis']['show_shareholders']
        )
        st.session_state['stock_analysis']['show_shareholders'] = major_shareholders

    col4, col5, col6 = st.columns(3)
    with col4:
        cashflow = st.checkbox(
            "Quarterly Cashflow", 
            key='show_cashflow', 
            value=st.session_state['stock_analysis']['show_cashflow']
        )
        st.session_state['stock_analysis']['show_cashflow'] = cashflow
        
    with col5:
        balance_sheet = st.checkbox(
            "Quarterly Balance Sheet", 
            key='show_balance_sheet', 
            value=st.session_state['stock_analysis']['show_balance_sheet']
        )
        st.session_state['stock_analysis']['show_balance_sheet'] = balance_sheet
        
    with col6:
        analyst_recommendation = st.checkbox(
            "Analysts Recommendation", 
            key='show_analyst_rec', 
            value=st.session_state['stock_analysis']['show_analyst_rec']
        )
        st.session_state['stock_analysis']['show_analyst_rec'] = analyst_recommendation
    
    st.markdown("<hr>", unsafe_allow_html=True)

    # Check if GO button is clicked or stock is selected
    if button_clicked or st.session_state['stock_analysis']['selected_stock']:
        try:
            # Download stock data
            stock_data = yf.Ticker(selected_stock)
            stock_df = stock_data.history(
                period='1d', 
                start=start_date, 
                end=end_date
            )

            # Price and Volume Charts
            col7, col8 = st.columns(2)
            
            with col7:
                st.subheader("Daily **Closing Price**")
                if not stock_df.empty:
                    chart = alt.Chart(stock_df.reset_index()).mark_line(color='maroon').encode(
                        x='Date:T',
                        y='Close:Q'
                    ).properties(
                        title=f'{selected_stock} Close Price Over Time'
                    )
                    st.write(chart)
                else:
                    st.warning("No price data available")

            with col8:
                st.subheader("Daily **Volume**")
                if not stock_df.empty:
                    chart = alt.Chart(stock_df.reset_index()).mark_line(color='maroon').encode(
                        x='Date:T',
                        y='Volume:Q'
                    ).properties(
                        title=f'{selected_stock} Volume Over Time'
                    )
                    st.write(chart)
                else:
                    st.warning("No volume data available")
            # Fetch stock data
            data = yf.download(selected_stock, start=start_date, end=end_date)

            # Calculate the 20-day simple moving average (SMA)
            data['SMA'] = data['Close'].rolling(window=20).mean()

            # Calculate the 20-day standard deviation
            data['std_dev'] = data['Close'].rolling(window=20).std()

            # Calculate the upper and lower Bollinger Bands
            data['Upper Band'] = data['SMA'] + (data['std_dev'] * 2)
            data['Lower Band'] = data['SMA'] - (data['std_dev'] * 2)

            # Plot the Bollinger Bands
            plt.figure(figsize=(10, 6))
            plt.plot(data['Close'], label="Close Price", color='blue')
            plt.plot(data['SMA'], label="20-Day SMA", color='orange')
            plt.plot(data['Upper Band'], label="Upper Bollinger Band", color='green')
            plt.plot(data['Lower Band'], label="Lower Bollinger Band", color='red')

            # Add title and labels
            plt.title(f'{selected_stock} Bollinger Bands ({start_date} to {end_date})')
            plt.xlabel("Date")
            plt.ylabel("Price (USD)")

            # Show legend
            plt.legend()

            # Display the plot in Streamlit
            st.pyplot(plt)

            st.subheader(f"Key Statistics for {selected_stock}")

            col9,col10=st.columns(2)
            daily_returns = stock_df['Close'].pct_change().dropna()
            avg_daily_return = daily_returns.mean() * 100
            annual_volatility = daily_returns.std() * (252 ** 0.5) * 100
            sharpe_ratio = avg_daily_return / (annual_volatility / 100)

            # Calculate Maximum Drawdown
            rolling_max = stock_df['Close'].cummax()
            drawdown = (stock_df['Close'] - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            with col9:
                st.write(f"""
                - **Average Daily Return**: {avg_daily_return:.2f}%
                - **Annualized Volatility**: {annual_volatility:.2f}%
                """)
            with col10:
                st.write(f"""
                - **Sharpe Ratio**: {sharpe_ratio:.2f}
                - **Maximum Drawdown**: {max_drawdown:.2f}%
                """)
            
            col11,col12=st.columns(2)
            with col11:
                # Combined cool graph
                st.subheader("Overall Visual")
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=stock_df.index,
                    open=stock_df['Open'],
                    high=stock_df['High'],
                    low=stock_df['Low'],
                    close=stock_df['Close'],
                    name='Candlestick'))
                fig.add_trace(go.Scatter(
                    x=stock_df.index,
                    y=stock_df['Close'],
                    mode='lines',
                    name='Closing Price',
                    visible='legendonly'))
                
                fig.add_trace(go.Bar(
                    x=stock_df.index,
                    y=stock_df['Volume'],
                    name='Volume',
                    marker=dict(color='rgba(255, 165, 0, 0.6)'),
                    visible='legendonly'
                ))
                fig.update_layout(
                    title="Combined Stock Data Visualization",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    template="plotly_dark",
                    yaxis2=dict(
                        title="Volume",
                        overlaying="y",
                        side="right"
                    )
                )
                st.plotly_chart(fig)
            with col12:
                st.subheader(f"Dividend History")
                dividends = stock_data.dividends
                if dividends.empty:
                    st.write("No dividend data available.")
                else:
                    st.write("**Latest Dividend**:", dividends[-1])
                    st.line_chart(dividends)
            
                
            col13,col14=st.columns(2)
            with col13:
                st.subheader(f"Earnings and Valuation")
                eps = stock_data.info.get("trailingEps", "N/A")
                pe_ratio = stock_data.info.get("trailingPE", "N/A")
                st.write(f"""
                - **Earnings Per Share (EPS)**: {eps}
                - **Price-to-Earnings (PE) Ratio**: {pe_ratio}
                """)
            with col14:
                st.subheader(f"Risk Metrics")
                beta = stock_data.info.get("beta", "N/A")
                # Value at Risk (VaR) - Historical Simulation
                VaR_95 = daily_returns.quantile(0.05) * 100
                st.write(f"""
                - **Beta**: {beta}
                - **95% Value at Risk (VaR)**: {VaR_95:.2f}%
                """)
                
            # Add Moving Averages Section
            if 'moving_average_periods' not in st.session_state['stock_analysis']:
                st.session_state['stock_analysis']['moving_average_periods'] = [20, 50]

            st.subheader(f"Moving Averages")
            
            # Use session state to persist multiselect
            ma_periods = st.multiselect(
                "Select Moving Average Periods (in days):", 
                [10, 20, 50, 100, 200], 
                default=[20],
                key='ma_multiselect'
            )

            # Update session state with current selection
            st.session_state['stock_analysis']['moving_average_periods'] = ma_periods

            # Calculate and plot moving averages
            fig_ma = go.Figure()
            fig_ma.add_trace(go.Scatter(
                x=stock_df.index,
                y=stock_df['Close'],
                mode='lines',
                name='Closing Price'
            ))

            for period in ma_periods:
                # Calculate moving average
                stock_df[f"MA_{period}"] = stock_df['Close'].rolling(window=period).mean()
                
                # Add to figure
                fig_ma.add_trace(go.Scatter(
                    x=stock_df.index,
                    y=stock_df[f"MA_{period}"],
                    mode='lines',
                    name=f"{period}-day MA"
                ))

            fig_ma.update_layout(
                title=f"Moving Averages for {selected_stock}",
                xaxis_title="Date",
                yaxis_title="Price",
                template="plotly_dark"
            )
            
            st.plotly_chart(fig_ma)

            # Add Stock Comparison Section
            if 'comparison_tickers' not in st.session_state['stock_analysis']:
                st.session_state['stock_analysis']['comparison_tickers'] = []

            st.subheader("Closing Price Comparison Metrics")

            # Predefined list of stocks
            stock_options = [
                "AAPL", "AMZN", "MSFT", "TSLA", "GOOG", "META", "NFLX", "NVDA", "BABA", "FB",
                "JPM", "V", "WMT", "DIS", "PFE", "BA", "CSCO", "INTC", "IBM", "KO", "MCD", "PYPL",
                "AMGN", "NVDA", "CVX", "XOM", "GS", "UNH", "ORCL", "LLY"
            ]

            # Multiselect with persistent state
            comparison_tickers = st.multiselect(
                "Select up to 3 additional stocks to compare with the selected ticker:",
                options=stock_options,
                default=[],
                max_selections=3,
                key='comparison_multiselect',
                help="Choose up to 3 additional stocks for comparison"
            )

            # Update session state
            st.session_state['stock_analysis']['comparison_tickers'] = comparison_tickers

            # Fetch and compare stock data
            if comparison_tickers:
                try:
                    # Fetch historical data for comparison tickers
                    comparison_data = {}
                    for ticker in comparison_tickers:
                        comparison_data[ticker] = yf.Ticker(ticker).history(
                            period='1y', 
                            start=start_date, 
                            end=end_date
                        )['Close']

                    # Include the primary stock data
                    comparison_data['Selected Stock'] = stock_df['Close']

                    # Convert to DataFrame
                    comparison_df = pd.DataFrame(comparison_data)

                    # Display comparison line chart
                    st.subheader("Stock Price Comparison")
                    st.line_chart(comparison_df, use_container_width=True)

                except Exception as e:
                    st.error(f"Error fetching comparison data: {e}")
            else:
                st.info("Select additional stocks to compare.")
            
            st.subheader(f"Technical Indicators for {selected_stock}")
            if stock_df['Close'][-1] > stock_df['Close'].mean():
                    st.write("The stock is currently in an **uptrend** (above its average price).")
            else:
                    st.write("The stock is currently in a **downtrend** (below its average price).")
            # Columns for layout
            col15, col16 = st.columns(2)

            # Calculate SMA (20-day and 50-day)
            stock_df['SMA_20'] = stock_df['Close'].rolling(window=20).mean()
            stock_df['SMA_50'] = stock_df['Close'].rolling(window=50).mean()

            # Calculate EMA (20-day and 50-day)
            stock_df['EMA_20'] = stock_df['Close'].ewm(span=20, adjust=False).mean()
            stock_df['EMA_50'] = stock_df['Close'].ewm(span=50, adjust=False).mean()

            # Calculate MACD and Signal Line
            stock_df['EMA_12'] = stock_df['Close'].ewm(span=12, adjust=False).mean()
            stock_df['EMA_26'] = stock_df['Close'].ewm(span=26, adjust=False).mean()
            stock_df['MACD'] = stock_df['EMA_12'] - stock_df['EMA_26']
            stock_df['Signal_Line'] = stock_df['MACD'].ewm(span=9, adjust=False).mean()

            # Extract latest values
            latest_sma20 = stock_df['SMA_20'].iloc[-1]
            latest_sma50 = stock_df['SMA_50'].iloc[-1]
            latest_ema20 = stock_df['EMA_20'].iloc[-1]
            latest_ema50 = stock_df['EMA_50'].iloc[-1]
            latest_macd = stock_df['MACD'].iloc[-1]
            latest_signal = stock_df['Signal_Line'].iloc[-1]

            # Display in columns
            with col15:
                st.write(f"""
                - **SMA (20-Day)**: {latest_sma20:.2f}
                - **SMA (50-Day)**: {latest_sma50:.2f}
                - **EMA (20-Day)**: {latest_ema20:.2f}
                """)
            with col16:
                st.write(f"""
                - **EMA (50-Day)**: {latest_ema50:.2f}
                - **MACD**: {latest_macd:.2f}
                - **Signal Line**: {latest_signal:.2f}
                """)

            # RSI Analysis
            st.subheader(f"RSI Analysis")
            rsi_period = st.slider("Select RSI Period:", min_value=7, max_value=28, value=14)

            # Calculate RSI
            delta = stock_df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            stock_df['RSI'] = 100 - (100 / (1 + rs))

            # Plot RSI
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=stock_df.index,
                y=stock_df['RSI'],
                mode='lines',
                name='RSI'
            ))
            fig_rsi.add_hline(y=70, line_dash="dot", annotation_text="Overbought", annotation_position="bottom right", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dot", annotation_text="Oversold", annotation_position="top right", line_color="green")

            fig_rsi.update_layout(
                title=f"RSI for {selected_stock}",
                xaxis_title="Date",
                yaxis_title="RSI",
                template="plotly_dark"
            )
            st.plotly_chart(fig_rsi)
            st.markdown("<hr>", unsafe_allow_html=True)

            # Conditionally display additional information based on checkboxes
            display_additional_info(stock_data, selected_stock)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error("Please check the stock ticker and try again.")

def display_additional_info(stock_data, selected_stock):
    # Stock Actions
    if st.session_state['stock_analysis']['show_actions']:
        st.subheader(f"Stock **Actions** for {selected_stock}")
        display_action = stock_data.actions
        if not display_action.empty:
            st.write(display_action)
        else:
            st.write("No actions data available")
        st.markdown("<hr>", unsafe_allow_html=True)

    # Quarterly Financials
    if st.session_state['stock_analysis']['show_financials']:
        st.subheader(f"**Quarterly Financials** for {selected_stock}")
        display_financials = stock_data.quarterly_financials
        if not display_financials.empty:
            st.write(display_financials)
        else:
            st.write("No financials data available")
        st.markdown("<hr>", unsafe_allow_html=True)

    # Institutional Shareholders
    if st.session_state['stock_analysis']['show_shareholders']:
        st.subheader(f"**Institutional Shareholders** for {selected_stock}")
        display_shareholders = stock_data.institutional_holders
        if not display_shareholders.empty:
            st.write(display_shareholders)
        else:
            st.write("No shareholders data available")
        st.markdown("<hr>", unsafe_allow_html=True)

    # Quarterly Balance Sheet
    if st.session_state['stock_analysis']['show_balance_sheet']:
        st.subheader(f"**Quarterly Balance Sheet** for {selected_stock}")
        display_balancesheet = stock_data.quarterly_balance_sheet
        if not display_balancesheet.empty:
            st.write(display_balancesheet)
        else:
            st.write("No balance sheet data available")
        st.markdown("<hr>", unsafe_allow_html=True)

    # Quarterly Cashflow
    if st.session_state['stock_analysis']['show_cashflow']:
        st.subheader(f"**Quarterly Cashflow** for {selected_stock}")
        display_cashflow = stock_data.quarterly_cashflow
        if not display_cashflow.empty:
            st.write(display_cashflow)
        else:
            st.write("No cashflow data available")
        st.markdown("<hr>", unsafe_allow_html=True)

    # Analysts Recommendation
    if st.session_state['stock_analysis']['show_analyst_rec']:
        st.subheader(f"**Analysts Recommendation** for {selected_stock}")
        display_analyst_rec = stock_data.recommendations
        if not display_analyst_rec.empty:
            st.write(display_analyst_rec)
        else:
            st.write("No analyst recommendations available")
        st.markdown("<hr>", unsafe_allow_html=True)

st.sidebar.image("images/stocks.jpg", use_container_width=True)

# Main function to control app flow
def main():
    
   
    choice = sidebar_navigation()
    if choice == "About":
        st.title("📊 About Our Stock Analysis Tool ")
        with open("images/stocks.json", "r") as f:
            lottie_animation = json.load(f)
        col1, col2 = st.columns([1, 1])  # Adjust proportions as needed

        # Display Lottie animation in one column
        with col1:
            st_lottie(lottie_animation, height=300, key="lottie")

        # Display image in the other column
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Adds two line spaces

            st.markdown("""
            ### **Welcome to this crazy shop of stock comprehension!** 🚀
            We're here to present before you a tiny effort towards easement of understanding stocks. 
            This project offers a suite of features made to help you understand and analyze 
            stock data better than us 😛.<br><br>
            <i>Stocks, a dance of rise and fall,</i> <br>
            <i>Dreams of fortune or the market's call,</i> <br>
            <i>In the gamble of time, we risk it all.</i>
            """, unsafe_allow_html=True)
           
        st.markdown("""
            

            ## **What We Have Built:**

            #### 1. **Visualize 📉:** 
                * Dive deep into historical data for any company of your choice.
                * Uncover trends, key metrics, and patterns with interactive visualizations.
                * Compare performance against industry peers.

            #### 2. **Predict 🔮:** 
                * Forecast future stock prices using machine learning models.
                * Train on a variety of models, including *Vanilla LSTM, Stacked LSTM, GRU, Momentum RNN* and *Mogrifier LSTM*.
                * Evaluate model performances in real time.

            #### 3. **News 📰:** 
                * Stay up-to-date with the latest financial news and insights.
                * Access curated news articles from reputable sources including *Yahoo Finance Youtube Channel* and *The Economic Times*.
                * Discover perceptions from expert analysis and commentary. """)
        st.markdown("<br>", unsafe_allow_html=True)  # Adds two line spaces

        st.markdown(
            """
            <h3 style="text-align: center;">So, with all set onboard, here we go! 🤝</h3>
            """, 
            unsafe_allow_html=True
        )        

        # Consider adding a team section or contact information.
        st.subheader("The Creators")
        st.markdown("""
            * **Asmita Roy Mondal - 2348018** 
            * **Sayan Pal - 2348056**
        """)

        st.subheader("Contact Us")
        st.markdown("""
            * **Asmita-Email:** asmita.mondal@msds.christuniversity.in
            * **Sayan-Email:** sayan.pal@msds.christuniversity.in
            * **GitHub:** [GitHub Project Repository]
        """)
    
    if choice == "Visualize":
        st.session_state["page"] = "Visualize"
    
        st.title("💹 Visualize Your Stocks")  
              
        st.markdown("""
            Choose your company of interest, and visualize stock trends from and to your dates of interests!
        """)
        show_visualize()
    
    # Predict Page
    if choice == "Predict":
        st.session_state["page"] = "Predict"
        st.title("💷 Build Your Stock Model")
        st.markdown("""
            Choose your company of interest, view a 5 year historical data and
            watch our models build to give accurate predictions!
        """)
        # Sidebar for user input
        st.sidebar.title("Stock Data Fetcher")
        ticker = st.sidebar.text_input(
            "Enter stock ticker (e.g., META):", "META",            
            help="Provide a specific stock ticker. Refer to our Ticker Guide if needed."
        )
        end_date = datetime.datetime.today().date()
        start_date = end_date - datetime.timedelta(days=5 * 365)
        options = [2, 5, 10, 20, 50, 100, 150, 200]
        selected_option = st.sidebar.selectbox("Select number of Epochs", options)
        if st.sidebar.button("🗃️ Fetch Data"):
            with st.spinner("Getting Your Data..."):

                # Fetch stock data
                data = fetch_meta_stock_data(start_date, end_date, ticker)
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [' '.join(col).strip() for col in data.columns.values]
                if not data.empty:
                    # Preprocess the data
                    data, data_scaled, scaler = preprocess_data(data,ticker)

                    # Save to session state
                    st.session_state["stock_data"] = data
                    st.session_state["data_scaled"] = data_scaled
                    st.session_state["scaler"] = scaler

                    st.success("Data successfully fetched and preprocessed!")

        # Display the data and interactive plot if available
        if st.session_state["stock_data"] is not None:
            st.subheader(f"Stock Data for {ticker}")
            st.dataframe(st.session_state["stock_data"].tail())  # Show the last 5 rows

            st.subheader("Company Stock Data")
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
            st.session_state["selected_option"]=selected_option
            st.write(f"Training set: X_train shape = {X_train.shape}, y_train shape = {y_train.shape}")
            st.write(f"Testing set: X_test shape = {X_test.shape}, y_test shape = {y_test.shape}")


            if st.button("Run All Models"):
                st.subheader(" Model Building")

                with st.spinner("Training and evaluating all models..."):
                    # Run and compare all models
                    results = compare_models(
                        st.session_state["X_train"], 
                        st.session_state["y_train"], 
                        st.session_state["X_test"], 
                        st.session_state["y_test"],
                        st.session_state["selected_option"],
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
            
    if choice=="News":
        st.session_state["page"] = "News"

        st.title("🚀 Generate Stock Insights")
        st.markdown("""
            Dive deep into personalized stock market insights. 
            Enter a company, stock symbol, or investment topic to get a comprehensive analysis.
        """)
        st.sidebar.title("Real Time News Analysis")

        # Input area for stock topic
        stock_topic = st.sidebar.text_input(
            "Enter Company Name, Stock Symbol, or Investment Topic",
            placeholder="e.g., Apple, AAPL, AI Stocks, Tech Sector",
            help="Provide a specific stock, company, or investment theme for targeted insights"
        )
        if st.sidebar.button("🔍 Generate Stock Insights"):
            if stock_topic:
                with st.spinner("Generating Comprehensive Stock Insights..."):
                    result = generate_stock_insights(stock_topic)
                
                st.success("✅ Insights Generated Successfully!")
                st.markdown("## 📈 Here are the Key Findings")
                st.markdown(result)
                
                # Download Option
                st.download_button(
                    label="📥 Download Stock Insights Report",
                    data=str(result),
                    file_name=f"{stock_topic.replace(' ', '_')}_stock_insights.md",
                    mime="text/plain"
                )
            else:
                st.error("Please enter a stock topic or company name to generate insights.")

    if choice == "Ticker Guide":
        st.session_state["page"] = "Ticker Guide"

        st.title("📈 Get To Know Your Stock Tickers")  
        st.markdown("<br>", unsafe_allow_html=True)  # Adds two line spaces

        st.markdown("""
            <font size='4'>
                A <b>ticker symbol</b> (or stock symbol) is a unique identifier used to represent a specific publicly traded company's stock on a stock exchange. It is typically made up of letters, numbers, or a combination of both. The ticker symbol helps investors and traders quickly identify and track the stock. Here are some popular companies that you might like. Feel free to access the internet for more company tickers!
            </font>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)  # Adds two line spaces

        # Create two columns to display the ticker symbols in two groups of 20
        col1, col2 = st.columns(2)

        # Ticker symbols and their full company names
        tickers = [
            ("AAPL", "Apple Inc. 🍏"), ("GOOG", "Alphabet Inc. (Google) 🧑‍💻"), ("AMZN", "Amazon.com, Inc. 📦"), 
            ("MSFT", "Microsoft Corporation 💻"), ("TSLA", "Tesla Inc. 🚗"), ("FB", "Meta Platforms, Inc. (Facebook) 📱"),
            ("NVDA", "NVIDIA Corporation 🎮"), ("DIS", "The Walt Disney Company 🎬"), ("NFLX", "Netflix, Inc. 📺"),
            ("BA", "The Boeing Company ✈️"), ("WMT", "Walmart Inc. 🛒"), ("JNJ", "Johnson & Johnson 💊"), 
            ("V", "Visa Inc. 💳"), ("MA", "Mastercard Incorporated 💳"), ("CSCO", "Cisco Systems, Inc. 🌐"),
            ("PYPL", "PayPal Holdings, Inc. 💰"), ("NVDA", "NVIDIA Corporation 🎮"), ("KO", "The Coca-Cola Company 🥤"),
            ("PEP", "PepsiCo, Inc. 🥤"), ("INTC", "Intel Corporation 💻"), ("GM", "General Motors Company 🚗"),
            ("AMD", "Advanced Micro Devices, Inc. 🔧"), ("ORCL", "Oracle Corporation 🖥️"), ("IBM", "International Business Machines Corporation 💼"),
            ("GE", "General Electric Company 🔧"), ("T", "AT&T Inc. 📱"), ("PFE", "Pfizer Inc. 💉"),
            ("GE", "General Electric Company 🏭"), ("XOM", "Exxon Mobil Corporation ⛽"), ("CVX", "Chevron Corporation ⛽"),
            ("BP", "BP p.l.c. 🌍"), ("WFC", "Wells Fargo & Company 🏦"), ("GS", "Goldman Sachs Group, Inc. 💰"),
            ("JPM", "JPMorgan Chase & Co. 🏦"), ("HSBC", "HSBC Holdings plc 🌍"), ("BAC", "Bank of America Corporation 💳"),
            ("TGT", "Target Corporation 🎯"), ("MCD", "McDonald's Corporation 🍔"), ("SBUX", "Starbucks Corporation ☕"),
            ("NKE", "Nike, Inc. 👟"), ("VZ", "Verizon Communications Inc. 📱"),("LYFT", "Lyft, Inc. 🚗")
        ]
        additional_tickers = [
            ("TSM", "Taiwan Semiconductor Manufacturing Company 🏭"), 
            ("SHEL", "Shell plc ⛽"), 
            ("MELI", "MercadoLibre, Inc. 🛒"), 
            ("SHOP", "Shopify Inc. 🛍️"),
            ("BABA", "Alibaba Group Holding Limited 🏯"), 
            ("LULU", "Lululemon Athletica Inc. 🧘‍♀️"), 
            ("RTX", "Raytheon Technologies Corporation 🛰️"),
            ("SNAP", "Snap Inc. 📸"),
            ("TWTR", "X (formerly Twitter) 🐦"),
            ("SPOT", "Spotify Technology S.A. 🎵")
        ]
        tickers.extend(additional_tickers)

        # Display the first table with tickers and their company names
        col1, col2 = st.columns(2)
        # Split the tickers into two sets of 20
        tickers_col1 = tickers[:26]
        tickers_col2 = tickers[26:]
        with col1:
            ticker_data_col1 = {"Symbol": [ticker[0] for ticker in tickers_col1], 
                                "Name": [ticker[1] for ticker in tickers_col1]}
            st.table(ticker_data_col1)

        with col2:
            ticker_data_col2 = {"Symbol": [ticker[0] for ticker in tickers_col2], 
                                "Name": [ticker[1] for ticker in tickers_col2]}
            st.table(ticker_data_col2)


if __name__ == "__main__":
    main()
