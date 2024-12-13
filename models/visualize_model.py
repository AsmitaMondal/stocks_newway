#import required libraries
import streamlit as st
import yfinance as yf
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from plotly import graph_objects as go
import warnings
warnings.filterwarnings("ignore")
import cufflinks as cf
import pandas_datareader as pdr


#function calling local css sheet
def local_css(file_name):
    with open(file_name) as f:
        st.sidebar.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

#local css sheet
local_css("style.css")

#main function
def main():
    # Date range selection
    st.sidebar.subheader("Select Date Range")
    start_date = st.sidebar.date_input("Start Date", datetime.datetime(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.datetime.today())
    st.subheader("Choose Your Metrics")
    
    
    # Create horizontal checkboxes using columns
    col1, col2, col3= st.columns(3)

    # Display checkboxes horizontally
    with col1:
        actions = st.checkbox("Stock Actions")
    with col2:
        financials = st.checkbox("Quarterly Financials")
    with col3:
        major_shareholders = st.checkbox("Institutional Shareholders")
    
    
    # Create another row of columns for analyst recommendations
    col4,col5,col6 = st.columns(3)
    with col4:
        cashflow = st.checkbox("Quarterly Cashflow")
    with col5:
        balance_sheet = st.checkbox("Quarterly Balance Sheet")
    with col6:
        analyst_recommendation = st.checkbox("Analysts Recommendation")
    
    st.markdown("<hr>", unsafe_allow_html=True)

    col7,col8=st.columns(2)
    with col7:
        st.subheader(" Daily **Closing Price**")
        #get data on searched ticker
        stock_data = yf.Ticker(selected_stock)
        #get historical data for searched ticker
        stock_df = stock_data.history(period='1d', start=start_date, end=end_date)
        #print line chart with daily closing prices for searched ticker
        st.line_chart(stock_df.Close)
    with col8:
        #get daily volume for searched ticker
        st.subheader("""Daily **Volume**""")
        st.line_chart(stock_df.Volume)

    # st.subheader("""Last **closing price** for """ + selected_stock)
    # #define variable today 
    # yesterday = (datetime.today() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    # #get previous day data for searched ticker
    # stock_lastprice = stock_data.history(period='1d', start=yesterday, end=yesterday)
    # #get previous day closing price for searched ticker
    # last_price = (stock_lastprice.Close)
    # #if market is closed on current date print that there is no data available
    # if last_price.empty == True:
    #     st.write("No data available at the moment")
    # else:
    #     st.write(last_price)
        
    # st.subheader("""Bollinger Bands""")
    # # Display stock data and its candlestick chart
    # st.header(f'{selected_stock} Stock Price')

    # if st.checkbox('Show raw data'):
    #     st.subheader('Raw data')
    #     st.write(stock_df)

    # Interactive data visualizations using cufflinks
    # Create candlestick chart 
    # qf = cf.QuantFig(stock_df, legend='top', name=selected_stock)

    # Technical Analysis Studies can be added on demand
    # Add Relative Strength Indicator (RSI) study to QuantFigure.studies
    # qf.add_rsi(periods=20, color='red')  # Using a valid RGB string

    # Add Bollinger Bands (BOLL) study to QuantFigure.studies
    # qf.add_bollinger_bands(periods=20, boll_std=2, colors=['magenta', 'grey'], fill=True)

    # Add 'volume' study to QuantFigure.studies with default color
    # qf.add_volume()

    # Create the interactive plot
    # fig = qf.iplot(asFigure=True, dimensions=(800, 600))

    # # Render plot using plotly_chart
    # st.plotly_chart(fig)

    
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

        
            
    # Moving Averages
    st.subheader(f"Moving Averages")
    ma_periods = st.multiselect("Select Moving Average Periods (in days):", [10, 20, 50, 100, 200], default=[20, 50])

    # Calculate and plot moving averages
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(
        x=stock_df.index,
        y=stock_df['Close'],
        mode='lines',
        name='Closing Price'
    ))

    for period in ma_periods:
        stock_df[f"MA_{period}"] = stock_df['Close'].rolling(window=period).mean()
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
    
    
    
    st.subheader(f"Closing Price Comparison Metrics")

    # User selects additional stocks to compare
    comparison_tickers = st.multiselect(
        "Select up to 3 additional stocks to compare with the selected ticker:",
        options = [
            "AAPL", "AMZN", "MSFT", "TSLA", "GOOGL", "META", "NFLX", "NVDA", "BABA", "FB",
            "JPM", "V", "WMT", "DIS", "PFE", "BA", "CSCO", "INTC", "IBM", "KO", "MCD", "PYPL",
            "AMGN", "NVDA", "CVX", "XOM", "GS", "UNH", "ORCL", "LLY"
        ],
        default=[],
        label_visibility='hidden',
        help="Choose up to 3 additional stocks for comparison"
    )

    # Restrict the selection to 3 tickers
    if len(comparison_tickers) > 3:
        st.warning("Please select a maximum of 3 stocks.")
    else:
        # Fetch historical data for comparison tickers
        comparison_data = {}
        for ticker in comparison_tickers:
            comparison_data[ticker] = yf.Ticker(ticker).history(period='1y', start=start_date, end=end_date)['Close']

        # Include the primary stock data
        comparison_data['Selected Stock'] = stock_df['Close']

        # Convert to DataFrame
        comparison_df = pd.DataFrame(comparison_data)

        # Display comparison line chart
        if len(comparison_tickers) > 0:
            st.subheader("Stock Price Comparison")
            st.line_chart(comparison_df, use_container_width=True)
        else:
            st.info("Select additional stocks to compare.")

    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Display results based on checkbox selections
    if actions:
        st.subheader(f"Stock **actions** for {selected_stock}")
        display_action = stock_data.actions  # Replace with actual data source
        if display_action.empty:
            st.write("No data available at the moment")
            st.markdown("<hr>", unsafe_allow_html=True)
        else:
            st.write(display_action)
            st.markdown("<hr>", unsafe_allow_html=True)

    if financials:
        st.subheader(f"**Quarterly financials** for {selected_stock}")
        display_financials = stock_data.quarterly_financials  # Replace with actual data source
        if display_financials.empty:
            st.write("No data available at the moment")
            st.markdown("<hr>", unsafe_allow_html=True)
        else:
            st.write(display_financials)
            st.markdown("<hr>", unsafe_allow_html=True)

    if major_shareholders:
        st.subheader(f"**Institutional investors** for {selected_stock}")
        display_shareholders = stock_data.institutional_holders  # Replace with actual data source
        if display_shareholders.empty:
            st.write("No data available at the moment")
            st.markdown("<hr>", unsafe_allow_html=True)
        else:
            st.write(display_shareholders)
            st.markdown("<hr>", unsafe_allow_html=True)

    if balance_sheet:
        st.subheader(f"**Quarterly balance sheet** for {selected_stock}")
        display_balancesheet = stock_data.quarterly_balance_sheet  # Replace with actual data source
        if display_balancesheet.empty:
            st.write("No data available at the moment")
            st.markdown("<hr>", unsafe_allow_html=True)
        else:
            st.write(display_balancesheet)
            st.markdown("<hr>", unsafe_allow_html=True)

    if cashflow:
        st.subheader(f"**Quarterly cashflow** for {selected_stock}")
        display_cashflow = stock_data.quarterly_cashflow  # Replace with actual data source
        if display_cashflow.empty:
            st.write("No data available at the moment")
            st.markdown("<hr>", unsafe_allow_html=True)
        else:
            st.write(display_cashflow)
            st.markdown("<hr>", unsafe_allow_html=True)

    # if earnings:
    #     st.subheader(f"**Quarterly earnings** for {selected_stock}")
    #     display_earnings = stock_data.quarterly_earnings  # Replace with actual data source
    #     if display_earnings.empty:
    #         st.write("No data available at the moment")
    #     else:
    #         st.write(display_earnings)

    if analyst_recommendation:
        st.subheader(f"**Analysts recommendation** for {selected_stock}")
        display_analyst_rec = stock_data.recommendations  # Replace with actual data source
        if display_analyst_rec.empty:
            st.write("No data available at the moment")
            st.markdown("<hr>", unsafe_allow_html=True)
        else:
            st.write(display_analyst_rec)
            st.markdown("<hr>", unsafe_allow_html=True)

#ticker search feature in sidebar
st.sidebar.subheader("""Visualize Your Stocks""")
selected_stock = st.sidebar.text_input("Enter a valid stock ticker...", "GOOG")
button_clicked = st.sidebar.button("GO")
if button_clicked == "GO":
    main()
    
if __name__ == "__main__":
    main()
