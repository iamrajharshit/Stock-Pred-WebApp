import streamlit as st
from datetime import date
import numpy as np
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects as go
import matplotlib.pyplot as plt

START= '2015-01-01'
TODAY= date.today().strftime("%Y-%m-%d")


#WebAPP
st.title("Stock Pred App")

stocks=("AAPL","GOOG","MSFT","GME") #tuple
selected_stock= st.selectbox("Select dataset for pred",stocks)

n_years=st.slider("Years of Pred:",1,4) #start and end time of our year
period= n_years*365 

#Load Stock Data

@st.cache_data #to cache the data which is downloaded already
def load_data(ticker):
    #returns in a pandas data frame yf.download
    data=yf.download(ticker,START,TODAY) #to downlaod data of the given stock from Start to todays date
    data.reset_index(inplace=True) #Dates will be put into very first column
    return data

#to indicate the state
data_load_state = st.text("Load Data...")
#to download the data of our desired stock
data=load_data(selected_stock)
#change of state
data_load_state.text("Loading data...done!")


#analyse
st.subheader("Raw Data")

#pandas tails (last 5)
st.write(data.tail(5))

st.subheader("Histogram")
def histo():
    # Create subplots with 2 rows and 3 columns to accommodate all 6 histograms
    # Calculate skewness for each column
    skew_open = data['Open'].skew()
    skew_high = data['High'].skew()
    skew_low = data['Low'].skew()
    skew_close = data['Close'].skew()
    skew_adj_close = data['Adj Close'].skew()
    skew_volume = data['Volume'].skew()

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    # Plot histograms for each column
    axs[0, 0].hist(data['Open'], bins=20, edgecolor='white')
    axs[0, 0].set_title(f'Open Price\nSkewness: {skew_open:.2f}')

    axs[0, 1].hist(data['High'], bins=20, edgecolor='white')
    axs[0, 1].set_title(f'High Price\nSkewness: {skew_high:.2f}')

    axs[0, 2].hist(data['Low'], bins=20, edgecolor='white')
    axs[0, 2].set_title(f'Low Price\nSkewness: {skew_low:.2f}')

    axs[1, 0].hist(data['Close'], bins=20, edgecolor='white')
    axs[1, 0].set_title(f'Close Price\nSkewness: {skew_close:.2f}')

    axs[1, 1].hist(data['Adj Close'], bins=20, edgecolor='white')
    axs[1, 1].set_title(f'Adjusted Close Price\nSkewness: {skew_adj_close:.2f}')

    axs[1, 2].hist(data['Volume'], bins=20, edgecolor='white')
    axs[1, 2].set_title(f'Volume\nSkewness: {skew_volume:.2f}')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Display the plot
    st.pyplot(fig)

histo()



st.subheader("Time Series Data")
def plot_raw_data():
    """Will create a plotly graph object!"""
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'], name='Stock_Open'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'], name='Stock_Close'))
    #fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()


#Forecasting

df_train=data[['Date','Close']]
df_train=df_train.rename(columns={"Date":"ds","Close":"y"})

m=Prophet()
#to train
m.fit(df_train)

#to forecast
#"""We need a data frame which goes into the future"""
future=m.make_future_dataframe(period) #in number of days

forecast=m.predict(future)
#analyse
st.subheader("Forecast Data")

#pandas tails (last 5)
st.write(forecast.tail(5))


st.subheader("Forecast Data")
fig_fore=plot_plotly(m,forecast)
fig_fore['data'][0]['x'] = np.array(fig_fore['data'][0]['x'])
fig_fore['data'][1]['x'] = np.array(fig_fore['data'][1]['x'])
st.plotly_chart(fig_fore)


st.subheader("Forecast Components")
fig_comp=m.plot_components(forecast)
for ax in fig_comp.get_axes():
    ax.get_lines()[0].set_xdata(np.array(ax.get_lines()[0].get_xdata()))
st.write(fig_comp)
