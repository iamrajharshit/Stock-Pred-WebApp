import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects as go

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

def plot_raw_data():
    """Will create a plotly graph object!"""
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'], name='Stock_Open'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'], name='Stock_Close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
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


st.write("Forecast Data")
fig_fore=plot_plotly(m,forecast)
st.plotly_chart(fig_fore)

st.write("Forecast Components")
fig_comp=m.plot_components(forecast)
st.write(fig_comp)
