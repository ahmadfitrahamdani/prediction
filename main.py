# pip install streamlit fbprophet yfinance plotly
import datetime
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

batas_mulai = datetime.date(2014, 1, 1)
START = "2014-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('US Dollar Index Prediction')
st.markdown("""
Aplikasi ini memprediksi nilai dari **US. Dollar Index** dengan rentang waktu **satu hari, satu minggu, satu bulan dan satu tahun ke depan**.
* **Python Libraries:** streamlit, datetime, yfinance, fbprophet, plotly
* **Sumber Data:** [Yahoo Finance](https://finance.yahoo.com/quote/DX-Y.NYB)            
""")

# sidebar 
st.sidebar.header("Pengaturan")

st.sidebar.subheader('Range dataset untuk prediksi')
mulai_dataset = st.sidebar.date_input("Mulai dari", datetime.date(2014, 1, 1))

if mulai_dataset < batas_mulai:
    mulai_dataset = batas_mulai
    
akhir_dataset = st.sidebar.date_input("Sampai dengan", date.today())
ticker = ('DX-Y.NYB')

st.sidebar.subheader('Atribut prediksi')
attribute = st.sidebar.selectbox('Tipe Harga',('Open', 'Close', 'High', 'Low', 'Adj Close'))

range_prediksi = st.sidebar.selectbox('Pilih periode:', ('Sehari', 'Seminggu', 'Sebulan', 'Setahun'))
periode = 0
if range_prediksi == 'sehari':
    periode = 1
    pass
elif range_prediksi == 'seminggu':
    periode = 7
    pass
elif range_prediksi == 'sebulan':
    periode = 30
    pass
else:
    periode = 365
    
#Pengambilan data
@st.cache
def load_data(ticker):
    data = yf.download(ticker, mulai_dataset, akhir_dataset)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(ticker)
data_load_state.text('Loading data... selesai!')

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    if attribute == 'Open':
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="DXY open", line_color='darkcyan'))
        pass
    elif attribute == 'Close':
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="DXY close", line_color='black'))
        pass
    elif attribute == 'High':
        fig.add_trace(go.Scatter(x=data['Date'], y=data['High'], name="DXY high", line_color='deepskyblue'))
        pass
    elif attribute == 'Low':
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Low'], name="DXY low", line_color='brown'))
        pass
    else:
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Low'], name="DXY high", line_color='burlywood'))
        pass
    fig.layout.update(title_text='Historis Indeks US Dollar ', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

st.dataframe(data, height=247, width=800)

# st.button('Prediksi')

if st.button('Prediksi'):
    # Predict forecast with Prophet.
    df_train = data[['Date', attribute]]
    df_train = df_train.rename(columns={"Date": "ds", attribute: "y"})

    model = Prophet(growth='linear', seasonality_prior_scale=10)
    model.fit(df_train)
    future = model.make_future_dataframe(periods=periode)
    forecast = model.predict(future)

    # Show and plot forecast
    st.subheader('Hasil Prediksi')
    forecast['ds'] = pd.to_datetime(forecast['ds'], format='%Y-%m-%d')
    satu_tahun = forecast[['ds', 'yhat']].iloc[-365:]
    satu_tahun.columns = ['tanggal', 'harga_prediksi']
    # Mengubah dataframe sesuai dengan periode yang dipilih
    if range_prediksi == 'Setahun':
        hasil_prediksi = satu_tahun
    elif range_prediksi == 'Sebulan':
        hasil_prediksi = satu_tahun.iloc[:30]
    elif range_prediksi == 'Seminggu':
        hasil_prediksi = satu_tahun.iloc[:7]
    else:
        hasil_prediksi = satu_tahun.iloc[:1]
    st.dataframe(hasil_prediksi, height=247, width=800)


# st.write(forecast.tail())

# st.write(f'Forecast plot for {n_years} years')
# fig1 = plot_plotly(m, forecast)
# st.plotly_chart(fig1)
