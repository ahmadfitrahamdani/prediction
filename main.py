import datetime
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
from numpy import mean, absolute

import yfinance as yf
from prophet import Prophet
from plotly import graph_objs as go

batas_mulai = datetime.date(2014, 1, 1)
START = "2014-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('US Dollar Index Prediction')
st.markdown("""
Aplikasi ini memprediksi nilai dari **US. Dollar Index** dengan rentang waktu **satu hari, satu minggu, satu bulan dan satu tahun ke depan**.
* **Python Libraries:** streamlit, datetime, yfinance, prophet, plotly
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
attribute = st.sidebar.selectbox('Tipe Nilai',('Open', 'Close', 'High', 'Low', 'Adj Close'))

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
    
chart_height = st.sidebar.slider("Tinggi Grafik", min_value=400, max_value=700, value=500, step=10)

#Pengambilan data
@st.cache
def load_data(ticker):
    data = yf.download(ticker, mulai_dataset, akhir_dataset)
    data.reset_index(inplace=True)
    return data

data = load_data(ticker)

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig = go.Figure(layout=go.Layout(height=chart_height))
    if attribute == 'Open':
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="DXY open", line_color='darkcyan'))
        pass
    elif attribute == 'Close':
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="DXY close", line_color='darkgreen'))
        pass
    elif attribute == 'High':
        fig.add_trace(go.Scatter(x=data['Date'], y=data['High'], name="DXY high", line_color='deepskyblue'))
        pass
    elif attribute == 'Low':
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Low'], name="DXY low", line_color='brown'))
        pass
    else:
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Low'], name="DXY adj close", line_color='burlywood'))
        pass
    fig.layout.update(title_text='Historis Indeks US Dollar ', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()
st.dataframe(data, height=247, width=800)

if st.sidebar.button('Prediksi'):
    # Predict forecast with Prophet.
    df_train = data[['Date', attribute]]
    df_train = df_train.rename(columns={"Date": "ds", attribute: "y"})

    model = Prophet(growth='linear', weekly_seasonality=False, n_changepoints=50 ,seasonality_prior_scale=0.1)
    model.fit(df_train)
    future = model.make_future_dataframe(periods=periode)
    future = future[future['ds'].dt.dayofweek < 5]
    forecast = model.predict(future)

    # Show and plot forecast
    st.subheader('Hasil Prediksi')
    forecast['ds'] = pd.to_datetime(forecast['ds'], format='%Y-%m-%d')
    hasil_prediksi = forecast[['ds', 'yhat']].iloc[-periode:]
    hasil_prediksi.columns = ['tanggal', 'nilai_prediksi']
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hasil_prediksi['tanggal'], y=hasil_prediksi['nilai_prediksi'], name="Hasil Prediksi", line_color='lightblue'))
    st.plotly_chart(fig)
    st.dataframe(hasil_prediksi, height=247, width=800)
    
    #Evaluasi Model
    train_data = df_train.sample(frac=0.8, random_state=0)
    test_data = df_train.drop(train_data.index)
    
    prediction = model.predict(pd.DataFrame({'ds':test_data['ds']}))
    y_test = test_data['y']
    y_pred = prediction['yhat']
    y_pred = y_pred.astype(int)

    def mad(data, axis=None):
        return mean(absolute(data - mean(data, axis)), axis)

    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    st.sidebar.subheader('Evaluasi Model')
    st.sidebar.text(f"MAD:\t{mad(y_pred):.2f}")
    st.sidebar.text(f"MAPE:\t{mean_absolute_percentage_error(y_test, y_pred):.2f}%")

