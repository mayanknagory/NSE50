import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from nselib import capital_market
import pickle
from sklearn.preprocessing import MinMaxScaler
from datetime import date, datetime, timedelta
import warnings
warnings.filterwarnings("ignore")
import requests
import joblib
from io import BytesIO

def fun(equity):
  ndf = pd.DataFrame(capital_market.price_volume_data(symbol=equity, from_date=str((datetime.today()- timedelta(days=366)).strftime('%d-%m-%Y')), to_date= str(datetime.today().strftime('%d-%m-%Y'))))
  df = ndf[ndf['Series'] == 'EQ']
  df['HighPrice'] = ',' + df['HighPrice'].astype(str)
  df['LowPrice'] = ',' + df['LowPrice'].astype(str)
  df['AveragePrice'] = ',' + df['AveragePrice'].astype(str)
  df['HighPrice'] = (df['HighPrice'].str.replace(r',', '', regex=True)).astype('float')
  df['LowPrice'] = (df['LowPrice'].str.replace(r',', '', regex=True)).astype('float')
  df['AveragePrice'] = (df['AveragePrice'].str.replace(r',', '', regex=True)).astype('float')
  #df = df[['Date',	'HighPrice',	'LowPrice',	'AveragePrice']]
  ddf = df[-10:]
  ddf['Date'] = pd.to_datetime(ddf['Date'])
  ddf['Date'] = ddf['Date'].dt.strftime('%d-%b')
  fig, ax = plt.subplots(figsize=(30, 15))
  ax.plot(ddf['Date'], ddf['AveragePrice'], color='blue',linewidth =3, label='Close Price')
  ax.set_title('Average price for the last 10 trading days',fontweight ='bold',fontsize =20)
  ax.tick_params(labelsize=20)
  ax.grid()
  ax.set_xlabel('Date',fontweight ='bold',fontsize =20)
  ax.set_ylabel(('Price(\u20B9)'),fontweight ='bold',fontsize =20)
  m =np.round(((ddf['AveragePrice'].pct_change())*100).mean() ,2)  
  f = ddf[['Date', 'HighPrice',	'LowPrice']]
  f.index = f['Date']
  f = f.drop('Date', axis=1)
  ten= f.transpose()
  
  p = pickle.load(requests.get(f'https://github.com/mayanknagory/NSE50/blob/main/model_l/l_{equity}.pickle').content)
  #p = pickle.load(open (r,'rb'))
  #p = pickle.load(open(f'https://github.com/mayanknagory/NSE50/blob/main/model_l/l_{equity}.pickle', 'rb'))
  lp = df[['LowPrice']].tail(1).values
  mm = MinMaxScaler(feature_range=(0,1))
  sh = mm.fit_transform(df[['LowPrice']])
  xdata = []
  ydata = []
  for j in range(10, len(sh)):
    xdata.append(sh[j-10:j])
    ydata.append(sh[j])
  xdata, ydata = np.array(xdata), np.array(ydata)
  p.fit(xdata, ydata)
  yt=p.predict(xdata[-10:])
  y = mm.inverse_transform(yt)
  low = np.round(y[-1: ],2)
  r = pickle.load(requests.get(f'https://github.com/mayanknagory/NSE50/blob/main/model_h/{equity}.pickle').content)
  #p = pickle.load(open (r,'rb'))
  #p = pickle.load(open(f'https://github.com/mayanknagory/NSE50/blob/main/model_h/{equity}.pickle', 'rb')) 	
  hp = df[['HighPrice']].tail(1).values
  sh = mm.fit_transform(df[['HighPrice']])
  xdata = []
  ydata = []
  for j in range(10, len(sh)):
    xdata.append(sh[j-10:j])
    ydata.append(sh[j])
  xdata, ydata = np.array(xdata), np.array(ydata)
  p.fit(xdata, ydata)
  yt=p.predict(xdata[-10:])
  y = mm.inverse_transform(yt)
  high = np.round(y[-1: ],2)
  return fig, m, ten ,lp, low, hp, high

st.set_page_config(layout="wide")
st.title('NSE Nifty 50')

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    button1 = st.button('Andani Ent', type="primary")
with col2:
    button2 = st.button('Adani Port', type="primary")
with col3:
    button3 = st.button('Apollo', type="primary")
with col4:
    button4 = st.button('Asian Paints', type="primary")
with col5:
    button5 = st.button('Axis Bank', type="primary")

col6, col7, col8, col9, col10, col11 = st.columns(6)
with col6:
    button6 = st.button('Bajaj Auto', type="primary")
with col7:
    button7 = st.button('Bajaj Finserv', type="primary")
with col8:
    button8 = st.button('Bajaj Finance', type="primary")
with col9:
    button9 = st.button('Bharti Airtel', type="primary")
with col10:
    button10 = st.button('BPCL', type="primary")
with col11:
    button11 = st.button('Brittania', type="primary")

col12, col13 = st.columns(2)
with col12:
    button12 = st.button('Cipla', type="primary")
with col13:
    button13 = st.button('Coal India', type="primary")

col14, col15 = st.columns(2)
with col14:
    button14 = st.button('Divi Lab', type="primary")
with col15:
    button15 = st.button('Dr. Reddy', type="primary")

button16 = st.button('Eicher Motors', type="primary")
button17 = st.button('Grasim', type="primary")

col18, col19, col20, col21, col22, col23 = st.columns(6)
with col18:
    button18 = st.button('HCL Tech', type="primary")
with col19:
    button19 = st.button('HDFC Bank', type="primary")
with col20:
    button20 = st.button('HDFC Life', type="primary")
with col21:
    button21 = st.button('Hero Motors', type="primary")
with col22:
    button22 = st.button('Hindalco', type="primary")
with col23:
    button23 = st.button('Hindustan Unilever', type="primary")

col24, col25, col26, col27 = st.columns(4)
with col24:
    button24 = st.button('ICICI Bank', type="primary")
with col25:
    button25 = st.button('IndusInd Bank', type="primary")
with col26:
    button26 = st.button('Infosys', type="primary")
with col27:
    button27 = st.button('ITC', type="primary")

button28 = st.button('JSW Steel', type="primary")
button29 = st.button('Kotak Mahindra', type="primary")

col30, col31 = st.columns(2)
with col30:
    button30 = st.button('Larsen & Tubro', type="primary")
with col31:
    button31 = st.button('LTIMindtree', type="primary")

col32, col33 = st.columns(2)
with col32:
    button32 = st.button('Mahindra and Mahindra', type="primary")
with col33:
    button33 = st.button('Maruti Suzuki', type="primary")

col34, col35 = st.columns(2)
with col34:
    button34 = st.button('Nestle', type="primary")
with col35:
    button35 = st.button('NTPC', type="primary")

button36 = st.button('ONGC', type="primary")
button37 = st.button('Power Grid', type="primary")
button38 = st.button('Reliance Industries', type="primary")

col39, col40, col41, col42 = st.columns(4)
with col39:
    button39 = st.button('SBI', type="primary")
with col40:
    button40 = st.button('SBI Life Insurance', type="primary")
with col41:
    button41 = st.button('Shriram Finance', type="primary")
with col42:
    button42 = st.button('Sun Pharma', type="primary")

col43, col44, col45, col46, col47, col48 = st.columns(6)
with col43:
    button43 = st.button('TCS', type="primary")
with col44:
    button44 = st.button('Tata Consumer Products', type="primary")
with col45:
    button45 = st.button('Tata Motors', type="primary")
with col46:
    button46 = st.button('Tata Steel', type="primary")
with col47:
    button47 = st.button('Tech Mahindra', type="primary")
with col48:
    button48 = st.button('Titan', type="primary")

button49 = st.button('Ultratech Cement', type="primary")
button50 = st.button('Wipro', type="primary")

if button1:
    fig, m, ten ,lp, low, hp, high = fun('ADANIENT')
    st.title('**Adani Enterprises**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button2:
    fig, m, ten ,lp, low, hp, high = fun('ADANIPORTS')
    st.title('**Adani Port**')    
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button3:
    fig, m, ten ,lp, low, hp, high = fun('APOLLOHOSP')
    st.title('**Apollo Hospitals**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button4:
    fig, m, ten ,lp, low, hp, high = fun('ASIANPAINT')
    st.title('**Asian Paints**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button5:
    fig, m, ten ,lp, low, hp, high = fun('AXISBANK')
    st.title('**Axis Bank**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button6:
    fig, m, ten ,lp, low, hp, high = fun('BAJAJ-AUTO')
    st.title('**Bajaj Auto**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button7:
    fig, m, ten ,lp, low, hp, high = fun('BAJAJFINSV')
    st.title('**Bajaj Finserv**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button8:
    fig, m, ten ,lp, low, hp, high = fun('BAJFINANCE')
    st.title('**Bajaj Finance**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button9:
    fig, m, ten ,lp, low, hp, high = fun('BHARTIARTL')
    st.title('**Bharti Airtel**')    
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button10:
    fig, m, ten ,lp, low, hp, high = fun('BPCL')
    st.title('**BPCL**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button11:
    fig, m, ten ,lp, low, hp, high = fun('BRITANNIA')
    st.title('**Britannia**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button12:
    fig, m, ten ,lp, low, hp, high = fun('CIPLA')
    st.title('**Cipla**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')


if button13:
    fig, m, ten ,lp, low, hp, high = fun('COALINDIA')
    st.title('**Coal India**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button14:
    fig, m, ten ,lp, low, hp, high = fun('DIVISLAB')
    st.title('**Divis Lab**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button15:
    fig, m, ten ,lp, low, hp, high = fun('DRREDDY')
    st.title('**Dr Reddy**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button16:
    fig, m, ten ,lp, low, hp, high = fun('EICHERMOT')
    st.title('**Eicher Motors**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button17:
    fig, m, ten ,lp, low, hp, high = fun('GRASIM')
    st.title('**Grasim**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button18:
    fig, m, ten ,lp, low, hp, high = fun('HCLTECH')
    st.title('**HCL Tech**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button19:
    fig, m, ten ,lp, low, hp, high = fun('HDFCBANK')
    st.title('**HDFC Bank**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button20:
    fig, m, ten ,lp, low, hp, high = fun('HDFCLIFE')
    st.title('**HDFC Life**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button21:
    fig, m, ten ,lp, low, hp, high = fun('HEROMOTOCO')
    st.title('**Hero Motors**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button22:
    fig, m, ten ,lp, low, hp, high = fun('HINDALCO')
    st.title('**Hindalco**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button23:
    fig, m, ten ,lp, low, hp, high = fun('HINDUNILVR')
    st.title('**Hindustan Unilever**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button24:
    fig, m, ten ,lp, low, hp, high = fun('ICICIBANK')
    st.title('**ICICI Bank**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button25:
    fig, m, ten ,lp, low, hp, high = fun('INDUSINDBK')
    st.title('**INDUSIND Bank**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button26:
    fig, m, ten ,lp, low, hp, high = fun('INFY')
    st.title('**Infosys**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button27:
    fig, m, ten ,lp, low, hp, high = fun('ITC')
    st.title('**ITC**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button28:
    fig, m, ten ,lp, low, hp, high = fun('JSWSTEEL')
    st.title('**JSW Steel**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button29:
    fig, m, ten ,lp, low, hp, high = fun('KOTAKBANK')
    st.title('**Kotak Mahindra Bank**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button30:
    fig, m, ten ,lp, low, hp, high = fun('LT')
    st.title('**Larsen & Tubro**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button31:
    fig, m, ten ,lp, low, hp, high = fun('LTIM')
    st.title('**LTIMindtree**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button32:
    fig, m, ten ,lp, low, hp, high = fun('M&M')
    st.title('**Mahindra and Mahindra**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button33:
    fig, m, ten ,lp, low, hp, high = fun('MARUTI')
    st.title('**Maruti Suzuki**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button34:
    fig, m, ten ,lp, low, hp, high = fun('NESTLEIND')
    st.title('**Nestle**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button35:
    fig, m, ten ,lp, low, hp, high = fun('NTPC')
    st.title('**NTPC**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button36:
    fig, m, ten ,lp, low, hp, high = fun('ONGC')
    st.title('**ONGC**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button37:
    fig, m, ten ,lp, low, hp, high = fun('POWERGRID')
    st.title('**Power Grid**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button38:
    fig, m, ten ,lp, low, hp, high = fun('RELIANCE')
    st.title('**Reliance Industries**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button39:
    fig, m, ten ,lp, low, hp, high = fun('SBIN')
    st.title('**State Bank of India**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button40:
    fig, m, ten ,lp, low, hp, high = fun('SBILIFE')
    st.title('**SBI Life Insurance**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button41:
    fig, m, ten ,lp, low, hp, high = fun('SHRIRAMFIN')
    st.title('**Shriram Finance**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button42:
    fig, m, ten ,lp, low, hp, high = fun('SUNPHARMA')
    st.title('**Sun Pharma**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button43:
    fig, m, ten ,lp, low, hp, high = fun('TCS')
    st.title('**TCS**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button44:
    fig, m, ten ,lp, low, hp, high = fun('TATACONSUM')
    st.title('**Tata Consumer Products**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button45:
    fig, m, ten ,lp, low, hp, high = fun('TATAMOTORS')
    st.title('**Tata Motors**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button46:
    fig, m, ten ,lp, low, hp, high = fun('TATASTEEL')
    st.title('**Tata Steel**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button47:
    fig, m, ten ,lp, low, hp, high = fun('TECHM')
    st.title('**Tech Mahindra**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button48:
    fig, m, ten ,lp, low, hp, high = fun('TITAN')
    st.title('**Titan**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button49:
    fig, m, ten ,lp, low, hp, high = fun('ULTRACEMCO')
    st.title('**UltraTech Cement**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')

if button50:
    fig, m, ten ,lp, low, hp, high = fun('WIPRO')
    st.title('**Wipro**')
    st.pyplot(fig)
    st.write(f'**The average daily return in the last 10 trading days was {m}%**')
    st.write('**Low and High Prices over the last 10 trading days**\n',ten)
    col1, col2 = st.columns(2)
    with col1:
      st.write('The Low Price of yesterday was:', u"\u20B9",f'**{str(lp[0])}**')
    with col2:
      st.write('The High Price for yesterday was:', u"\u20B9", f'**{str(hp[0])}**')
    col3, col4 = st.columns(2)
    with col3:
      st.write('The Predicted Low Price for today is:', u"\u20B9",f'**{str(low[0])}**')
    with col4:
      st.write('The Predicted High Price for today is:', u"\u20B9", f'**{str(high[0])}**')
