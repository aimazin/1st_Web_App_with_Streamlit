import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale, Normalizer
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.metrics import mean_squared_error, recall_score, plot_precision_recall_curve
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor 
import pandas_datareader as pdr
import datetime
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

def main():
    st.sidebar.title("MSFT Returns Prediction Web App")
    st.title("MSFT Returns Prediction Web App")
    st.sidebar.markdown("What kind of return can one expect from my Microsoft Stock? ")
    st.markdown("What kind of return can one expect from my Microsoft Stock? ")
                   

    @st.cache(persist=True)
    def RSI(close,timeperiod):
        delta=pd.DataFrame(close).diff(1).dropna()
        u=delta*0
        d=delta*0
        u[delta>0]=delta[delta>0]
        d[delta<0]=delta[delta<0]
        uav= u.ewm(com=timeperiod-1,min_periods=timeperiod).mean()
        dav= d.ewm(com=timeperiod-1,min_periods=timeperiod).mean()
        rs= abs(uav/dav)[timeperiod-1:]
        return  (100 - 100/(1 + rs))
    
    
    def BBANDS(close, timeperiod, nbdevup, nbdevdn):
        stdev= pd.DataFrame(close).rolling(timeperiod).std().dropna()
        rolclos= pd.DataFrame(close).rolling(timeperiod).mean().dropna()
        up=nbdevup*stdev + rolclos
        dn=rolclos - nbdevdn*stdev
        return pd.DataFrame(up), pd.DataFrame(rolclos), pd.DataFrame(dn)
    
    
    def CCI(close, high, low, timeperiod, constant):
        TP = (close + high + low)/3
        CCI = pd.Series((TP - TP.rolling(timeperiod).mean()) / (constant * TP.rolling(timeperiod).std()), name = 'CCI_' + str(timeperiod)) 
        return CCI

    
    def load_data(stock):
        data = pdr.get_data_yahoo(stock, 
                          start=datetime.datetime.strftime(datetime.datetime.today() + datetime.timedelta(days = -1095) , '%Y/%m/%d'), 
                          end=datetime.datetime.strftime(datetime.datetime.today() , '%Y/%m/%d'))
        
        bx = np.round(data.iloc[:,-1],2) 
        tx = np.round(data.iloc[:,1],2)
        cx = np.round(data.iloc[:,2],2)
     
        # Create moving averages and rsi for timeperiods of 14, 30, 50, and 200
        #for n in [30]:
        n=30
        # Create the RSI indicator
        data['rsix' + str(n)] = RSI(bx, n)
        
        # Create the CCI indicator
        data['ccix' + str(n)] = CCI(bx, tx, cx, n, 1)
        
       
        # Create bbands indicators
        data["upperband"+str(n)], data["middleband"+str(n)], data["lowerband"+str(n)] = BBANDS(bx, n, 2, 2)
        
        
            
      
        data['5cum_pct']=(1+data['Close'].pct_change(-5).diff(periods=1)).cumprod()


  
        return data 

    
    def split(df):
      
        con=df.shape[0]
        con60=con-60

        x_train = scale(df.iloc[201:con60,4:12].values)
        x_test = scale(df.iloc[(con60-10):con,4:12].values)[:56,:]

        y_train = df.iloc[201:con60,-1].values
        y_test = df.iloc[(con60-1):con,-1].values[:56]
        return x_train, x_test, y_train, y_test

     
    
    
    x_train, x_test, y_train, y_test = [],[],[],[]


    df = load_data(stock= 'MSFT')
    x_train, x_test, y_train, y_test = split(df)

    

    
    x_train, x_test, y_train, y_test = np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)


    def plot_metrics(metrics_list):
        if 'Price and Prediction' in metrics_list:
            st.subheader('Price and Prediction') 
            plt.plot(y_test, color = 'red', label = 'Real MSFT Stock Price')
            plt.plot(y_pred, color = 'cyan', label = 'Predicted Model')
            plt.title('MSFT cumulative percentage return Prediction last 60 days')
            plt.legend()
            plt.xlabel('Time')
            plt.ylabel('MSFT cumulative percentage return')
            st.pyplot()

        

    
    st.sidebar.subheader('Choose Regressor')
    reg = st.sidebar.selectbox("Regressor",('Linear Reg','SVM','Random Forest'))

    if reg == 'Linear Reg':
        st.sidebar.subheader('Model Hyperparameters')

        metrics = st.sidebar.multiselect('What metrics to plot?',('Price and Prediction','ROC Curve','Precision-Recall Curve'))

        if st.sidebar.button('Classify', key='classify'):
            st.subheader('Linear Results')
            model = LinearRegression()
            model.fit(x_train,y_train)                   
            y_pred = model.predict(x_test)
            st.write('Accuracy', model.score(x_test,y_test).round(2))
            st.write('Precision', mean_squared_error(y_test,y_pred).round(2))
            plot_metrics(metrics)

    if reg == 'SVM':
        st.sidebar.subheader('Model Hyperparameters')
        C = st.sidebar.number_input('C (Reg Param)', 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio('Kernel',('rbf','linear'), key='kernel')
        gamma = st.sidebar.radio('Gamma Kernel Coeff.', ('scale','auto'), key='gamma')

        metrics = st.sidebar.multiselect('What metrics to plot?',('Price and Prediction', 'ROC Curve','Precision-Recall Curve'))

        if st.sidebar.button('Classify', key='classify'):
            st.subheader('SVM')
            model = SVR(C=C, kernel=kernel, gamma=gamma)
            model.fit(np.array(x_train), np.array(y_train))
            y_pred = model.predict(x_test)
            st.write('Accuracy', model.score(x_test,y_test).round(2))
            st.write('Precision rsme', mean_squared_error(y_test,y_pred).round(2))
            plot_metrics(metrics)


    if reg == 'Random Forest':
        st.sidebar.subheader('Model Hyperparameters')
        n_estimators = st.sidebar.number_input('Number of Trees', 100, 5000, step=100, key='n_estimators')
        max_depth = st.sidebar.number_input('Tree Depth', 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio('Bootstrap Yay or Neigh', ('True','False'), key='bootstrap')

        metrics = st.sidebar.multiselect('What metrics to plot?',('Price and Prediction', 'ROC Curve','Precision-Recall Curve'))

        if st.sidebar.button('Classify', key='classify'):
            st.subheader('Random Forest Results')
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            model.fit(np.array(x_train),np.array(y_train))
            y_pred = model.predict(x_test)
            st.write('Accuracy', model.score(x_test,y_test).round(2))
            st.write('Precision rmse', mean_squared_error(y_test,y_pred).round(2))
            plot_metrics(metrics)            

    if st.sidebar.checkbox('Show raw data', False):
        st.subheader(' data set (Classisfication)')
        st.write(x_train,x_test)              






if __name__ == '__main__':
    main()


