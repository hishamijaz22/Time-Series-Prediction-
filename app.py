import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load the pickled model results
with open('model_results.pkl', 'rb') as f:
    model_results = pickle.load(f)

# Define UI components
def dataset_selection():
    st.sidebar.header('Dataset Selection')
    dataset = st.sidebar.selectbox('Select Dataset', ['Dataset 1', 'Dataset 2', 'Dataset 3'])
    return dataset

def model_views():
    st.sidebar.header('Model Views')
    view = st.sidebar.radio('Select Model View', ['ARIMA', 'ANN', 'SARIMA', 'SVR'])
    return view

def execute_forecast():
    if st.button('Execute Forecast'):
        st.write('Forecasting...')

# Main function to render UI
def main():
    st.title('Time Series Forecasting System')

    dataset = dataset_selection()

    st.sidebar.header('Model Views')
    view = model_views()

    if view == 'ARIMA':
        st.header('ARIMA Model View')
        st.write('ARIMA RMSE:', model_results['arima_rmse'])
        st.write('ARIMA Forecast:', model_results['arima_forecast'])

        # Plot actual vs predicted values
        plt.figure(figsize=(10, 5))
        plt.plot(model_results['arima_forecast'], label='Predicted')
        plt.plot(model_results['test_data'], label='Actual')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('ARIMA Forecast vs Actual')
        plt.legend()
        st.pyplot(plt)

        # Model comparison histogram
        st.subheader('Model Comparison')
        st.bar_chart({'ARIMA RMSE': model_results['arima_rmse']})

    elif view == 'ANN':
        st.header('ANN Model View')
        st.write('ANN RMSE:', model_results['ann_rmse'])
        st.write('ANN Forecast:', model_results['ann_forecast'])

        # Plot actual vs predicted values
        plt.figure(figsize=(10, 5))
        plt.plot(model_results['ann_forecast'], label='Predicted')
        plt.plot(model_results['test_data'], label='Actual')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('ANN Forecast vs Actual')
        plt.legend()
        st.pyplot(plt)

        # Model comparison histogram
        st.subheader('Model Comparison')
        st.bar_chart({'ANN RMSE': model_results['ann_rmse']})

    elif view == 'SARIMA':
        st.header('SARIMA Model View')
        st.write('SARIMA RMSE:', model_results['sarima_rmse'])
        st.write('SARIMA Forecast:', model_results['sarima_forecast'])

        # Plot actual vs predicted values
        plt.figure(figsize=(10, 5))
        plt.plot(model_results['sarima_forecast'], label='Predicted')
        plt.plot(model_results['test_data'], label='Actual')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('SARIMA Forecast vs Actual')
        plt.legend()
        st.pyplot(plt)

        # Model comparison histogram
        st.subheader('Model Comparison')
        st.bar_chart({'SARIMA RMSE': model_results['sarima_rmse']})

    elif view == 'SVR':
        st.header('SVR Model View')
        st.write('SVR RMSE:', model_results['svr_rmse'])
        st.write('SVR Forecast:', model_results['svr_forecast'])

        # Plot actual vs predicted values
        plt.figure(figsize=(10, 5))
        plt.plot(model_results['svr_forecast'], label='Predicted')
        plt.plot(model_results['test_data'], label='Actual')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('SVR Forecast vs Actual')
        plt.legend()
        st.pyplot(plt)

        # Model comparison histogram
        st.subheader('Model Comparison')
        st.bar_chart({'SVR RMSE': model_results['svr_rmse']})

if __name__ == '__main__':
    main()
