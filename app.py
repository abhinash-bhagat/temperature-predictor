import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from prophet import Prophet
from datetime import date

#*******Custom Styling***********
st.markdown("""
    <style>
    .css-uf99v8{
        background-color: #561acb;
    }
    .css-fblp2m{
        display:inline;
    }
    .css-6qob1r{
        background-color:#2c3257;
    }
    .css-1avcm0n{
        background-color:#2c3257;
    }
    .css-1y4p8pa{
        width: 100%;
        max-width: 50rem;
    }
    .css-cio0dv{
        display:none;
    }
    </style>
""", unsafe_allow_html=True)
#******Custom Styling************

#Sidebar*****
st.sidebar.markdown("<h3 style='text-align:center;'>Input Zone</h3>",unsafe_allow_html=True)
min_date = date(2013, 1, 1)
max_date = date(2023, 12, 31)
user_input= st.sidebar.date_input("Select the date:", min_value=min_date, max_value=max_date)
user_input_2 = st.sidebar.radio("Display the Chart?",options=['Yes','No'])


# Load the model from the pkl file
def load_model():
    with open('temp_model.pkl', 'rb') as f:
        # model = pickle.load(f)
        model = pd.read_pickle(f)
    return model


# Function to make predictions using the loaded model
def make_predictions(model, data):
    future = model.make_future_dataframe(periods=2555)  # Adjust the number of periods as needed
    forecast = model.predict(future)
    return forecast

# Main function
def main():
    st.markdown("""
        <h1 style='text-align: center;
        margin-bottom:40px;
        background-color:#5b39e1;
        border-radius: 21px;'>
        Delhi Temperature Predictor</h1>
    """,unsafe_allow_html=True)

    st.markdown("""
    <h4>A minimal web application that allows you to predict the tempearture of Delhi, India.</h4> 
        """, unsafe_allow_html=True)
    st.caption("Currently, you will only be able to predict the temperature of 2023 AD")
    model = load_model()
    
    if st.sidebar.button('Predict'):
        # Prepare input data for prediction
        input_data = pd.DataFrame({'ds': [user_input]})
        input_data['ds'] = pd.to_datetime(input_data['ds'])
        
        # Make predictions
        forecast = make_predictions(model, input_data)
        
        # Retrieve the 'yhat' value for the specified date
        temperature = forecast.loc[forecast['ds'] == input_data['ds'].iloc[0], 'yhat'].values[0]
        max_temperature = forecast.loc[forecast['ds'] == input_data['ds'].iloc[0], 'yhat_upper'].values[0]
        min_temperature = forecast.loc[forecast['ds'] == input_data['ds'].iloc[0], 'yhat_lower'].values[0]
        col1, col2, col3 = st.columns(3)
        col1.markdown("<h3>Expected Temp</h3>",unsafe_allow_html=True)
        col1.metric(label="ᴼC ", value=round(temperature,2), delta=0)
        col2.markdown("<h3>Expected Max. Temp</h3>", unsafe_allow_html=True)
        col2.metric(label="°C", value=round(max_temperature, 2), delta=round(max_temperature - temperature, 2))
        col3.markdown("<h3>Expected Min. Temp</h3>",unsafe_allow_html=True)
        col3.metric(label="°C", value=round(min_temperature, 2), delta=round(min_temperature - temperature, 2))

        #Data For Charts
        data = forecast.rename(columns={"ds":"Date","yhat":"Temperature","yhat_upper":"High","yhat_lower":"Low"})
        
        #Line Chart
        if user_input_2 == 'Yes':
            st.markdown("""
        <h4 style='text-align: center;
        margin-top:18px;'>
        Line Chart </h4>
    """,unsafe_allow_html=True)
            st.line_chart(data=data, x="Date", y=["Temperature", "High", "Low"])
        else:
            pass

        #Displaying records
        filtered_row_index = data[data['Date'] == input_data['ds'].iloc[0]].index[0]
        filtered_data = data.loc[0:filtered_row_index, ['Date', 'Temperature', 'High', 'Low']]

        st.markdown(f"""
        <h4 style='text-align: center;
        margin-top:18px;'>
        Past Records</h4>
    """,unsafe_allow_html=True)
        
        st.dataframe(filtered_data,width=780)

if __name__ == '__main__':
    main()