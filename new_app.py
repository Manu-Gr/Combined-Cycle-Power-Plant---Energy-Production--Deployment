# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 08:54:51 2022

@author: 91890
"""


import pickle
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import warnings
import plotly.express as px
warnings.filterwarnings("ignore")


st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: 90%;
        padding-top: 5rem;
        padding-right: 5rem;
        padding-left: 5rem;
        padding-bottom: 5rem;
    }}
    img{{
    	max-width:40%;
    	margin-bottom:40px;
    }}
</style>
""",
        unsafe_allow_html=True,
    )

header   = st.container()
dataset  = st.container()
Model    = st.container()


@st.cache(suppress_st_warning=True)
def get_data(file):
    energy_data = pd.read_csv(file,sep=';') 
    return energy_data



  
with header:
    st.title('Combined Cycle Power Plant Energy Prediction')
    st.markdown('Project by - **Manu G**')


filename= r"Final_Random_foret_model.pkl"
pickle_file = open(filename, 'rb') 
classifier = pickle.load(pickle_file)

def predict(Temperature,Vacuume,Pressure,Humidity):
 
    dataframe2 = pd.DataFrame([[Temperature, Vacuume,Pressure,Humidity]], columns=['temperature', 'exhaust_vacuum', 'amb_pressure','r_humidity'])

    result= classifier.predict(dataframe2)

    return result


with dataset:
    st.header('Energy Dataset')
    
    energy_data = get_data('energy_production.csv',    )
    st.write(energy_data.head(50))
    st.write(energy_data.shape)

 # EDA
    st.header('EDA')
    st.subheader('Histogram Distributions')

    #Histogram Visualization
    sel_col,dis_col = st.columns(2)
    colums = energy_data.columns
    hist_col = sel_col.selectbox('Select The Column for Visulization', options= colums,  index=0 )
    
    hist_plots = px.histogram(energy_data,x = hist_col, )
    dis_col.write(hist_plots)
    
    # Boxplot Visulization
    st.subheader('Boxplots to check for outliers')
    sel_col1,dis_col1 = st.columns(2)
    box_col = sel_col1.selectbox('Select The Column for Visulization', options= colums, key = 'start_box' ,  index=0 )
    
    box_plots = px.box(energy_data,y = box_col, )
    dis_col1.write(box_plots)
    
    # Scatter plots
    st.subheader("Scatter Plots")
    sel_col2,dis_col2 = st.columns(2)
    scat_col = sel_col2.selectbox('Select The Column for Visulization', options= colums, key = 'start_scat' ,  index=0 )
    
    scat_plots = px.scatter(energy_data, x= 'energy_production' ,y = scat_col, trendline='ols' )
    dis_col2.write(scat_plots)      
    
    # Heatmap
    st.subheader("Heatmap")
    corr_matrix = energy_data.corr()
    heatmap = px.imshow(corr_matrix, text_auto= True)
    st.write(heatmap)    

with Model:
    st.header("Random Forest Model")

    Temperature = st.number_input("Enter The Temperature Values")
    Vacuum = st.number_input("Enter The Vacuum Values")
    Pressure = st.number_input("Enter The Pressure Values")
    Humidity= st.number_input("Enter The Humidity Values") 

df = pd.DataFrame([[Temperature,Vacuum ,Pressure,Humidity]], columns=['temperature', 'exhaust_vacuum', 'amb_pressure','r_humidity'])
if st.button("Predict Energy Production"):
    prediction= predict(Temperature,Vacuum,Pressure,Humidity)
    st.table(df)
    
    st.subheader("The Estimated Energy Output is ",)
    st.write(prediction)




    





