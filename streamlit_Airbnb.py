#per lanciarlo: python -m streamlit run .\streamlit_Airbnb.py

# import librearies
import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import plotly.express as px NON PENSO SERVA
import folium 
from io import BytesIO
#from streamlit_folium import st_folium
#from streamlit_folium import folium_static

##################
#Importing Dataset
##################
airbnb_df = pd.read_csv('Airbnb_Open_Data.csv')

##########################
#Set tabs for the chapters
##########################

tab_names = ["Introduction", "Cleaning", "Correlation", "Exploratory Data Analysis", "Modeling with ML algorithms"]
current_tab = st.sidebar.selectbox("Summary", tab_names)
st.sidebar.markdown(
    """
    **Anna Bicelli**  
    [GitHub](https://github.com/AnnaBicelli)  
    """
)


##############
#Introduction
##############

if current_tab == "Introduction":
    st.markdown("<h1 style='text-align: center;'>Exploring the Airbnb Open Data</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Programming and Database: Final Project</h2>", unsafe_allow_html=True)
    st.markdown('''
                **Author**: Anna Bicelli
                ''')
    st.markdown("""
    This dataset covers Airbnb activity in New York City \n
     **Data source:** https://www.kaggle.com/datasets/arianazmoudeh/airbnbopendata
    """)
    #st.write('This dataset covers Airbnb activity in New York City and is available at the following link: https://www.kaggle.com/datasets/arianazmoudeh/airbnbopendata .')
    st.write('It reports the listing activity of homestays in New York City, their reviews, prices, availability, location, room types and cancellation policies.')

    selected_columns = st.multiselect('Explore the Airbnb dataset by selecting columns', airbnb_df.columns)
    if selected_columns:
        columns_df = airbnb_df.loc[:, selected_columns]
        st.dataframe(columns_df.head(15)) #creo la barra per le opzioni
    else:
        st.dataframe(airbnb_df.head(15)) #metto head
        # è possibile vedere tutte le colonne, oppure esplorare il dataset selezionando solamente alcune colonne, ad esempio (faccio vedere con alcune) e mestro che in questo caso è il df iniziale quindi con NA value 
    st.write('General informations about the DataFrame:')
    st.dataframe(airbnb_df.info())#guardo info in modo da dire quante righe e quante colonne sono presenti nel dataset scelto
    import io

    # Cattura l'output di airbnb_df.info() in una stringa
    buffer = io.StringIO()
    airbnb_df.info(buf=buffer)

    # Estrai la stringa dall'output
    info_str = buffer.getvalue()

    # Visualizza le informazioni utilizzando st.write()
    st.write('General informations about the DataFrame:')
    st.write(info_str)#PROVA A SISTEMARLO 

#########
#CLEANING
#########
elif current_tab == "Cleaning": 
    st.title("Cleaning NA values")
   
    st.write('Before proceeding with the analysis, the null values in the dataset were analyzed and then replaced or eliminated.')

    tab1, tab2, tab3, tab4 = st.tabs(["NA's values", "Cleaning", "-", "-"]) 
    
    with tab1:       
 
        columns_to_check = ['NAME', 'host_identity_verified', 'host name', 'neighbourhood group', 'neighbourhood', 'lat', 'long', 'country', 'country code', 'instant_bookable', 'cancellation_policy', 'Construction year', 'price', 'service fee', 'minimum nights', 'number of reviews', 'last review','reviews per month', 'calculated host listings count', 'availability 365', 'house_rules', 'license']
        def calculate_na_percentage(column):
            return (airbnb_df[column].isna().sum() / len(airbnb_df)) * 100
        for column in columns_to_check:
            percentage_na = calculate_na_percentage(column)
            st.code(f"Percentage of {column} NA value: {percentage_na:.2f}%")

    with tab2:
        st.markdown('''
                based on this information, the data will be rearranged as follows:
                - **Replace** missing values for the variable **age** with the **mean value** (1.16%).
                - Missing values for the variables **sex**, **place_of_residence**, **place_of_residence_district**, **type_of_location**, and **notes** are **replaced** with the **mode value**. this is so as not to change the distribution of the data too much.
                - **Remove** the variables **has_participated_in_hostilities**, **ammunition** from the variable itself. The percentage of null values is excessive and would not lead to useful information for the entire population of the dataset.
                                ''')   
    


    

