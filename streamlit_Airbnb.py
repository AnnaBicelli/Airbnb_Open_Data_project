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
import io
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

tab_names = ["📄 Introduction", "🗑️ Cleaning", "🔗 Correlation", "📊 Exploratory Data Analysis", " 🤖 Modeling with ML algorithms"]
current_tab = st.sidebar.selectbox("Table of content", tab_names)
st.sidebar.markdown(
    """
    **My page on GitHub:**   [GitHub](https://github.com/AnnaBicelli)  
    """
)



##############
#Introduction
##############

if current_tab == "📄 Introduction":
    st.markdown("<h1 style='text-align: center;'>Exploring the Airbnb Open Data</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Programming and Database: Final Project</h2>", unsafe_allow_html=True)
    st.markdown('''
                **Author**: Anna Bicelli
                ''')
    st.markdown("""
    This dataset covers Airbnb activity in New York City. \n
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
    

    st.write('General informations about the DataFrame')
    # Creazione di un buffer per catturare le informazioni sull'Airbnb DataFrame
    buffer = io.StringIO()
    airbnb_df.info(buf=buffer)
    s = buffer.getvalue()
    # Mostra il multiselect per selezionare le colonne da visualizzare
    selected_columns1 = st.multiselect("Select the variables", airbnb_df.columns.tolist(), default=airbnb_df.columns.tolist())

    # Se sono selezionate colonne, mostra le informazioni solo per quelle colonne
    if selected_columns1:
        selected_info_buffer = io.StringIO()
        airbnb_df[selected_columns1].info(buf=selected_info_buffer)
        selected_info = selected_info_buffer.getvalue()
        st.text(selected_info)
    else:
        # Altrimenti, mostra le informazioni per tutte le colonne
        st.text(s)

#########
#CLEANING
#########
elif current_tab == "🗑️ Cleaning": 
    st.title("Cleaning NA values")
   
    st.write('Before proceeding with the analysis, the null values in the dataset were analyzed and then replaced or eliminated.')

    ###########
    #Functions
    ##########
    # Funzione per pulire i dati
    def clean_data(df):
        # Copia il DataFrame per evitare modifiche al DataFrame originale
        cleaned_df = df.copy()
    
        # Converti i nomi delle colonne in minuscolo e sostituisci gli spazi con underscores
        cleaned_df.columns = cleaned_df.columns.map(lambda x: x.lower().replace(' ', '_'))
    
        # Rimuovi i simboli '$' e ',' dalla colonna 'price' e 'service_fee' e converti in float
        cleaned_df['price'] = cleaned_df['price'].replace({'\$': '', ',': ''}, regex=True).astype(float)
        cleaned_df['service_fee'] = cleaned_df['service_fee'].replace({'\$': '', ',': ''}, regex=True).astype(float)
    
        # Calcola le medie dei valori mancanti per le colonne numeriche e riempi i valori mancanti
        cleaned_df['lat'] = cleaned_df['lat'].fillna(cleaned_df['lat'].mean())
        cleaned_df['long'] = cleaned_df['long'].fillna(cleaned_df['long'].mean())
        cleaned_df['construction_year'] = cleaned_df['construction_year'].fillna(cleaned_df['construction_year'].mean())
        cleaned_df['minimum_nights'] = cleaned_df['minimum_nights'].fillna(cleaned_df['minimum_nights'].mean())
        cleaned_df['number_of_reviews'] = cleaned_df['number_of_reviews'].fillna(cleaned_df['number_of_reviews'].mean())
        cleaned_df['review_rate_number'] = cleaned_df['review_rate_number'].fillna(cleaned_df['review_rate_number'].mean())
        cleaned_df['calculated_host_listings_count'] = cleaned_df['calculated_host_listings_count'].fillna(cleaned_df['calculated_host_listings_count'].mean())
        cleaned_df['availability_365'] = cleaned_df['availability_365'].fillna(cleaned_df['availability_365'].mean())
    
        # Rimuovi la colonna "reviews_per_month"
        cleaned_df.drop(columns=["reviews_per_month"], inplace=True)
    
        # Riempi i valori mancanti nelle colonne 'name', 'host_identity_verified', 'host_name', ecc. con le modalità
        cleaned_df['name'] = cleaned_df['name'].fillna(cleaned_df['name'].mode()[0])
        cleaned_df['host_identity_verified'] = cleaned_df['host_identity_verified'].fillna(cleaned_df['host_identity_verified'].mode()[0])
        cleaned_df['host_name'] = cleaned_df['host_name'].fillna(cleaned_df['host_name'].mode()[0])
        cleaned_df['neighbourhood_group'] = cleaned_df['neighbourhood_group'].fillna(cleaned_df['neighbourhood_group'].mode()[0])
        cleaned_df['neighbourhood'] = cleaned_df['neighbourhood'].fillna(cleaned_df['neighbourhood'].mode()[0])
        cleaned_df['country'] = cleaned_df['country'].fillna(cleaned_df['country'].mode()[0])
        cleaned_df['country_code'] = cleaned_df['country_code'].fillna(cleaned_df['country_code'].mode()[0])
        cleaned_df['instant_bookable'] = cleaned_df['instant_bookable'].fillna(cleaned_df['instant_bookable'].mode()[0])
        cleaned_df['cancellation_policy'] = cleaned_df['cancellation_policy'].fillna(cleaned_df['cancellation_policy'].mode()[0])
        cleaned_df['room_type'] = cleaned_df['room_type'].fillna(cleaned_df['room_type'].mode()[0])
    
        # Rimuovi le colonne "license", "house_rules", "last_review"
        cleaned_df.drop(columns=["license", "house_rules", "last_review"], inplace=True)
    
        return cleaned_df
    # Pulisci i dati
    cleaned_df = clean_data(airbnb_df)

    tab1, tab2, tab3, tab4 = st.tabs(["NA's values", "Cleaning", "-", "-"]) 
    
    with tab1:       
        #st.write(airbnb_df.isna().sum())
        # Calcola il conteggio dei valori mancanti e la percentuale di valori mancanti per ciascuna variabile
        missing_values_count = airbnb_df.isna().sum()
        total_values = airbnb_df.shape[0]
        missing_values_percentage = (missing_values_count / total_values) * 100

        # Crea un nuovo DataFrame con il conteggio e la percentuale di valori mancanti
        missing_df = pd.DataFrame({
            'Variabile': missing_values_count.index,
            'Numero di valori mancanti': missing_values_count.values,
            'Percentuale di valori mancanti': missing_values_percentage.values
        })

        # Mostra il DataFrame dei valori mancanti
        st.write(missing_df)

        #ELIMINA QUESTA PARTE DA QUI
        columns_to_check = ['NAME', 'host_identity_verified', 'host name', 'neighbourhood group', 'neighbourhood', 'lat', 'long', 'country', 'country code', 'instant_bookable', 'cancellation_policy', 'Construction year', 'price', 'service fee', 'minimum nights', 'number of reviews', 'last review','reviews per month', 'calculated host listings count', 'availability 365', 'house_rules', 'license']
        def calculate_na_percentage(column):
            return (airbnb_df[column].isna().sum() / len(airbnb_df)) * 100
        for column in columns_to_check:
            percentage_na = calculate_na_percentage(column)
            st.code(f"Percentage of {column} NA value: {percentage_na:.2f}%")
        #A QUI
    with tab2:
        st.markdown('''
                based on this information, the data will be rearranged as follows:
                - **Replace** missing values for the variable **age** with the **mean value** (1.16%).
                - Missing values for the variables **sex**, **place_of_residence**, **place_of_residence_district**, **type_of_location**, and **notes** are **replaced** with the **mode value**. this is so as not to change the distribution of the data too much.
                - **Remove** the variables **has_participated_in_hostilities**, **ammunition** from the variable itself. The percentage of null values is excessive and would not lead to useful information for the entire population of the dataset.
                - sistemate gli indici delle colonne
                - eliminate alcune colonne
                                ''') 
        
        #METTI SOTTO CON COLONNE SISTEMATE E VALORI NULLI SISTEMATI: IN CLEANING
        with st.expander('Resulted DataFrame Preview'):
            st.write(cleaned_df.head(15)) 
     
    


    

