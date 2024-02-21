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
#Cleaning
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

        # Arrotonda la percentuale a due decimali
        missing_values_percentage = missing_values_percentage.round(2)

        # Crea un nuovo DataFrame con il conteggio e la percentuale di valori mancanti
        missing_df = pd.DataFrame({
            'Variable': missing_values_count.index,
            'NA values': missing_values_count.values,
            '%  NA values': missing_values_percentage.values
        })

        # Mostra il DataFrame dei valori mancanti
        st.write(missing_df)

    with tab2:
        st.markdown('''
                Based on these information the data will be rearranged as follows:
                - **fixed** the column indexes
                - **replaced** the null values related to *float* data with their **mean value** because the percentages were really close to zero
                - **replaced** the null values related to *object* variables with the **mode**, so it doesn't change the distribution a lot
                - **removed** the NaN values in the columns `price` and `service_fee`
                -  **removed** the columns `reviews_per_month`, `last_review`,`license` and `house_rules` because the percentage of null values was excessive and would not lead to useful information
                
                                ''') 
        
        #METTI SOTTO CON COLONNE SISTEMATE E VALORI NULLI SISTEMATI: IN CLEANING
        with st.expander('Resulted DataFrame Preview'):
            st.write(cleaned_df.head(15)) 

############
#Correlation
############
elif current_tab == "🔗 Correlation": 
    st.title("Correlations between values")
    
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
    
    cleaned_df = clean_data(airbnb_df)
    
    st.write("A preliminary graphical analysis through heatmap proved useful in exploring the correlations between numerical variables in the dataset.")
    # Heatmap 
    cleaned_df_corr = cleaned_df.corr(numeric_only=True)
    plt.figure(figsize=(28, 17))
    sns.heatmap(cleaned_df_corr, annot=True, cmap="BuPu",annot_kws={"size": 15}, xticklabels=cleaned_df_corr.columns, yticklabels=cleaned_df_corr.columns, linewidths=0.5, linecolor='white')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    st.pyplot(plt.gcf())

    st.write('Through this graph is possible to see that all numerical variables are uncorrelated with each other, except `price` and `service_fee` which are perfectly positively correlated.')

    ##########
    #Function
    ##########   
    # Definisci la funzione plot_scatter fuori da qualsiasi condizione
    def plot_scatter(cleaned_df):
        plt.figure(figsize=(6, 3))
        sns.set_palette("husl")
        sns.scatterplot(data=cleaned_df, x='service_fee', y='price', color='purple')
        plt.xlabel('Service Fee')
        plt.ylabel('Price')
        plt.title('Relationship between Service Fee and Price')
        st.pyplot(plt.gcf())  # Usa plt.gcf() per ottenere l'attuale figura in matplotlib

    # Inizializza o aggiorna lo stato quando viene premuto il pulsante
    if 'show_scatter' not in st.session_state:
        st.session_state.show_scatter = False

    if st.button('Click to see the Scatterplot'):
        st.session_state.show_scatter = not st.session_state.show_scatter

    # Mostra o nasconde il grafico in base allo stato della sessione
    if st.session_state.show_scatter:
        plot_scatter(cleaned_df)  # Assicurati che cleaned_df sia definito e accessibile
    
    st.divider() 
    st.write('To observe how the main categorical variables relate to each other, contingency tables were used, which help identify areas of higher or lower frequency of combinations between types of variables:')

    tab1, tab2, tab3, tab4 = st.tabs(["Room_type vs Cancellation_policy", "Room_type vs Neighbourhood_group", "Neighbourhood vs Neighbourhood_group", "Host_identity_verified vs Neighbourhood_group"]) 
    
    with tab1:
            contingency_table = pd.crosstab(cleaned_df['room_type'], cleaned_df['cancellation_policy'])
            plt.figure(figsize=(8, 6))
            sns.heatmap(contingency_table, annot=True, cmap='BuPu', fmt='d')
            #plt.title('Heatmap of contingency tables between room_type and cancellation_policy')
            plt.xlabel('Cancellation Policy')
            plt.ylabel('Room Type')
            st.pyplot(plt.gcf())

            st.markdown('''
                From the information provided by the heatmap, you can infer that:
                - for "entire apt" reservations, more cancellations are available with flexible and moderate policies than strict policies
                - for "hotel room" type reservations, cancellations are relatively low and similar among different cancellation policies
                - for private room type reservations, cancellations are similar between flexible, moderate and strict policies
                - for shared room reservations, cancellations are similar among the different cancellation policies, with a slight difference between flexible and moderate versus strict policies.
            ''')


    with tab2:
        contingency_table = pd.crosstab(cleaned_df['room_type'], cleaned_df['neighbourhood_group'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(contingency_table, annot=True, cmap='BuPu', fmt='d')
        #plt.title('Heatmap of contingency tables between room_type and neighourhood group')
        plt.xlabel('Neighbourhood group NYC')
        plt.ylabel('Room Type')
        st.pyplot(plt.gcf())

        st.markdown('''
                From this heatmap, it's possible to draw some conclusions.:
                - It can be seen that the types of apartments offered on Airbnb are differentially distributed in different neighborhoods of New York City. For example, Manhattan seems to have a higher concentration of whole houses and private rooms than other types.
                - It's possible to see which type of apartment is more common in each neighborhood. For example, in Manhattan there might be a greater presence of whole houses than other types, while in Staten Island private rooms might be more common.
            ''') 
        
    with tab3:
        neighbourhood_by_most_Airbnbs = cleaned_df['neighbourhood'].value_counts()
        top_10_neighbourhood = neighbourhood_by_most_Airbnbs.head(10).index
        cleaned_df_top_10_neighbourhood = cleaned_df[cleaned_df['neighbourhood'].isin(top_10_neighbourhood)]
        contingency_table = pd.crosstab(cleaned_df_top_10_neighbourhood['neighbourhood'], cleaned_df_top_10_neighbourhood['neighbourhood_group'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(contingency_table, annot=True, cmap='BuPu', fmt='d')
        #plt.title('Heatmap of contingency tables between neighbourhood and neighbourhood group')
        plt.xlabel('Neighbourhood group NYC')
        plt.ylabel('Neighbourhood')
        st.pyplot(plt.gcf())  

        st.markdown('''
                This heatmap shows only the columns for 'Brooklyn' and 'Manhattan' because the selected top 10 neighborhoods belong only to these two neighborhood groups. If the top 10 neighborhoods were distributed across all five neighborhood groups, then the heatmap would also show the columns corresponding to the other neighborhood groups such as 'Queens', 'Bronx' and 'Staten Island'.
                    ''')
        
    with tab4:
        contingency_table = pd.crosstab(cleaned_df['host_identity_verified'], cleaned_df['neighbourhood_group'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(contingency_table, annot=True, cmap='BuPu', fmt='d')
        #plt.title('Heatmap of contingency tables between host_identity_verified and neighourhood group')
        plt.xlabel('Neighbourhood group NYC')
        plt.ylabel('host_identity_verified')
        st.pyplot(plt.gcf())

        st.markdown('''
                From this heatmap, it's possible to draw some conclusions.:
                it's possibile to clearly see that Manhattan and Brooklyn are the neighborhood groups in which the highest number of Airbnb hosts are found, both for verified and unverified hosts. This might indicate a higher popularity of these areas for hosting on Airbnb than the other neighborhood groups.
                The Bronx, Queens, and Staten Island neighborhoods have the fewest airbnbs, as seen above.
                It might be interesting to further examine how identity verification affects guest reviews.
            ''')
        

############
#EDA
############
elif current_tab == "📊 Exploratory Data Analysis": 
    st.title("Exploratory Data Analysis")


     
    


    

