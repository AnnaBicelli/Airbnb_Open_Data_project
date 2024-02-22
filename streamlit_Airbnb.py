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
from streamlit_folium import st_folium
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

######################
#Functions and dataset
######################
    
def clean_data(df):
     
    cleaned_df = df.copy()
    
    # columns
    cleaned_df.columns = cleaned_df.columns.map(lambda x: x.lower().replace(' ', '_'))
    
    # remove '$' e ',' from 'price' e 'service_fee' and  convert in float
    cleaned_df['price'] = cleaned_df['price'].replace({'\$': '', ',': ''}, regex=True).astype(float)
    cleaned_df['service_fee'] = cleaned_df['service_fee'].replace({'\$': '', ',': ''}, regex=True).astype(float)
    
    #transform instant_bookable in bool
    cleaned_df['instant_bookable'] = cleaned_df['instant_bookable'].astype(bool)
    
    # mean
    cleaned_df['lat'] = cleaned_df['lat'].fillna(cleaned_df['lat'].mean())
    cleaned_df['long'] = cleaned_df['long'].fillna(cleaned_df['long'].mean())
    cleaned_df['construction_year'] = cleaned_df['construction_year'].fillna(cleaned_df['construction_year'].mean())
    cleaned_df['minimum_nights'] = cleaned_df['minimum_nights'].fillna(cleaned_df['minimum_nights'].mean())
    cleaned_df['number_of_reviews'] = cleaned_df['number_of_reviews'].fillna(cleaned_df['number_of_reviews'].mean())
    cleaned_df['review_rate_number'] = cleaned_df['review_rate_number'].fillna(cleaned_df['review_rate_number'].mean())
    cleaned_df['calculated_host_listings_count'] = cleaned_df['calculated_host_listings_count'].fillna(cleaned_df['calculated_host_listings_count'].mean())
    cleaned_df['availability_365'] = cleaned_df['availability_365'].fillna(cleaned_df['availability_365'].mean())
    
    # Remove "reviews_per_month"
    cleaned_df.drop(columns=["reviews_per_month"], inplace=True)

    #dprona 'price', 'service_fee' e 'host_identity_verified'
    cleaned_df.dropna(subset=['price'],inplace=True)
    cleaned_df.dropna(subset=['service_fee'],inplace=True)
    cleaned_df.dropna(subset=['host_identity_verified'],inplace=True)
    
    # mode
    cleaned_df['name'] = cleaned_df['name'].fillna(cleaned_df['name'].mode()[0])
    cleaned_df['host_name'] = cleaned_df['host_name'].fillna(cleaned_df['host_name'].mode()[0])
    cleaned_df['neighbourhood_group'] = cleaned_df['neighbourhood_group'].fillna(cleaned_df['neighbourhood_group'].mode()[0])
    cleaned_df['neighbourhood'] = cleaned_df['neighbourhood'].fillna(cleaned_df['neighbourhood'].mode()[0])
    cleaned_df['country'] = cleaned_df['country'].fillna(cleaned_df['country'].mode()[0])
    cleaned_df['country_code'] = cleaned_df['country_code'].fillna(cleaned_df['country_code'].mode()[0])
    cleaned_df['cancellation_policy'] = cleaned_df['cancellation_policy'].fillna(cleaned_df['cancellation_policy'].mode()[0])
    cleaned_df['room_type'] = cleaned_df['room_type'].fillna(cleaned_df['room_type'].mode()[0])
    
    # Remove "license", "house_rules", "last_review"
    cleaned_df.drop(columns=["license", "house_rules", "last_review"], inplace=True)

    # 'neighbourhood_group'
    cleaned_df["neighbourhood_group"] = cleaned_df["neighbourhood_group"].replace({"brookln": "Brooklyn"})
    cleaned_df["neighbourhood_group"] = cleaned_df["neighbourhood_group"].replace({"manhatan": "Manhattan"})
    
    #transform 'construction_year' in int
    cleaned_df.construction_year = cleaned_df.construction_year.astype(int)
    
    #  select the rows with negative 'minimum_nights' and assign them the value 1
    cleaned_df.loc[cleaned_df['minimum_nights'] < 0, 'minimum_nights'] = 1

    cleaned_df=cleaned_df.replace('#NAME?', np.nan)
    
    return cleaned_df
    


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
    
    st.write('It reports the listing activity of homestays in New York City, their reviews, prices, availability, location, room types and cancellation policies.')

    selected_columns = st.multiselect('Explore the Airbnb dataset by selecting columns', airbnb_df.columns)
    if selected_columns:
        columns_df = airbnb_df.loc[:, selected_columns]
        st.dataframe(columns_df.head(15)) #option bar
    else:
        st.dataframe(airbnb_df.head(15))  
    

    st.write('General informations about the DataFrame')
    # Creating a buffer to capture information on the Airbnb DataFrame
    buffer = io.StringIO()
    airbnb_df.info(buf=buffer)
    s = buffer.getvalue()
    # Show multiselect to select columns to display
    selected_columns1 = st.multiselect("Select the variables", airbnb_df.columns.tolist(), default=airbnb_df.columns.tolist())

    # If columns are selected, it shows information only for those columns
    if selected_columns1:
        selected_info_buffer = io.StringIO()
        airbnb_df[selected_columns1].info(buf=selected_info_buffer)
        selected_info = selected_info_buffer.getvalue()
        st.text(selected_info)
    else:
        # Otherwise, it shows the information for all columns
        st.text(s)

#########
#Cleaning
#########
elif current_tab == "🗑️ Cleaning": 
    st.title("Cleaning NA values")
   
    st.write('Before proceeding with the analysis, the null values in the dataset were analyzed and then replaced or eliminated.')

    cleaned_df = clean_data(airbnb_df)

    tab1, tab2, tab3, tab4 = st.tabs(["NA values", "Cleaning", "-", "-"]) 
    
    with tab1:       
        
        # Calculates the count of missing values and the percentage of missing values for each variable
        missing_values_count = airbnb_df.isna().sum()
        total_values = airbnb_df.shape[0]
        missing_values_percentage = (missing_values_count / total_values) * 100

        # Round the percentage to two decimal places
        missing_values_percentage = missing_values_percentage.round(2)

        # Create a new DataFrame with the count and percentage of missing values
        missing_df = pd.DataFrame({
            'Variable': missing_values_count.index,
            'NA values': missing_values_count.values,
            '%  NA values': missing_values_percentage.values
        })

        # Show the DataFrame of missing values
        st.write(missing_df)

    with tab2:
        st.markdown('''
                Based on these information the data will be rearranged as follows:
                - **fixed** the column indexes
                - **replaced** the null values related to *float* data with their **mean value** because the percentages were really close to zero
                - **replaced** the null values related to *object* variables with the **mode**, so it doesn't change the distribution a lot
                - **removed** the NaN values in the columns `price`, `service_fee` and `host_identity_verified`
                -  **removed** the columns `reviews_per_month`, `last_review`,`license` and `house_rules` because the percentage of null values was excessive and would not lead to useful information
                - **replaced** the negative cells in the column `minimum_nights` with 1, because a guest stays at least one night in the Airbnb
                - **replaced**  the values *brookln* and *manahtan* in the `neighbourhood_groups` column with Brooklyn and with Manhattan
                - **transformed** the vaors in the `construction_year` column from floats to *integers*, since these are years
                
                                ''') 
        
        
        with st.expander('Resulted DataFrame Preview'):
            st.write(cleaned_df.head(15)) 

############
#Correlation
############
elif current_tab == "🔗 Correlation": 
    st.title("Correlations between values")
    
     # Pulisci i dati
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
    #Define the plot_scatter function 
    def plot_scatter(cleaned_df):
        plt.figure(figsize=(6, 3))
        sns.set_palette("husl")
        sns.scatterplot(data=cleaned_df, x='service_fee', y='price', color='purple')
        plt.xlabel('Service Fee')
        plt.ylabel('Price')
        plt.title('Relationship between Service Fee and Price')
        st.pyplot(plt.gcf())  # 

    # Initializes or updates the state when the button is pressed
    if 'show_scatter' not in st.session_state:
        st.session_state.show_scatter = False

    if st.button('Click to see the Scatterplot'):
        st.session_state.show_scatter = not st.session_state.show_scatter

    # Shows or hides the graph based on the state of the session
    if st.session_state.show_scatter:
        plot_scatter(cleaned_df)  
    
    st.divider() 
    st.write('To observe how the main categorical variables relate to each other, contingency tables were used, which help identify areas of higher or lower frequency of combinations between types of variables:')

    tab1, tab2, tab3, tab4 = st.tabs(["Room_type vs Cancellation_policy", "Room_type vs Neighbourhood_group", "Neighbourhood vs Neighbourhood_group", "Host_identity_verified vs Neighbourhood_group"]) 
    
    with tab1:
            contingency_table = pd.crosstab(cleaned_df['room_type'], cleaned_df['cancellation_policy'])
            plt.figure(figsize=(8, 6))
            sns.heatmap(contingency_table, annot=True, cmap='BuPu', fmt='d')
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
        plt.xlabel('Neighbourhood group NYC')
        plt.ylabel('Room Type')
        st.pyplot(plt.gcf())

        st.markdown('''
                From this heatmap, it's possible to draw some conclusions:
                - it can be seen that the types of apartments offered on Airbnb are differentially distributed in different neighborhoods of New York City. For example, Manhattan seems to have a higher concentration of whole houses and private rooms than other types.
                - it's possible to see which type of apartment is more common in each neighborhood. For example, in Manhattan there might be a greater presence of whole houses than other types, while in Staten Island private rooms might be more common.
            ''') 
        
    with tab3:
        neighbourhood_by_most_Airbnbs = cleaned_df['neighbourhood'].value_counts()
        top_10_neighbourhood = neighbourhood_by_most_Airbnbs.head(10).index
        cleaned_df_top_10_neighbourhood = cleaned_df[cleaned_df['neighbourhood'].isin(top_10_neighbourhood)]
        contingency_table = pd.crosstab(cleaned_df_top_10_neighbourhood['neighbourhood'], cleaned_df_top_10_neighbourhood['neighbourhood_group'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(contingency_table, annot=True, cmap='BuPu', fmt='d')
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
        plt.xlabel('Neighbourhood group NYC')
        plt.ylabel('host_identity_verified')
        st.pyplot(plt.gcf())

        st.markdown('''
                From this heatmap, it's possible to draw some conclusions:\n
                - it's possibile to clearly see that Manhattan and Brooklyn are the neighborhood groups in which the highest number of Airbnb hosts are found, both for verified and unverified hosts. This might indicate a higher popularity of these areas for hosting on Airbnb than the other neighborhood groups.
                - the Bronx, Queens, and Staten Island neighborhoods have the fewest Airbnbs.
            ''')
        

############
#EDA
############
elif current_tab == "📊 Exploratory Data Analysis": 
    st.title("Exploratory Data Analysis")
    
    cleaned_df = clean_data(airbnb_df)    
    
    st.write('First of all, knowing that  New York City is composed by five major neighborhood groups, which are Brooklyn, Manhattan, Queens, Bronx and Staten Isalnd, is appropriate to identify how the Airbnb located in the various neighborhood groups are divided.')
    neighbourhood_group_counts = cleaned_df['neighbourhood_group'].value_counts()
    explode = [0.01] * len(neighbourhood_group_counts)
    palette = ['powderblue', 'mediumpurple', 'lightcoral', 'mediumseagreen', 'sandybrown']
    plt.figure(figsize=(6,5)) 
    plt.pie(neighbourhood_group_counts, explode = explode, labels=neighbourhood_group_counts.index ,  autopct='%1.1f%%', startangle=30, colors=palette, textprops={'fontsize': 6})
    plt.title('Number of Airbnbs in each Neighbourhood Group', fontsize=7)
    st.pyplot(plt.gcf())

    ####################################################################################################################################################
    st.write(' The neighborhoods with the highest density of Airbnb are:')
    
    neighbourhood_by_most_Airbnbs = cleaned_df.neighbourhood.value_counts()
    top_10_neighbourhood = neighbourhood_by_most_Airbnbs.head(10)
    palette = 'Set2'
    plt.figure(figsize=(10,6))
    x = top_10_neighbourhood.index 
    y = top_10_neighbourhood # Number of neighbourhoods
    plt.title('Top 10 neighbourhood with the most number of Airbnbs')
    plt.bar(x, y, color=plt.get_cmap(palette)(range(len(x))))
    plt.xticks(rotation=45, ha='right') 
    plt.ylabel('Number of Airbnb')
    plt.xlabel('Neighbourhood')
    for i, v in enumerate(y):
        plt.text(i, v + 20, str(v), ha='center', va='bottom')
    plt.tight_layout()
    st.pyplot(plt.gcf())

    ##################################################################################################################################################################
    st.write('Therefore, the distribution of neighborhoods in the neighborhood groups is:')
        
    neighbourhood_by_most_Airbnbs = cleaned_df['neighbourhood'].value_counts()
    top_10_neighbourhood = neighbourhood_by_most_Airbnbs.head(10).index
    cleaned_df_top_10_neighbourhood = cleaned_df[cleaned_df['neighbourhood'].isin(top_10_neighbourhood)]
    
    plt.figure(figsize=(6,4))
    contingency_table = pd.crosstab(cleaned_df_top_10_neighbourhood['neighbourhood_group'], cleaned_df_top_10_neighbourhood['neighbourhood']) 
    pastel_colors = ['#FFB6C1', '#FFD700', '#98FB98', '#ADD8E6', '#FFA07A', '#87CEFA', '#F08080', '#20B2AA', '#9370DB', '#FF69B4']
    contingency_table.plot(kind='bar', stacked=True, figsize=(12, 8), color=pastel_colors)
    plt.title('Distribution of neighbourhoods by neighbourhood_group')
    plt.xlabel('Neighbourhood Group')
    plt.ylabel('Number of Airbnbs')
    plt.xticks(rotation=45)  
    plt.legend(title='Neighbourhood')
    st.pyplot(plt.gcf())

    ######################################################################################################################################################
    
    tab1, tab2 = st.tabs(["Top 20 Airbnb names", "Top 20 hosts names"])
    with tab1:
        st.write('The number of Airbnb within the dataframe is very large, and the most popular apartment names are:')
        
        name_for_most_Airbnbs = cleaned_df.name.value_counts()
        top_20_names = name_for_most_Airbnbs.head(20)

        plt.figure(figsize=(10,8))
        x = top_20_names.index 
        y = top_20_names # Number of neighbourhoods
        plt.title('Top 20 names for Airbnbs')
        bars = plt.bar(x, y, color=plt.cm.magma(np.linspace(0, 1, len(x))))
        plt.xticks(rotation=45, ha='right') #with 'right' I'm sure that the xticks are under the corresponding bar
        plt.ylabel('Number of Airbnbs with the name')
        plt.xlabel('Name of the Airbnbs')
        for bar, v in zip(bars, y):
            plt.text(bar.get_x() + bar.get_width() / 2, v + 20, str(v), ha='center', va='bottom')
        plt.tight_layout()
        st.pyplot(plt.gcf())

        st.write('The most popular airbnb name in New York is **Home away from home**.')

    with tab2:
        st.write('The hosts who own the most Airbnbs in New York City are:')

        host_name = cleaned_df.host_name.value_counts()
        top_20__host_names = host_name.head(20)

        plt.figure(figsize=(10,8))
        x = top_20__host_names.index 
        y = top_20__host_names # Number of neighbourhoods
        plt.title('Top 20 host names')
        bars = plt.bar(x, y, color=plt.cm.magma(np.linspace(0, 1, len(x)))) 
        plt.xticks(rotation=45, ha='right') #with 'right' I'm sure that the xticks are under the corresponding bar
        plt.ylabel('Number of airbnbs of that host')
        plt.xlabel('Host names')
        for bar, v in zip(bars, y):
            plt.text(bar.get_x() + bar.get_width() / 2, v + 20, str(v), ha='center', va='bottom')
        plt.tight_layout()
        st.pyplot(plt.gcf())

        st.write('The name of the host with the most Airbnb in New York is **Michael**.')

    #############################################################################################################################################################
    st.write('The following plots were created showing the maximum and minimum daily price for each year of Airbnb construction from 2002 to 2022. This representation can be useful in assessing whether hosts of more recently built Airbnbs demand higher prices and deviate from the maximum price of older built apartments.')

    plt.figure(figsize=(10, 5))
    cleaned_df.groupby('construction_year')['id'].count().plot(kind='line')
    plt.title('Construction years of Airbnbs')
    plt.xlabel('Construction year')
    plt.ylabel('Number of Airbnb accommodations')
    plt.legend()
    plt.grid(True, axis = 'y')
    st.pyplot(plt.gcf())
    st.write('The largest number of airbnbs were built in 2012.')

    #############################################################################################################################################################
    st.write('The minimum and maximum daily prices of Airbnb built from 2002 to 2022 were also investigated to check whether the higher price had a close relationship with the recent construction of the apartment')

    max_price_per_year= cleaned_df.groupby('construction_year')['price'].max()
    min_price_per_year= cleaned_df.groupby('construction_year')['price'].min()

    fig, axs = plt.subplots(2, 1, figsize=(8,12))

    # Plot for maximum daily price by year
    axs[0].plot(max_price_per_year.index, max_price_per_year.values, marker='o', color='mediumpurple')
    axs[0].set_xlabel('Construction Year')
    axs[0].set_ylabel('Maximum Price')
    axs[0].set_title('Maximum Daily Price per Construction Year')
    axs[0].grid(True, axis='y')

    # Adding price tags to markers above or below in order to see better
    for year, price in max_price_per_year.items():
        if price > 1199:
            axs[0].text(year, price, f'${price:.0f}', rotation=45, ha='right', va='top')
        else:
            axs[0].text(year, price, f'${price:.0f}', rotation=45, ha='right', va='bottom')

    # Formatting the y-axes as integers
    axs[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))

    # Plot for minimum daily price by year
    axs[1].plot(min_price_per_year.index, min_price_per_year.values, marker='o', color='mediumpurple')
    axs[1].set_xlabel('Construction Year')
    axs[1].set_ylabel('Minimum Price')
    axs[1].set_title('Minimum Daily Price per Construction Year')
    axs[1].grid(True, axis='y')

    plt.tight_layout() 
    plt.legend()
    
    st.pyplot(plt.gcf())
    st.write('It is noted that the daily price is not strictly related to the year of construction; rather, for each year it goes back to an apartment with a minimum price of $50 and a maximum of $1199 or $1200.')

    #############################################################################################################################################################
    st.write('The map shows the most expensive Airbnbs, highlighted in red, and the least expensive, in green, in New York City. Each of these, when selected, indicates the neighborhood to which it belongs.')
    
    tab1, tab2 = st.tabs(["Folium chart", "Script"])
    with tab1:
        neighborhood_coordinates = cleaned_df.groupby('neighbourhood').agg({'lat': 'mean', 'long': 'mean'})
        max_price_per_neighborhood = cleaned_df.groupby('neighbourhood')['price'].max()
        max_price = max_price_per_neighborhood.max()
        most_expensive_neighborhoods = max_price_per_neighborhood[max_price_per_neighborhood == max_price].index.tolist()
        neighborhood_coordinates1 = cleaned_df.groupby('neighbourhood').agg({'lat': 'mean', 'long': 'mean'}) 
        min_price_per_neighborhood = cleaned_df.groupby('neighbourhood')['price'].min()
        min_price = min_price_per_neighborhood.min()
        less_expensive_neighborhoods = min_price_per_neighborhood[min_price_per_neighborhood == min_price].index.tolist()
        map_nyc = folium.Map(location=[40.76438361546867, -73.92027622745046], zoom_start=10)
        for neighborhood in most_expensive_neighborhoods:
            lat = neighborhood_coordinates.loc[neighborhood, 'lat']
            long = neighborhood_coordinates.loc[neighborhood, 'long']
            folium.Marker([lat, long], popup=neighborhood, tooltip='Price: $1200', icon=folium.Icon(color='red', icon='home', prefix='fa')).add_to(map_nyc)
        for neighborhood in less_expensive_neighborhoods:
            lat = neighborhood_coordinates.loc[neighborhood, 'lat']
            long = neighborhood_coordinates.loc[neighborhood, 'long']
            folium.Marker([lat, long], popup=neighborhood, tooltip='Price: $50', icon=folium.Icon(color='green', icon='home', prefix='fa')).add_to(map_nyc)
        st.data = st_folium(map_nyc, width=1200, height=480)
        
    with tab2:
        st.code('''
                from streamlit_folium import st_folium

                # neighbourhood selection

                neighborhood_coordinates = cleaned_df.groupby('neighbourhood').agg({'lat': 'mean', 'long': 'mean'})
                max_price_per_neighborhood = cleaned_df.groupby('neighbourhood')['price'].max()
                max_price = max_price_per_neighborhood.max()
                most_expensive_neighborhoods = max_price_per_neighborhood[max_price_per_neighborhood == max_price].index.tolist()
                neighborhood_coordinates1 = cleaned_df.groupby('neighbourhood').agg({'lat': 'mean', 'long': 'mean'}) 
                min_price_per_neighborhood = cleaned_df.groupby('neighbourhood')['price'].min()
                min_price = min_price_per_neighborhood.min()
                less_expensive_neighborhoods = min_price_per_neighborhood[min_price_per_neighborhood == min_price].index.tolist()

                # map

                map_nyc = folium.Map(location=[40.76438361546867, -73.92027622745046], zoom_start=10)
                for neighborhood in most_expensive_neighborhoods:
                    lat = neighborhood_coordinates.loc[neighborhood, 'lat']
                    long = neighborhood_coordinates.loc[neighborhood, 'long']
                    folium.Marker([lat, long], popup=neighborhood, tooltip='Price: $1200', icon=folium.Icon(color='red', icon='house', prefix='fa')).add_to(map_nyc)
                for neighborhood in less_expensive_neighborhoods:
                    lat = neighborhood_coordinates.loc[neighborhood, 'lat']
                    long = neighborhood_coordinates.loc[neighborhood, 'long']
                    folium.Marker([lat, long], popup=neighborhood, tooltip='Price: $50', icon=folium.Icon(color='green', icon='house', prefix='fa')).add_to(map_nyc)
                st.data = st_folium(map_nyc, width=800, height=480)
                ''')
    #################################################################################################################################################################################
    st.write('The Airbnb offered by hosts can be of 4 types and with the following plots it is possible to identify the quantity for each room typology and the most common room type in NYC and in each neighbourhood group.')
    
    tab1, tab2 = st.tabs(["New York City", "Neighbourhood groups"])
    with tab1:
        total_counts = cleaned_df['room_type'].value_counts()
        colors = colors = ['darkseagreen', 'sandybrown', 'mediumpurple', 'paleturquoise']
        plt.figure(figsize=(12,10))
        plt.pie(total_counts, labels=total_counts.index,  autopct='%1.1f%%', startangle=40, colors=colors, textprops={'fontsize': 8})
        plt.title('Proportion of apartment types in New York City.', fontsize=10)
        plt.legend()
        st.pyplot(plt.gcf())
        st.write('From this pie chart it was possible to observe that the solutions for which most hosts opt are entire apartments or private rooms')

    with tab2:
        num_rows = 3
        num_columns = 2
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(15,10))
        neighborhood_group_counts = cleaned_df.groupby(['neighbourhood_group', 'room_type']).size().unstack(fill_value=0) 
        colors = ['darkseagreen',  'mediumpurple','sandybrown', 'paleturquoise']
        # 1st plot
        axes[0, 0].pie(neighborhood_group_counts.loc['Manhattan'], labels=neighborhood_group_counts.columns, autopct='%1.1f%%',colors=colors, startangle=140, textprops={'fontsize': 10})
        axes[0, 0].set_title(f'Proportion of apartment types in the neighborhood Manhattan', fontsize = 10)
        # 2nd plot
        axes[0, 1].pie(neighborhood_group_counts.loc['Brooklyn'], labels=neighborhood_group_counts.columns, autopct='%1.1f%%',colors=colors, startangle=140,  textprops={'fontsize': 10})
        axes[0, 1].set_title(f'Proportion of apartment types in the neighborhood Brooklyn', fontsize = 11)
        # 3rd plot
        axes[1, 0].pie(neighborhood_group_counts.loc['Queens'], labels=neighborhood_group_counts.columns, autopct='%1.1f%%', startangle=140 ,colors=colors,  textprops={'fontsize': 10})
        axes[1, 0].set_title(f'Proportion of apartment types in the neighborhood Queens', fontsize = 10)
        # 4th plot
        axes[1, 1].pie(neighborhood_group_counts.loc['Bronx'], labels=neighborhood_group_counts.columns, autopct='%1.1f%%', startangle=140 ,colors=colors,  textprops={'fontsize': 10})
        axes[1, 1].set_title(f'Proportion of apartment types in the neighborhood Bronx', fontsize = 11)
        # 5th plot
        axes[2, 0].pie(neighborhood_group_counts.loc['Staten Island'], labels=neighborhood_group_counts.columns, autopct='%1.1f%%' ,colors=colors, startangle=140,  textprops={'fontsize': 10})
        axes[2, 0].set_title(f'Proportion of apartment types in the neighborhood Staten Island', fontsize = 10)
        # Eliminate the last excess axes
        axes[2, 1].axis('off')  
        plt.tight_layout()
        st.pyplot(plt.gcf())

        st.write('From this sublpot with pie charts related to neighborhood groups, it can be seen that most hosts offer entire apartments or private rooms')


    #########################################################################################################################################################################################################################
    st.write('As just seen, hosts offer different room solutions to their customers and do the same with cancellation policies as well. With the following chart, the differences between Airbnb s number of different room types based on the cancellation policy offered are highlighted.')
    
    tab1, tab2 = st.tabs(["Cancellation policies", "Room type vs Cancellatin policy"])
    with tab2:
        room_types = ['Entire apt', 'Hotel room', 'Private room', 'Shared room']
        flexible_counts = [17831, 43, 15293, 739]
        moderate_counts = [17874, 37, 15599, 742]
        strict_counts = [17749, 34, 15438, 734]

        plt.figure(figsize=(10, 6))
        bar_width = 0.2
        index = range(len(room_types))

        plt.bar(index, flexible_counts, bar_width, label='Flexible', color='pink')
        plt.bar([i + bar_width for i in index], moderate_counts, bar_width, label='Moderate', color='darkseagreen') 
        plt.bar([i + 2 * bar_width for i in index], strict_counts, bar_width, label='Strict', color='powderblue')
        plt.xlabel('Room Type')
        plt.ylabel('Number of Cancellations/Airbnbs con quel cancellation policy') 
        plt.title('Number of Airbns with different cancellations policies by Room Type and Cancellation Policy')
        plt.xticks([i + bar_width for i in index], room_types)
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt.gcf())  

    with tab1:
        policies = cleaned_df['cancellation_policy'].value_counts()
        explode=(0.05,0.05,0.05)
        colors = ['darkseagreen','powderblue',  'pink']

        plt.figure(figsize=(4,4))
        plt.pie(policies, labels=policies.index,colors=colors, explode = explode, autopct='%1.1f%%', startangle=40,  textprops={'fontsize': 7})
        plt.title('Cancellation policy strictness by percentage', fontsize=7)
        #plt.legend()
        st.pyplot(plt.gcf())
        st.write('There is a small difference between the cancellation policies guaranteed by hosts, the one most commonly used is the moderate one.')
    
    #############################################################################################################################################################
    st.write('Even if hosts do not confirm customers reservations, the guests can often proceed with the booking anyway. Instant bookable is not strictly tied to a room type  as can be seen from the following plot.')
    counts = cleaned_df.groupby(['instant_bookable', 'room_type']).size().unstack(fill_value=0)
    colors = ['plum', 'mediumseagreen'  , 'lightsteelblue','pink']

    plt.figure(figsize=(10, 6))
    counts.plot(kind='bar', stacked=True, color=colors)
    plt.title('Distribution of Instant Bookable by Room Type')
    plt.xlabel('Instant Bookable')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.legend(title='Room Type')
    st.pyplot(plt.gcf())
    st.write('the fact whether Airbnb is bookable instantly or later does not depend on the type of room because this graph  does not show a substantial difference.')

    ##################################################################################################################################################################################
    st.write('The next boxplot analyzes the possible changes in daily Airbnb prices in New York City according to the type of room they refer to. ')   
    
    plt.figure(figsize=(10,8))
    sns.boxplot(x='room_type', y='price', data=cleaned_df)
    plt.xlabel('Room Type')
    plt.ylabel('Price')
    plt.title('Price distribution by Room Type')
    st.pyplot(plt.gcf())
    st.write('It is observed that the median is similar for all solutions so the average prices are similar. Only for hotel room it is possible to observe that the prices are slightly higher.')
    ##############################################################################################################################################################################
    st.write('Regarding the rate obtained in the reviews:')
    
    tab1, tab2 = st.tabs(["By neighbourhood", "By verified identity"])
    with tab1:
        average_review_rate = cleaned_df.groupby('neighbourhood')['review_rate_number'].mean()
        best_neighbourhood = average_review_rate.idxmax() 
        best_review_rate = average_review_rate.max()
        average_review_rate_sorted = average_review_rate.sort_values(ascending=False)
        some_neighbourhood_average_review_rate = average_review_rate_sorted.head(15)
        plt.figure(figsize=(10, 8))
        some_neighbourhood_average_review_rate.plot(kind='barh', color='lightsteelblue')
        plt.xlabel('Average Review Rate')
        plt.ylabel('Neighbourhood')
        plt.title('Average Review Rate by Neighbourhood')
        plt.axhline(y=best_neighbourhood, color='red', linestyle='--', linewidth=2, label='Best Neighbourhood')
        plt.legend()
        st.pyplot(plt.gcf())
        st.write("The neighborhood with the best review rate on average is: **Glen Oaks**. \nIts average review rate is 4.5")

    with tab2:
        st.write('this plot analyzes whether host identity verification is important to guests and thus how it might affect the score left in reviews.')
        reviews_counts = cleaned_df.groupby(['host_identity_verified', 'review_rate_number']).size().unstack(fill_value=0)
        reviews_counts.plot(kind='bar',  figsize=(10, 6)) 
        plt.xlabel('Identity Verified')
        plt.ylabel('Number of Reviews')
        plt.title('Distribution of Reviews rate by Verified Identity')
        plt.legend(title='Reviews rate')
        plt.xticks(rotation=0)  
        st.pyplot(plt.gcf())
        st.write('This stacked plot shows that the number of reviews with each score appears to be similar between verified and unverified hosts. This observation might suggest that the identity verification status is not a determining factor in guests overall evaluations. Other factors, such as cleanliness, comfort, location, and hospitality, may have a more significant influence on guest evaluations. Alternatively, it could indicate a lack of awareness among guests regarding the host s identity confirmation process.')
       





