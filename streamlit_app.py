import pandas as pd
import numpy as np

import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.dataframe_explorer import dataframe_explorer 
from streamlit_extras.colored_header import colored_header

from sklearn.preprocessing import StandardScaler, RobustScaler
import random



def filter_data(df_):
    # Choose the columns to use
    columns_to_select = st.multiselect(
            'Selecciona las variables que se utilizarán',
            df_.columns,
            default  = [df_.columns.tolist()[0], df_.columns.tolist()[1]])         
    df = df_[columns_to_select]

    # Change quantity of data:
    n = st.number_input('Selecciona la cantidad de datos', value = 100)
    df = df.sample(n=n, random_state = 3, replace = False)
    
    # Create a df to filter data based on columns values
    df = dataframe_explorer(df, case=False)

    
    # Select a scaler and scale data frame based on select choice
    scaler = st.radio("Selecciona el escalado de los datos",
                      ["Sin escalar", "Estándar", "Robusto"], horizontal = True )
    
    # Non scaler
    if scaler == 'Sin escalar':
        pass
    # Standard scaler
    elif scaler == "Estándar":
        scaler_standard = StandardScaler()
        df = pd.DataFrame(scaler_standard.fit_transform(df),
                          columns=df.columns)
    # Robust scaler
    elif scaler == "Robusto":
        scaler_robust = RobustScaler()
        df = pd.DataFrame(scaler_robust.fit_transform(df),
                          columns=df.columns)

    return df, columns_to_select

# Read data
df = pd.read_csv('https://raw.githubusercontent.com/bastianbm7/clusterization_NBA/main/data/PPG_data.csv')

# Create menu on top of the page with all pages:
# 1.- Home 
# 2.- DBSCAN
# 3.- OPTICS
menu_option = option_menu(None, ["Home", "DBSCAN", "HDBSCAN"],
                          icons=['house-fill', 'bar-chart-fill', 'folder-fill', 'send-fill'],
                          menu_icon="cast", default_index=0, orientation="horizontal",
                          styles={
                              "container": {"padding": "0!important", "background-color": "#fafafa"},
                                "icon": {"color": "#E74C3C ", "font-size": "25px"},
                                  "nav-link": {"font-size": "17px", "text-align": "left", "margin": "12px", "--hover-color": "#fffff"},
                                  "nav-link-selected": {"background-color": "#95A5A6"},
                        })
        
# ---------------------
# Page 1: Home
if menu_option == "Home":
    pages.home()

with st.sidebar:
    colored_header(
        label="Filtra tus datos",
        description="Cambia los parámetros a tu medida",
        color_name="red-90",
        )
    df, columns_to_select = filter_data(df)
# ---------------------
# Page 2: DBSCAN
if menu_option == 'DBSCAN':
    pages.page_DBSCAN(df)

if menu_option == 'HDBSCAN':
    pages.HDBSCAN_page(df)
    

