import pandas as pd
import numpy as np

import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.dataframe_explorer import dataframe_explorer 

from sklearn.preprocessing import StandardScaler, RobustScaler

from codes import pages


# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("Codes/style.css")

def filter_data(df_):
    # Choose the columns to use
    columns_to_select = st.multiselect(
            'Selecciona las variables que se utilizarán',
            df_.columns,
            default  = df_.columns.tolist())         
            
    df = df_[columns_to_select]
    
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

    return df

# Read data
df = pd.read_csv('data\\PPG_data.csv')

# Create menu on top of the page with all pages:
# 1.- Home 
# 2.- DBSCAN
# 3.- OPTICS
menu_option = option_menu(None, ["Home", "DBSCAN", "OPTICS", "SNNClust", "Resultados"],
                          icons=['house-fill', 'bar-chart-fill', 'folder-fill', 'send-fill'],
                          menu_icon="cast", default_index=0, orientation="horizontal",
                          styles={
                              "container": {"padding": "0!important", "background-color": "#fafafa"},
                                "icon": {"color": "orange", "font-size": "25px"},
                                  "nav-link": {"font-size": "17px", "text-align": "left", "margin": "10px", "--hover-color": "#fffff"},
                                  "nav-link-selected": {"background-color": "black"},
                        })


# ---------------------
# Page 1: Home
if menu_option == "Home":
    pages.home()

with st.sidebar:
    df = filter_data(df)
# ---------------------
# Page 2: DBSCAN
if menu_option == 'DBSCAN':
    pages.page_DBSCAN(df)