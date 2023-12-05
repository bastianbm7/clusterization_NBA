# Start connection with R
import os 
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.express as px

from streamlit_option_menu import option_menu
from streamlit_extras.colored_header import colored_header
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.grid import grid 
from streamlit_extras.dataframe_explorer import dataframe_explorer 

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.metrics import davies_bouldin_score

from codes import graphs
# os.environ['R_HOME'] = f'C:\Program Files\R\R-4.3.2'

# %load_ext rpy2.ipython
#import rpy2.rinterface
#
#from rpy2.robjects import r, pandas2ri
#from rpy2.robjects.packages import importr
#import rpy2.robjects as robjects
#from rpy2.robjects.conversion import localconverter
#import rpy2.robjects as ro

# Install necesary packages on R

# utils.install_packages('dplyr') 
# utils.install_packages('tidyr') 
# utils.install_packages('dbscan') 


# Import packages on R
#utils = importr('utils')
#base = importr('base')
#dplyr = importr('dplyr')
#tidyr = importr('tidyr')
#dbscan = importr('dbscan')

def home():
    st.markdown("""## Análisis Interactivo de Clústers de Jugadores de la NBA en el periodo de 2013 a 2023 mediante Métodos de Agrupamiento Basados en Densidad
#### Objetivo general:
- Crear un tablero interactivo para comparar dos métodos de clustering basado en densidades con la capacidad de modificar los parámetros para los datos de las estadísticas de los jguadores de la NBA entre 2013 a 2023

#### Objetivos específicos:
- Aplicar clustering DBSCAN y OPTICS para identificar jugadores similares según la distribución de densidades de las variables seleccionadas
- Identificar parámetros óptimos para que los jugadores entre grupos sean similares, mientras que la distancia entre grupos sea la mayor posible.
                """)
# Create DBSCAN Page
def page_DBSCAN(df):

    # ----------- Create metrics ------------
    # Metrics added: Noisy data
    #                Total of clusters
    def metrics_sidebar(total_clusters, noise_quantity):
        
        # Create two columns 
        col1, col2 = st.columns(2)
        # Noise column
        col1.metric("Datos de ruido", f'{noise_quantity}')
        
        # Total clusters column
        col2.metric("Total de clusters", f'{total_clusters}')
        
        # Style metric cards
        style_metric_cards(border_left_color="#0a0a0a", border_color="#0a0a0a", border_radius_px=25)

    # -------- Add metrics on top of the page ----------
    # Metrics added: Silohuette Score
    #                Davies-Bouldin Index
    def add_top_metrics(df):
        # Get sillohuette score
        predicts = dbscan.fit_predict(df)
    
        # Check if there is mos than 1 group
        unique_labels = len(set(predicts))
        if unique_labels > 1:
            silohuette_score = metrics.silhouette_score(df, predicts)
        else:
            silohuette_score = None

        # Calculate Davies-Bouldin Index
        db_index = davies_bouldin_score(df, predicts)

        # Create two columns in page
        col1, col2 = st.columns(2)

        # Check if is silohuette score is valid 
        # and write silohuette score metric
        if silohuette_score is not None:
            col1.metric("Silohuette Score", f'{round(silohuette_score,3)}')
        else:
            col1.metric("Silohuette Score", 'No es válido')
        
        # Write Davies-Bouldin Index
        col2.metric("Davies-Bouldin Index", f'{round(db_index, 3)}')
        # Style metric cards
        style_metric_cards(border_left_color="#0a0a0a", border_color="#0a0a0a", border_radius_px=25)


    
    # ---------- Make buble chart ---------------
    def graph_bubble(df, x, y, labels):

        # Get labeled data
        df['labels'] = labels
        
        # Convert 'labels' column to categorical
        df['labels'] = pd.Categorical(df['labels'])
        
        # Count of values based on x and y values
        bubble_sizes = df.groupby([x, y, 'labels']).size().reset_index(name='count')
        
        # Join values of bubble_sizes
        df = pd.merge(df, bubble_sizes, on=[x, y, 'labels'], how='left')
        
        # Create bubble chart in plotly
        fig = px.scatter(df, x=x, y=y, 
                         size='count', color='labels',
                         color_discrete_sequence=px.colors.qualitative.Set1,
                         title=f'Cantidad de jugadores en cada clúster',
                         labels={'x': f'Valores de {x}', 'y': f'Valores de {y}'})
        
        return fig
        
    #  ----- Make heatmap graph -----                       
    def graph_heatmp(df, x, y):
        # Count of values based on x and y values
        bubble_sizes = df.groupby([f'{x}', f'{y}']).size().reset_index(name='count')
        
        # Join values of bubble_sizes
        df_ = pd.merge(df, bubble_sizes, on=[f'{x}', f'{y}'], how='left')
        
        fig = px.density_heatmap(df_, x=f'{x}', y=f'{y}', z='count')
        return fig

    #  ----- Make boxplot graph ----- 
    def graph_boxplot(df, labels):

        # Labels predicted
        df['labels'] = labels
        
        # Make fig with plotly
        fig = px.box(df, x='labels', y=y, color='labels',
                    title=f'Zonas con mayor aglomeración de jugadores')
        return fig

    
    # ------ Quantity of players by cluster graph ------ 
    def graph_prop_by_cluster(df):
        # Crear un DataFrame con las etiquetas y contar la cantidad en cada grupo
        group_counts = labels.value_counts().reset_index()
        group_counts.columns = ['Group', 'Count']
        group_counts = group_counts.sort_values(by='Group')
        
        # Crear el gráfico de barras con Plotly
        fig = px.bar(group_counts, x='Group', y='Count', labels={'Group': 'Cluster', 'Count': 'Número de Individuos'},
                 title='Cantidad de Individuos en cada Grupo después de DBSCAN')
    
        # Mostrar el gráfico
        return fig, group_counts


    
    # --------- Make graph to determine minPts --------
    def graph_minPts(df, eps):
        min_pts_values = range(1, 11)
        silhouette_scores = []
    
        for min_pts in min_pts_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_pts)

            predicts = dbscan.fit_predict(df)
            
            # Verificar si hay al menos dos grupos
            unique_labels = len(set(predicts))
            if unique_labels > 1:
                silhouette_scores.append(metrics.silhouette_score(df, predicts))
            else:
                silhouette_scores.append(None)
    
        fig, ax = plt.subplots()
        ax.plot(min_pts_values, silhouette_scores, marker='o', linestyle='-')
    
        ax.set_title('Método del Codo para DBSCAN')
        ax.set_xlabel('MinPts')
        ax.set_ylabel('Silhouette Score')
        ax.legend().set_visible(False)
    
        plt.show()
        
        return fig

    
    # ----- K distance graph -----
    def graph_kdistance(df, total_clusters):

        # Calculate the distances to the k-nearest neighbors
        neigh = NearestNeighbors(n_neighbors=total_clusters)
        neigh.fit(df)
        distances, _ = neigh.kneighbors(df)
        # Ordena las distancias y crea el gráfico en Plotly
        sorted_distances = np.sort(distances[:, -1])
    
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, len(sorted_distances) + 1), sorted_distances, marker='o', color='blue', markersize=8)
    
        ax.set_title('K-Distance Graph')
        ax.set_xlabel('Data Point Index')
        ax.set_ylabel(f'{total_clusters}-th Nearest Neighbor Distance')
    
        # Devuelve la figura
        return fig


    ###########################################
    #  ----- Write in sidebar in DBSCAN page ----- 
    with st.sidebar:
        
        # Slider with eps value
        eps = st.slider('Select eps', 0.0, 20.0, 2.0, 0.2)

        # Slider with min_samples vlaue
        min_samples = st.slider('min_samples', 1, 20, 2)

    
    # ----- Apply DBSCAN with specified parameters ----- 
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(df)

    # Get labels created
    labels = dbscan.labels_
    
    # Transform labels to pandas format
    labels = pd.Series(labels)
    labels = labels + 1
    
    # Get total of clusters created ------
    # Check if noise cluster is created or not
    if 0 in labels.values:
        total_clusters = len(labels.unique()) - 1
    else: 
        total_clusters = len(labels.unique())
    
    # Get quantity of noisy rows created
    noise_quantity = (labels == 0).sum()

    
    ###########

    
    # Add metrics to sidebar in DBSCAN page----- 
    with st.sidebar:
        # Show metric cards
        metrics_sidebar(total_clusters, noise_quantity)

    # ----- Create main page----- 
    st.markdown("# DBSCAN")

    # Add Silohuette and Davies Boundis metrics
    add_top_metrics(df)

    # Make two columns to the selectbox of x and y
    col1, col2 = st.columns(2)
        
    # Selectbox of X variable
    x = col1.selectbox('Selecciona variable X', 
                           (df.columns), 
                           index = 0)
    # Selectbox of y variable
    y = col2.selectbox('Selecciona variable Y', 
                           (df.columns),
                          index = 1)

    # Create graphs to use
    fig_bar, df_bar = graph_prop_by_cluster(df)
    fig_minPts = graph_minPts(df, eps)
    fig_kdistance = graph_kdistance(df, total_clusters)
    fig_buble = graph_bubble(df, x, y, labels)
    fig_heatmap = graph_heatmp(df, x, y)
    fig_boxplot = graph_boxplot(df, labels)

    # Create a 1x2 grid to write the graphs
    my_grid = grid([3,3], 1, vertical_align="center")
    
    # charts
    my_grid.plotly_chart(fig_buble, theme="streamlit", use_container_width=True)
    my_grid.plotly_chart(fig_heatmap, theme="streamlit", use_container_width=True)
    my_grid.plotly_chart(fig_boxplot, theme="streamlit", use_container_width=True)

    # Create two tabs with different topics
    tab1, tab2 = st.tabs(["Seleccionar parámetros", "Resultados"])
    
    #  Write in tab1 ----------------------
    with tab1:

        # Create a 2x2 grid to write the graphs
        my_grid = grid([3,2], 2, vertical_align="center")
    
        # Row 1:
        my_grid.plotly_chart(fig_bar, theme="streamlit", use_container_width=True)
        my_grid.dataframe(df_bar, use_container_width = True)
        
        # Row 2:
        my_grid.pyplot(fig_minPts, clear_figure  = True, use_container_width=True)
        my_grid.pyplot(fig_kdistance,clear_figure  = True, use_container_width=True)

    # Write in tab 2 ----------------------
    with tab2:
        # Write buble chart
        st.plotly_chart(fig_buble, theme="streamlit", use_container_width=True)



# Create DBSCAN Page
#def OPTICS():