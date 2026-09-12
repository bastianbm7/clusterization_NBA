# Start connection with R
import os 
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.express as px

import seaborn as sns

import hdbscan

from streamlit_option_menu import option_menu
from streamlit_extras.colored_header import colored_header
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.grid import grid 
from streamlit_extras.dataframe_explorer import dataframe_explorer 

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from sklearn import metrics
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.decomposition import PCA


def home():
    
    colored_header(
        label="Análisis Interactivo de Clústers de Jugadores de la NBA en el periodo de 2013 a 2025 mediante Métodos de Agrupamiento Basados en Densidad",
        description="IECD421: Visualización de datos - Bastián Barraza",
        color_name="light-blue-90",
        )

    st.markdown("""#### Introducción

- El propósito principal de este panel actual es facilitar la aplicación de dos tipos de clustering por densidades de manera intuitiva. Esto posibilita la visualización de la variación de los clústeres al ajustar los parámetros de los algoritmos.

- Se han aplicado clustering basados en densidades a las estadísticas de los jugadores de la NBA durante las temporadas comprendidas entre los años 2013 y 2025. """)
    
    st.divider()
    st.markdown("""#### Objetivo general:
- Crear un tablero interactivo para comparar dos métodos de clustering basado en densidades con la capacidad de modificar los parámetros para los datos de las estadísticas de los jguadores de la NBA entre 2013 a 2025.

#### Objetivos específicos:
- Aplicar clustering DBSCAN y HDBSCAN para identificar jugadores similares según la distribución de densidades de las variables seleccionadas
- Identificar parámetros óptimos para que los jugadores entre grupos sean similares, mientras que la distancia entre grupos sea la mayor posible.""")

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
    def graph_bubble(df, x, y, dbscan):

        # Get labeled data
        df['labels'] = dbscan.labels_ + 1
        
        
        # Crea el gráfico con Plotly Express y asigna la paleta de colores fijos
        fig = px.scatter(df, x=x, y=y, color='labels',
                     title=f'Jugadores en cada clúster según<br>{x} y {y}',
                     labels={'labels': 'Clusters'},
                                color_continuous_scale='Cividis')
        
        # Change opacity
        fig.update_traces(opacity=0.8)

        # Update layour
        fig.update_layout(width=600, height = 400)
        return fig

        
    # --------- Create 3d plot -----------
    def plot_3d(df, x, y, clusterer):
        labels = clusterer.labels_ + 1
                
        # Labels predicted
        df['labels'] = labels

        z = st.selectbox('Selecciona variable Z', 
                           (df.columns),
                          index = 2)

        #Create a 3D scatter plot with colors separated by column value
        fig = go.Figure(data=[go.Scatter3d(
            x=df[x],
            y=df[y],
            z=df[z],
            mode='markers',
            marker=dict(
                size=3,
                color=df['labels'],  # Use the 'color' column for coloring
                colorscale='Cividis',
                opacity=0.8
            ),
            text=df['labels'],  # Use the 'label' column for hover text
            hoverinfo='text'  # Show hover text
        )])
    
        # Update layout for better visualization
        fig.update_layout(scene=dict(aspectmode="cube"))
        
        # Update layour
        fig.update_layout(width=600, height = 400)
        return fig
        
    #  ----- Make heatmap graph -----                       
    def graph_heatmp(df, x, y):
        # Count of values based on x and y values
        bubble_sizes = df.groupby([f'{x}', f'{y}']).size().reset_index(name='count')
        
        # Join values of bubble_sizes
        df_ = pd.merge(df, bubble_sizes, on=[f'{x}', f'{y}'], how='left')
        
        fig = px.density_heatmap(df_, x=f'{x}', y=f'{y}', z='count', title=f'Zonas con mayor aglomeración de jugadores',
                                color_continuous_scale='Cividis')

        # Change legend title name
        fig.update_coloraxes(colorbar_title='Cantidad de<br>jugadores')

        # Update layout
        fig.update_layout(width=400, height = 460)
        
        return fig

    #  ----- Make boxplot graph ----- 
    def graph_boxplot(df, dbscan):

        labels = dbscan.labels_ + 1
        
        # Labels predicted
        df['labels'] = labels
        
        # Make fig with plotly
        fig = px.box(df, x='labels', y=y,
                    title=f'Distribución de jugadores por cluster')

        # Update layout
        fig.update_layout(width=400, height = 600)
        return fig


    
    # ------ Quantity of players by cluster graph ------ 
    def graph_prop_by_cluster(df):
        # Crear un DataFrame con las etiquetas y contar la cantidad en cada grupo
        group_counts = labels.value_counts().reset_index()
        group_counts.columns = ['Group', 'Count']
        group_counts = group_counts.sort_values(by='Group')
        
        # Crear el gráfico de barras con Plotly
        fig = px.bar(group_counts, x='Group', y='Count', labels={'Group': 'Cluster', 'Count': 'Número de Individuos'},
                 title='Cantidad de Individuos en<br>cada Cluster')

        fig.update_layout(width=400, height = 300)
        # Mostrar el gráfico
        return fig, group_counts


    
    # --------- Make graph to determine minPts --------
    def graph_minPts(df, eps):
        min_pts_values = list(range(1, 20))
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
    
        fig = px.line(x=min_pts_values, y=silhouette_scores, markers=True, line_shape='linear',
                      labels={'x': 'MinPts', 'y': 'Silhouette Score'},
                      title='Método del Codo para DBSCAN')
        
        fig.update_layout(width=400, height = 300, showlegend=False)
        return fig

    
    # ----- K distance graph -----
    def graph_kdistance(df, total_clusters):

        # Calculate the distances to the k-nearest neighbors
        neigh = NearestNeighbors(n_neighbors=total_clusters)
        neigh.fit(df)
        distances, _ = neigh.kneighbors(df)
        
        # Ordena las distancias y crea el gráfico en Plotly
        sorted_distances = np.sort(distances[:, -1])
        
        fig = go.Figure()
    
        fig.add_trace(go.Scatter(
            x=list(range(1, len(sorted_distances) + 1)),
            y=sorted_distances,
            mode='markers+lines',
            #marker=dict(color='blue', size=8),
            name=f'{total_clusters}-th Nearest Neighbor Distance'
        ))
    
        fig.update_layout(
            title='K-Distance Graph',
            xaxis=dict(title='Data Point Index'),
            yaxis=dict(title=f'{total_clusters}-th Nearest Neighbor Distance'),
            showlegend=True,
            autosize=False,
            width=800,
            height=600,
        )

        fig.update_layout(width=400, height = 300, showlegend=False)

        return fig
    ###########################################
    #  ----- Write in sidebar in DBSCAN page ----- 
    with st.sidebar:
        st.divider()
        colored_header(
        label="Cambia los parámetros de DBSCAN",
        description="",
        color_name="red-90",
        )
        # Slider with eps value
        eps = st.number_input('Selecciona el valor de eps', value = 0.5)

        # Slider with min_samples vlaue
        min_samples = st.number_input('Selecciona la cantidad de datos', value = 5)

    
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

    # Page title
    colored_header(
        label="Density-Based Spatial Clustering of Applications with Noise (DBSCAN)",
        description="",
        color_name="light-blue-90",
        )
    st.markdown("""
DBSCAN es un algoritmo de clustering que se basa en la densidad de los datos para agrupar puntos cercanos en regiones densas y separar regiones menos densas. Este método estadístico es especialmente útil para identificar agrupamientos de forma no paramétrica y detectar puntos anómalos en conjuntos de datos.
    
### Principales Conceptos
- **Densidad:** DBSCAN define la densidad en términos de la cantidad de puntos dentro de una vecindad específica. Puntos con suficientes vecinos cercanos se consideran parte de una región densa.

- **Puntos Núcleo:** Puntos dentro de regiones densas que tienen al menos un número mínimo predefinido de vecinos dentro de una distancia especificada.

- **Ruido:** Puntos que no pertenecen a ninguna región densa y no cumplen con los criterios de densidad mínima.

### Ventajas y Consideraciones

- **Robustez ante Formas Irregulares:** DBSCAN es efectivo para identificar agrupamientos de formas no convencionales y puede manejar conjuntos de datos con densidades variables.

- **Sensibilidad a Parámetros:** La elección adecuada de varepsilon y MinPts es crucial y puede afectar significativamente los resultados del clustering.

- **Resistente al Ruido:** Al asignar puntos como ruido, DBSCAN es robusto ante valores atípicos y no depende de la forma geométrica de los agrupamientos.
""")
    st.divider()
    
    # Make first dashboard ---------
    colored_header(
        label="Descripción de cada grupo",
        description="Selecciona las zonas de interés acercando el ratón a los gráficos",
        color_name="light-blue-90",
        )
    

    # Add Silohuette and Davies Boundis metrics
    add_top_metrics(df)

    # Make title of filtering 
    st.markdown("#### ***Cambia los ejes de los gráficos***")
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
    fig_buble = graph_bubble(df, x, y, dbscan)
    fig_heatmap = graph_heatmp(df, x, y)
    fig_boxplot = graph_boxplot(df, dbscan)
    fig_3d = plot_3d(df, x, y, dbscan)

    
    # Create a 1x2 grid to write the graphs
    my_grid = grid(2, [3, 2], vertical_align="center")
    # Row 1:
    # Write 2d and 3d scatterplot
    tab1_, tab2_ = my_grid.tabs(['2D', '3D'])
    with tab1_:
        st.plotly_chart(fig_buble, theme="streamlit", use_container_width=True)
    with tab2_:
        st.plotly_chart(fig_3d, theme="streamlit", use_container_width=True)
    
    # Write heatmap
    my_grid.plotly_chart(fig_heatmap, theme="streamlit", use_container_width=True)
    
    # Row 2:
    my_grid.plotly_chart(fig_boxplot, theme="streamlit", use_container_width=True)
    # Row 1:
    with my_grid.container():
            st.plotly_chart(fig_bar, theme="streamlit", use_container_width=True)
            st.dataframe(df_bar, use_container_width = True, hide_index = True, height = 200)
    
    # Row 2:
    # Title
    st.divider()
    colored_header(
        label="Descripción de DBSCAN",
        description="Rendimiento de cada grupo",
        color_name="light-blue-90",
        )

    st.plotly_chart(fig_minPts, clear_figure  = True, use_container_width=True)
    st.plotly_chart(fig_kdistance, theme="streamlit", use_container_width=True)




# --------------------------------------------------------------------------
# Create DBSCAN Page
def HDBSCAN_page(df):
    
    # ------ Create sidebar metrics ------------
    def sidebar_metrics(df, clusterer):
    
        # Create labels of the clusters
        df['labels'] = clusterer.labels_
        df.groupby('labels').count().reset_index
        
        # Save quantity of noisy data
        noise_quantity = df.loc[df['labels'] == -1]
        noise_quantity = len(noise_quantity['labels'])
        
        # Save of total non noisy clusters
        total_clusters = len(df['labels'].unique()) - 1
        
         # Create two columns 
        col1, col2 = st.columns(2)
        
        # Noise column
        col1.metric("Datos de ruido", f'{noise_quantity}')
            
        # Total clusters column
        col2.metric("Total de clusters", f'{total_clusters}')
    
        # Style metric cards
        style_metric_cards(border_left_color="#0a0a0a", 
                           border_color="#0a0a0a", 
                           border_radius_px=25)

    
    # -------- Add metrics on top of the page ----------
    # Metrics added: Silohuette Score
    #                Davies-Bouldin Index
    def add_top_metrics(df,clusterer):
        # Get sillohuette score
        predicts = clusterer.labels_
    
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


    
    # ------ Create HDBSCAN plots -----------
    def hdbscan_plots(clusterer):
        
        # Make spanning_tree ------------- 
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        mst_plot = clusterer.minimum_spanning_tree_.plot(edge_cmap='cividis', 
                                                         edge_alpha=1,node_size=40,
                                                         edge_linewidth=1)
        ax1.set_title("Árbol de mínima expansión ", fontsize=15)  
        
        # Make condensed tree without cluster ---------------
        fig2, ax2 = plt.subplots(figsize=(10, 10))
        mst_plot = clusterer.condensed_tree_.plot(cmap='cividis',)
    
        # Make Condensed tree w cluster --------------
        fig3, ax3 = plt.subplots(figsize=(20, 8))
        mst_plot = clusterer.condensed_tree_.plot(select_clusters=True, cmap='cividis',
                                                  selection_palette=sns.color_palette())
        ax3.set_title("Árbol condensado de dendograma con enlace Single\n", fontsize=20) 
        
        # Make single linkage tree ------------
        fig4, ax4 = plt.subplots(figsize=(10, 10))
        mst_plot = clusterer.single_linkage_tree_.plot(cmap='cividis', colorbar=True)
        ax4.set_title("Dendrograma con enlace Single", fontsize=22)  
        # Return figs
        return fig1, fig2, fig3, fig4

    
    # ----------- Create scatter labeled by color -----------
    def scatter_clustered(df, x, y, clustered):

        df['labels'] = clusterer.labels_.astype('int') + 1

        # Crea el gráfico con Plotly Express y asigna la paleta de colores fijos
        fig = px.scatter(df, x=x, y=y, color='labels',
                     title=f'Jugadores en cada clúster según<br>{x} y {y}',
                     labels={'labels': 'Clusters'},
                                color_continuous_scale='Cividis')
        
        # Change opacity
        fig.update_traces(opacity=0.8)

        # Update layour
        fig.update_layout(width=600, height = 400)
        return fig

    # --------- Create 3d plot -----------
    def plot_3d(df, x, y, clusterer):
        labels = clusterer.labels_ + 1
                
        # Labels predicted
        df['labels'] = labels

        z = st.selectbox('Selecciona variable Z', 
                           (df.columns),
                          index = 2)

        #Create a 3D scatter plot with colors separated by column value
        fig = go.Figure(data=[go.Scatter3d(
            x=df[x],
            y=df[y],
            z=df[z],
            mode='markers',
            marker=dict(
                size=3,
                color=df['labels'],  # Use the 'color' column for coloring
                colorscale='Cividis',
                opacity=0.8
            ),
            text=df['labels'],  # Use the 'label' column for hover text
            hoverinfo='text'  # Show hover text
        )])
    
        # Update layout for better visualization
        fig.update_layout(scene=dict(aspectmode="cube"))
        
        # Update layour
        fig.update_layout(width=600, height = 400)
        return fig
        
    #  ----- Make heatmap graph -----                       
    def graph_heatmp(df, x, y):
        # Count of values based on x and y values
        bubble_sizes = df.groupby([f'{x}', f'{y}']).size().reset_index(name='count')
        
        # Join values of bubble_sizes
        df_ = pd.merge(df, bubble_sizes, on=[f'{x}', f'{y}'], how='left')
        
        fig = px.density_heatmap(df_, x=f'{x}', y=f'{y}', z='count', title=f'Zonas con mayor aglomeración de jugadores',
                                color_continuous_scale='Cividis')

        # Change legend title name
        fig.update_coloraxes(colorbar_title='Cantidad de<br>jugadores')

        # Update layout
        fig.update_layout(width=400, height = 460)
        
        return fig

    #  ----- Make boxplot graph ----- 
    def graph_boxplot(df, clusterer):

        labels = clusterer.labels_ + 1
        
        # Labels predicted
        df['labels'] = labels
        
        # Make fig with plotly
        fig = px.box(df, x='labels', y=y,
                    title=f'Distribución de jugadores por cluster')

        # Update layout
        fig.update_layout(width=400, height = 600)
        return fig


    # ------ Quantity of players by cluster graph ------ 
    def graph_prop_by_cluster(df, clusterer):

        df['labels'] = clusterer.labels_ + 1
        
        # Crear un DataFrame con las etiquetas y contar la cantidad en cada grupo
        group_counts = df['labels'].value_counts().reset_index()
        
        group_counts.columns = ['Group', 'Count']
        group_counts = group_counts.sort_values(by='Group')
        # Crear el gráfico de barras con Plotly
        fig = px.bar(group_counts, x='Group', y='Count', labels={'Group': 'Cluster', 'Count': 'Número de Individuos'},
                 title='Cantidad de Individuos en<br>cada Cluster')

        fig.update_layout(width=400, height = 300)
        # Mostrar el gráfico
        return fig, group_counts


    #################################
    
    # Write in sidebar 
    with st.sidebar:
        st.divider()
        colored_header(
        label="Cambia los parámetros de HDBSCAN",
        description="",
        color_name="red-90",
        )
        # Min of clusters to use in HDBSCAN
        min_cluster = st.number_input('Selecciona el mínimo de clusters', value = 5)

    # Apply HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster, metric='euclidean',
                                gen_min_span_tree=True)
    clusterer.fit(df)

    
    
    # Page title    
    colored_header(
        label="Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN)",
        description="",
        color_name="light-blue-90",
        )
    st.markdown("""HDBSCAN es una extensión de DBSCAN que incorpora la idea de la densidad en la construcción de una estructura jerárquica de clusters. Este algoritmo estadístico es particularmente útil para identificar clusters de forma variable y manejar conjuntos de datos con densidades variables de manera más flexible.

### Principales Conceptos

- **Densidad y Persistencia:** HDBSCAN introduce el concepto de persistencia, que mide la estabilidad de un cluster en múltiples escalas de densidad. Los clusters más densos y persistentes se consideran más significativos.

- **Árbol de Clustering:** HDBSCAN utiliza un árbol de clustering para representar la jerarquía de los clusters en diferentes niveles de densidad. Esto permite la identificación de clusters a diferentes escalas.

- **Afinidad Mutual:** HDBSCAN utiliza un enfoque de afinidad mutual para calcular la conectividad entre puntos, considerando tanto la densidad como la distancia entre ellos.

### Ventajas y Consideraciones

- **Jerarquía de Clusters:** HDBSCAN proporciona una visión jerárquica de la estructura de clusters, permitiendo la identificación de clusters a diferentes niveles de detalle.

- **Robustez ante Densidades Variables:** HDBSCAN es capaz de manejar conjuntos de datos con densidades variables y identificar clusters de forma adaptativa.

- **Menos Sensible a Parámetros:** HDBSCAN reduce la sensibilidad a la selección de parámetros en comparación con DBSCAN, ya que la jerarquía proporciona una variedad de clusters a diferentes niveles de resolución.

- **Afinidad Mutual:** La utilización de afinidad mutual mejora la capacidad de HDBSCAN para identificar clusters en espacios de alta dimensionalidad.

""")
    st.divider()
    
    # Make first dashboard ---------
    colored_header(
        label="Descripción de cada grupo",
        description="Selecciona las zonas de interés acercando el ratón a los gráficos",
        color_name="light-blue-90",
        )
    add_top_metrics(df,clusterer)
    # Make title of filtering 
    st.markdown("#### ***Cambia los ejes de los gráficos***")
    
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

    # Write noisy daya and total quantity metrics in sidebar
    with st.sidebar:
        sidebar_metrics(df, clusterer)
        
    # Make plots
    spanning, condensed_1, condensed_2, sl_tree = hdbscan_plots(clusterer)
    scatter = scatter_clustered(df, x, y, clusterer)
    heatmap = graph_heatmp(df, x, y)
    boxplot = graph_boxplot(df, clusterer)
    plot_3d = plot_3d(df, x, y, clusterer)
    barplot, porp_table = graph_prop_by_cluster(df, clusterer)



    # Create a 2x3 grid to write the graphs
    my_grid = grid(2, [3,2], vertical_align="bottom")

    # First row:
    tab1, tab2 = my_grid.tabs(['2D', '3D'])
    with tab1:
        st.plotly_chart(scatter, theme="streamlit", use_container_width=True)
    with tab2:
        st.plotly_chart(plot_3d, theme="streamlit", use_container_width=True)
        
    my_grid.plotly_chart(heatmap, theme="streamlit", use_container_width=True)

    # Second row:
    my_grid.plotly_chart(boxplot, theme="streamlit", use_container_width=True)
    with my_grid.container():
        st.plotly_chart(barplot, theme="streamlit", use_container_width=True)
        st.dataframe(porp_table, use_container_width = True, hide_index = True, height = 200)
    # Make second dashboard -----------
    st.divider()

    # Title
    colored_header(
        label="Descripción de HDBSCAN",
        description="Distancias y jerarquía de cada grupo",
        color_name="light-blue-90",
        )

    # Create a 2x1 grid to write the graphs
    my_grid = grid([3,2], 1, vertical_align = "bottom")

    # First row
    my_grid.pyplot(spanning, clear_figure  = True, use_container_width=True)
    my_grid.pyplot(sl_tree, clear_figure  = True, use_container_width=True)

    # Second row
    col1, col2 = st.columns(2)
    my_grid.pyplot(condensed_2, clear_figure  = True, use_container_width=True)


# --------------------------------------------------------------------------
# Hallazgos page: fixed (non-interactive) findings -- optimal parameters for
# DBSCAN/HDBSCAN found offline via grid search, K-Means as a baseline
# comparison, a PCA 2D projection, and 4 named player archetypes.
#
# Uses its own fixed feature set (independent of the sidebar filters used by
# the DBSCAN/HDBSCAN pages) since this page reports a single settled result,
# not another interactive exploration.
FEATURES_HALLAZGOS = ['pts_per_game', 'trb_per_game', 'ast_per_game',
                       'stl_per_game', 'blk_per_game', 'tov_per_game']

ARCHETYPES = {
    3: ("Rotación corta / bajo uso", "#7f8c8d"),
    1: ("Rotación perimetral", "#3498db"),
    0: ("Interiores / reboteadores", "#e67e22"),
    2: ("Anotadores y creadores de juego", "#e74c3c"),
}


def hallazgos_page(df_full):

    colored_header(
        label="Hallazgos",
        description="DBSCAN vs. HDBSCAN vs. K-Means: qué método encuentra arquetipos reales",
        color_name="light-blue-90",
        )

    st.markdown(f"""
Esta página fija los parámetros y no tiene controles interactivos -- a diferencia de las páginas
DBSCAN y HDBSCAN, que invitan a explorar, esta resume una conclusión ya buscada y verificada.

**Metodología:** se usaron 6 estadísticas por partido -- {', '.join(FEATURES_HALLAZGOS)} -- estandarizadas
(`StandardScaler`), sobre el dataset completo de {len(df_full):,} temporadas de jugador (2013 a 2024-25).
Para DBSCAN se probó una grilla de `eps` y `min_samples`; para HDBSCAN, un rango de `min_cluster_size`
entre 5 y 150; para K-Means, se fijó `k` igual a la cantidad de clusters de la mejor solución de DBSCAN,
para poder comparar métricas directamente.
""")

    scaler = StandardScaler()
    X = df_full[FEATURES_HALLAZGOS].copy()
    Xs = scaler.fit_transform(X)

    DBSCAN_EPS, DBSCAN_MIN_SAMPLES = 1.20, 6
    db_labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(Xs)
    db_n = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    db_noise = (db_labels == -1).mean() * 100
    db_sil = silhouette_score(Xs, db_labels)
    db_dbi = davies_bouldin_score(Xs, db_labels)

    HDB_MIN_CLUSTER_SIZE = 20
    hdb_clusterer = hdbscan.HDBSCAN(min_cluster_size=HDB_MIN_CLUSTER_SIZE, metric='euclidean',
                                     gen_min_span_tree=True)
    hdb_labels = hdb_clusterer.fit_predict(Xs)
    hdb_n = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
    hdb_noise = (hdb_labels == -1).mean() * 100
    hdb_sil = silhouette_score(Xs, hdb_labels) if hdb_n >= 2 else float('nan')
    hdb_dbi = davies_bouldin_score(Xs, hdb_labels) if hdb_n >= 2 else float('nan')

    km = KMeans(n_clusters=db_n, init='k-means++', random_state=42, n_init=10)
    km_labels = km.fit_predict(Xs)
    km_sil = silhouette_score(Xs, km_labels)
    km_dbi = davies_bouldin_score(Xs, km_labels)

    st.markdown("#### Comparación de los tres métodos")
    comparison = pd.DataFrame([
        {"Método": "DBSCAN", "Parámetros": f"eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES}",
         "Clusters": db_n, "Ruido": f"{db_noise:.1f}%", "Silhouette": round(db_sil, 3),
         "Davies-Bouldin": round(db_dbi, 3)},
        {"Método": "HDBSCAN", "Parámetros": f"min_cluster_size={HDB_MIN_CLUSTER_SIZE}",
         "Clusters": hdb_n, "Ruido": f"{hdb_noise:.1f}%", "Silhouette": round(hdb_sil, 3),
         "Davies-Bouldin": round(hdb_dbi, 3)},
        {"Método": "K-Means (baseline)", "Parámetros": f"k={db_n}",
         "Clusters": db_n, "Ruido": "0.0%", "Silhouette": round(km_sil, 3),
         "Davies-Bouldin": round(km_dbi, 3)},
    ])
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.markdown(f"""
**El silhouette de DBSCAN (0.454) se ve bien en la tabla, pero esconde un resultado degenerado:**
del {db_n}-cluster que encuentra a `eps={DBSCAN_EPS}`, un {(pd.Series(db_labels).value_counts(normalize=True).iloc[0]*100):.1f}%
de las temporadas caen en un solo cluster gigante ("todo el mundo"), y los otros {db_n - 1} clusters
son grupos diminutos (4 a 18 filas) de temporadas estadísticamente extremas -- básicamente,
DBSCAN aísla a un puñado de temporadas de superestrella como sus propios "clusters" en vez de
encontrar arquetipos de jugador reales.

**HDBSCAN es aún más extremo:** en ningún `min_cluster_size` probado (de 5 a 150) se encontró una
solución con silhouette positivo y ruido razonable -- con valores chicos, decenas de micro-clusters
y 85-95% de ruido; con valores más grandes, prácticamente el 100% de las temporadas quedan
marcadas como ruido.

**La razón de fondo:** las estadísticas de jugadores NBA no forman regiones de alta densidad
separadas por vacíos -- son un espectro continuo de habilidad, desde jugadores de banca hasta
superestrellas, sin límites naturales nítidos. Eso es exactamente lo que rompe a los métodos
basados en densidad, y exactamente lo que **K-Means sí maneja bien** (no necesita vacíos de
densidad entre grupos, solo centroides razonablemente separados). Por eso los 4 arquetipos de
abajo salen de K-Means, no de DBSCAN ni HDBSCAN, pese a que el proyecto originalmente comparaba
solo métodos de densidad.
""")

    st.divider()

    # ---- PCA 2D projection, colored by K-Means archetype ----
    st.markdown("#### Proyección 2D (PCA) coloreada por arquetipo")
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Xs)
    var_explained = pca.explained_variance_ratio_

    df_plot = pd.DataFrame({
        'PC1': coords[:, 0],
        'PC2': coords[:, 1],
        'Arquetipo': [ARCHETYPES[c][0] for c in km_labels],
    })
    color_map = {name: color for name, color in ARCHETYPES.values()}
    fig_pca = px.scatter(
        df_plot, x='PC1', y='PC2', color='Arquetipo',
        color_discrete_map=color_map, opacity=0.5,
        title=f"PCA de las 6 estadísticas (PC1+PC2 explican {var_explained.sum():.0%} de la varianza)",
    )
    fig_pca.update_layout(height=550)
    st.plotly_chart(fig_pca, use_container_width=True)

    st.divider()

    # ---- Named archetypes with their statistical profile ----
    st.markdown("#### Los 4 arquetipos (de K-Means)")
    df_prof = X.copy()
    df_prof['cluster'] = km_labels
    profile = df_prof.groupby('cluster')[FEATURES_HALLAZGOS].mean().round(2)
    counts = df_prof['cluster'].value_counts()

    for cluster_id, (name, color) in ARCHETYPES.items():
        row = profile.loc[cluster_id]
        pct = counts[cluster_id] / len(df_prof) * 100
        st.markdown(f"##### :{('gray' if color=='#7f8c8d' else 'blue' if color=='#3498db' else 'orange' if color=='#e67e22' else 'red')}[{name}] -- {counts[cluster_id]:,} temporadas ({pct:.1f}%)")
        cols = st.columns(6)
        # Not using style_metric_cards() here on purpose: it forces a white card
        # background without adapting the label color, so labels disappear in dark
        # theme. Plain st.metric already adapts colors to whatever theme is active.
        labels_short = ['PTS', 'REB', 'AST', 'ROB', 'TAP', 'PER']
        for col, label, val in zip(cols, labels_short, row.values):
            col.metric(label, f"{val:.1f}")
        st.write("")

    st.divider()
    st.markdown("""
*Nota: el dataset se extendió para incluir la temporada 2024-25 (antes llegaba solo hasta
2023-24), agregada desde Basketball-Reference.*
""")
