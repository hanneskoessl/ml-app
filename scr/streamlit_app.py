from matplotlib import widgets
import streamlit as st

import k_nearest_neighbors
import collaborative_filtering 

# Setze einen Titel für die App
st.title("Machine learning algorithms")

# Zeige eine Information mit zusätzlichen Hinweisen zur Bedienung der App
st.info("""
Welcome to the Machine Learning demo app! Here you can explore different machine learning algorithms.
""")

# Definiere die Seiten der App
pages = {
    "1. K-Nearest-Neighbor"         : k_nearest_neighbors,
    "2. Collaborative filtering"     : collaborative_filtering,
}

# Erstelle eine Seitenleiste für die Navigation im Projekt
st.sidebar.title("Navigation")
select = st.sidebar.radio("Go to:", list(pages.keys()))

# Starte die ausgewählte Seite
pages[select].app()

