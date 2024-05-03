from matplotlib import widgets
import streamlit as st

import k_nearest_neighbors
import collaborative_filtering 

# Setze einen Titel für die App
st.title("Maschinelle Lernalgorithmen")

# Zeige eine Information mit zusätzlichen Hinweisen zur Bedienung der App
st.info("""
Willkommen bei der Maschinellen Lern-Demo-App! Hier kannst du verschiede Maschinelle Lernalgorithmen erkunden.
""")

# Definiere die Seiten der App
pages = {
    "1. K-Nearest-Neighbor"   : k_nearest_neighbors,
    "2. Kollaboratives Filtern"     : collaborative_filtering,
}

# Erstelle eine Seitenleiste für die Navigation im Projekt
st.sidebar.title("Navigation")
select = st.sidebar.radio("Gehe zu:", list(pages.keys()))

# Starte die ausgewählte Seite
pages[select].app()

