from matplotlib import widgets
import streamlit as st

import k_nearest_neighbors
import collaborative_filtering 

# Set a title for the app
st.title("Machine learning algorithms")

# Display information with additional hints on how to use the app
st.info("""
Welcome to the Machine Learning demo app! Here you can explore different machine learning algorithms.
""")

# Define the pages of the app.
pages = {
    "1. K-Nearest-Neighbor"         : k_nearest_neighbors,
    "2. Collaborative filtering"     : collaborative_filtering,
}

# Create a sidebar for navigation within the project.
st.sidebar.title("Navigation")
select = st.sidebar.radio("Go to:", list(pages.keys()))

# Launch the selected page
pages[select].app()

