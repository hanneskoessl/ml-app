import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def app():
    # Setting a header for the page
    st.header("K-nearest neighbor algorithm")

    st.markdown("""
    This interactive application demonstrates how the K-Nearest Neighbors Algorithm (KNN) works. Here you can:
    
    - **Visualize Data Points**: See how different data points are distributed on a plane.
    - **Make Predictions**: Choose values for new points and determine their classes based on nearest neighbors.
    - **Adjust Parameters**: Change the number of neighbors and see how that affects the classification.
    
    Use the sliders below to adjust the parameters and observe how the model responds to your inputs.
    """)

    # Manually created data
    data_points = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    labels = np.array([0, 1, 1, 0, 1, 0])  # Example labels for two classes
    
    # Show table
    df = pd.DataFrame(data_points, columns=['X', 'Y'])
    df['Label'] = labels
    st.write("Data points:", df)

    # Definition of KNN function
    def knn(data, test_pt, k, labels):
        neighbor_distances_and_indices = []
        for index, example in enumerate(data):
            distance = np.linalg.norm(example - test_pt) 
            neighbor_distances_and_indices.append((distance, index))
        sorted_neighbors = sorted(neighbor_distances_and_indices)
        class_votes = {}
        for i in range(k):
            label = labels[sorted_neighbors[i][1]]
            class_votes[label] = class_votes.get(label, 0) + 1
        sorted_votes = sorted(class_votes.items(), key=lambda item: item[1], reverse=True)
        return (sorted_votes[0][0], sorted_neighbors)
    
    # User input for classification
    test_x = st.slider('Test X', min_value=0.0, max_value=10.0, value=5.0, step=0.01)
    test_y = st.slider('Test Y', min_value=0.0, max_value=10.0, value=5.0, step=0.01)
    test_point = np.array([test_x, test_y])
    k = st.slider('Number of neighbors (k)', 1, len(data_points), 3)

    # Classification
    predicted_class, sorted_neighbors = knn(data_points, test_point, k, labels)
    st.write(f'The predicted class for point {test_point} is: {predicted_class}')
    
    # Visualization of data points
    colors = {0: 'darkorange', 1: 'cornflowerblue', 2: 'lightcoral'}
    fig, ax = plt.subplots()
    # Plot data points
    for label, df_group in df.groupby('Label'):
        ax.scatter(df_group['X'], df_group['Y'], s=80, color=colors[label], label=f'Klasse {label}', alpha=0.6)  
    ax.scatter(test_point[0], test_point[1], s=120, color=colors[predicted_class])
    # Plot lines to the nearest neighbors
    for n, neighbors in enumerate(sorted_neighbors):
        x_coords = [test_point[0], data_points[neighbors[1]][0]]
        y_coords = [test_point[1], data_points[neighbors[1]][1]] 
        ax.plot(x_coords, y_coords, color='red', alpha=0.2)
        if n == k-1:
            break   
    ax.legend()
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    st.pyplot(fig)

# Executes the app function when this script is called directly
if __name__ == "__main__":
    app()