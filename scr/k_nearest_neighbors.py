import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def app():
    # Setzen eines Headers für die Seite
    st.header("K-Nearest-Neighbor-Algorithmus")

    st.markdown("""
    Diese interaktive Anwendung demonstriert die Funktionsweise des K-Nearest-Neighbors Algorithmus (KNN). Hier kannst du:
    
    - **Datenpunkte visualisieren**: Sehe, wie verschiedene Datenpunkte auf einer Ebene verteilt sind.
    - **Vorhersagen treffen**: Wähle Werte für neue Punkte und bestimme deren Klassen basierend auf den nächstgelegenen Nachbarn.
    - **Parameter einstellen**: Ändere die Anzahl der Nachbarn und beobachte, wie sich das auf die Klassifizierung auswirkt.
    
    Nutze die Sliders unten, um die Parameter zu verändern und beobachte, wie das Modell auf deine Eingaben reagiert.
    """)

    # Manuell erstellte Daten
    data_points = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    labels = np.array([0, 1, 1, 0, 1, 0])  # Beispiellabels für zwei Klassen
    
    # Tabelle anzeigen
    df = pd.DataFrame(data_points, columns=['X', 'Y'])
    df['Label'] = labels
    st.write("Datenpunkte:", df)

    # Definition der KNN-Funktion
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
    
    # Benutzereingaben für die Klassifizierung
    test_x = st.slider('Test X', min_value=0.0, max_value=10.0, value=5.0, step=0.01)
    test_y = st.slider('Test Y', min_value=0.0, max_value=10.0, value=5.0, step=0.01)
    test_point = np.array([test_x, test_y])
    k = st.slider('Anzahl der Nachbarn (k)', 1, len(data_points), 3)

    # Klassifikation
    predicted_class, sorted_neighbors = knn(data_points, test_point, k, labels)
    st.write(f'Der vorhergesagte Klasse für Punkt {test_point} ist: {predicted_class}')
    
    # Visualisierung der Datenpunkte
    colors = {0: 'darkorange', 1: 'cornflowerblue', 2: 'lightcoral'}
    fig, ax = plt.subplots()
    # plot Datenpunkte
    for label, df_group in df.groupby('Label'):
        ax.scatter(df_group['X'], df_group['Y'], s=80, color=colors[label], label=f'Klasse {label}', alpha=0.6)  
    ax.scatter(test_point[0], test_point[1], s=120, color=colors[predicted_class])
    # plot Linien zu den nächten Nachbarn
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

# Führt die App-Funktion aus, wenn dieses Skript direkt aufgerufen wird
if __name__ == "__main__":
    app()