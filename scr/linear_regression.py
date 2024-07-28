import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def app(): 
    # Setting a header for the page
    st.header("Linear Regression")

    st.markdown("""
    This interactive application demonstrates how Linear Regression works. Here you can:
    
    - **Visualize Data Points**: See how different data points are distributed on a plane.
    - **Adjust Parameters**: Change the intercept and slope coefficients and see how that affects the regression line.
    - **Find Best Parameters**: Run the app to automatically calculate the best intercept and slope that fit the data.
    
    Use the sliders below to adjust the parameters and observe how the model responds to your inputs.
    """)

    # Manually created data
    x = np.array([0.6, 2, 4 , 6, 8.6, 9.4 ])
    y = np.array([2.1, 3.2, 4.5, 6.6, 7.3, 9.1])
    
    # Show table
    data = np.hstack([x[:,np.newaxis], y[:,np.newaxis]])
    df = pd.DataFrame(data, columns=['X', 'Y'])
    st.write("Data points:", df)
    
    # User input for classification
    y_mean = y.mean()
    intercept = st.slider('Intercept', min_value=0.0, max_value=10.0, value=y_mean, step=0.01)
    slope = st.slider('Slope coefficient', min_value=-1.5, max_value=1.5, value=0.0, step=0.01)
    
    
    button_run = st.button('Run')
    if button_run:
        ones = np.ones((x.shape[0], 1))
        X = np.hstack((ones, x[:,np.newaxis]))
        beta_hat = np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.dot(np.transpose(X), y))
        intercept = beta_hat[0]
        slope = beta_hat[1]

    y_hat = intercept + slope * x
    
    # Visualization of data points
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=80)
    ax.scatter(x, y_hat, s=20, c="tab:red")
    #ax.legend()
    ax.plot(x, y_hat, c="tab:red")
    
    # Plot lines
    for n in range(x.shape[0]):
        x_coords = [x[n],x[n]]
        y_coords = [y[n], y_hat[n]]
        ax.plot(x_coords, y_coords, color='tab:red', alpha=0.2)
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    st.pyplot(fig)

    # MSE
    diff = y - y_hat
    squared_diff = np.square(diff)
    mse = np.mean(squared_diff)
    
    # R2-Score
    SS_res = np.sum(np.square(y - y_hat))
    SS_tot = np.sum(np.square(y - np.mean(y)))
    R2 = 1 - SS_res / SS_tot

    st.write(f"Intercept: {intercept:0.4}")
    st.write(f"Slope coefficient: {slope:0.4}")
    st.write(f"Mean Squared Error (MSE): {mse:0.4}")
    st.write(f"$R^2$-Score: {R2:0.4}") 

# Executes the app function when this script is called directly
if __name__ == "__main__":
    app()