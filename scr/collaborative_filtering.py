import streamlit as st
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import time


nr_para = 2

def write_table(params, n_users, n_items):
    users = [f'User {i+1}' for i in range(n_users)]
    items = [f'Item {i+1}' for i in range(n_items)]
    #st.write(fit(params))
    para1_opt = params[:nr_para*n_items].reshape(nr_para, n_items)
    para2_opt = params[nr_para*n_items:].reshape(n_users, nr_para)
    
    df_para1_opt = pd.DataFrame(para1_opt, index=np.arange(nr_para), columns=items)
    df_para2_opt = pd.DataFrame(para2_opt, index=users, columns=np.arange(nr_para))
    
    col1, col2 = st.columns([1,3])
    with col2:
        st.write(df_para1_opt)
    
    col1, col2 = st.columns([1,3])
    with col1:
        st.write(df_para2_opt)
    with col2:
        st.write(df_para2_opt.dot(df_para1_opt))


def app():
    if 'seed' not in st.session_state:
        st.session_state.seed = 0

    st.header("Matrix Factorization Model for Collaborative Filtering")

   
    data = [
        [3, 0, 1, 0, 1],
        [1, 0, 4, 1, 0],
        [0, 3, 0, 4, 4],
        [4, 2, 2, 3, 0]
    ]
    

    # Create numpy array
    ratings = np.array(data)
    n_users, n_items = ratings.shape
    users = [f'User {i+1}' for i in range(n_users)]
    items = [f'Item {i+1}' for i in range(n_items)]

    

    df_ratings = pd.DataFrame(ratings, index=users, columns=items)
    
    df_ratings=df_ratings.where(df_ratings!=0, None) 
    col1, col2 = st.columns([1,3])
    with col2:
        st.write("Rating Data", df_ratings)

    # Create a two-column layout
    col1, col2 = st.columns(2)
    with col1:
        button_reset = st.button('Reset')
        if button_reset:
            st.session_state.seed = int(time.time())
    with col2:
        button_run = st.button('Run')

    np.random.seed(st.session_state.seed)
    parameter1 = np.full((nr_para, n_items), 0.8)
    parameter1  = np.random.uniform(0, 1, size=(nr_para, n_items))
    
    parameter2 = np.full((n_users, nr_para), 0.9)
    parameter2  = np.random.uniform(0, 1, size=(n_users, nr_para))
      

    def objective_function(x):
        para1 = x[0:nr_para*n_items].reshape(nr_para, n_items)  
        para2 = x[nr_para*n_items:].reshape(n_users, nr_para)
        fit = para2.dot(para1)
        diff = ratings - fit
        diff_clean = np.where(ratings == 0, 0, diff)
        squared_diff = np.square(diff_clean)
        sum_squared_diff = np.sum(squared_diff)
        count = np.size(ratings)
        return np.sqrt(sum_squared_diff / count)
    
    
    initial_guess = np.concatenate([parameter1.flatten(), parameter2.flatten()])

    if button_run:
        bounds_para1 = [(0, 5.5)] * nr_para * n_items
        bounds_para2 = [(0, 5.5)] * n_users * nr_para
        bounds = bounds_para1 + bounds_para2
        result = minimize(objective_function, initial_guess, bounds=bounds)
        #result = minimize(run_fit, initial_guess, method='L-BFGS-B', bounds=bounds, 
        #            options={'disp': True, 'maxiter': 500, 'gtol': 1e-6})

        

        if result.success:
            optimized_params = result.x
            
            write_table(optimized_params, n_users, n_items)
            st.write(f"Target value: {objective_function(optimized_params)}")

        else:
            st.write("Optimization failed:", result.message)

        
    else:
        write_table(initial_guess, n_users, n_items)
        st.write(f"Target value: {objective_function(initial_guess)}")
   

# Execute the app function when this script is called directly
if __name__ == "__main__":
    app()
