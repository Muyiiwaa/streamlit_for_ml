import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle

model_cols = ['age', 'bmi', 'children', 'region_northeast', 'region_northwest', 
       'region_southeast', 'region_southwest', 'smoker_no', 'smoker_yes',
       'sex_female', 'sex_male']

with open('rfr_model.pkl', 'rb') as file:
    model = pickle.load(file)

# create user interface
st.title('Appclick Insurance App')

column1, column2 = st.columns(2)
with column1:
    age = st.text_input(label='age')
    children = st.text_input(label='children')
    bmi = st.text_input(label='bmi')

with column2:
    smoker = st.text_input(label='smoker')
    region = st.text_input(label='region')
    sex = st.text_input(label='sex')
    

try:
    num_list = [age,children,bmi]
    num_list = [float(x) for x in num_list]   
    cat_cols = [['yes','southwest','male'],
                ['no','southeast','female'],
                ['yes','northwest','male'],
                ['no','northeast','female'],
                [smoker,region,sex]]
    
    num_cols = [[1,1,1],
                [1,1,1],
                [1,1,1],
                [1,1,1],
                num_list]
    cat_cols = pd.DataFrame(cat_cols, columns=['smoker','region','sex'])
    cat_cols = pd.get_dummies(cat_cols, dtype = float)
    num_cols = pd.DataFrame(num_cols, columns=['age','children','bmi'])
    X = pd.concat([num_cols, cat_cols], axis = 1)
    scaler = StandardScaler()
    cols = X.columns
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, columns=cols)

except ValueError:
    st.write('Enter values first')

if st.button(label='predict', key='predict', type='primary'):
    pred_df = X.drop([0,1,2,3], axis=0)
    pred_df = pred_df[model_cols]
    prediction = model.predict(pred_df)
    st.success(body= f'Your insurance charge is predicted to be ₦{round(prediction[0], 2)}')
