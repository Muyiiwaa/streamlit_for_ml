import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# load in the dataset
data = pd.read_csv('insurance.csv')

# scale and preprocess the dataset
cat_cols = ['region','smoker','sex']
num_cols = ['age','bmi','children']

cat_cols = pd.get_dummies(data[cat_cols], dtype = float) # one-hot encode the cat variables
X = pd.concat([data[num_cols], cat_cols], axis = 1) # concatenate all columns

y = data['charges'] # create y series

scaler = StandardScaler()
columns = X.columns
X = scaler.fit_transform(X)
X = pd.DataFrame(data = X, columns = columns)

print(X.columns)

'''
# begin the model training and prediction
X_train, X_test, y_train, y_test = train_test_split(X,y, 
                                                    test_size = 0.2,random_state = 23)
model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# model evaluation
#rmse = mean_squared_error(y_test, y_pred, squared=False)

# pickle the model

with open('rfr_model.pkl', 'wb') as file:
    pickle.dump(model, file)
'''