# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 21:53:34 2022

@author: USER
"""

import pandas as pd
from sklearn.linear_model import Lasso
import math


emission = pd.read_csv("C:/Users/kelvi/OneDrive/Desktop/Cleaned_Data.csv")

#define predictor and response variables
X = emission[["Jet Fuel_avi_BTU", "Gasoline_avi_BTU", "LDV_SWB_road_BTU", "LDV_LWB_road_BTU","Combination_Truck_road_BTU","Bus_Road_BTU","Railways_BTU","Water_BTU","Natural_Gas_BTU","LDV_SWB_EFF","LDV_LWB_EFF","Passenger_Car_EFF","Domestic_EFF","Imported_EFF","Light_Truck_EFF","Passenger_Car_Age","Light_Truck_Age","Light_vehicle_Age","Demand_petroleum_transportation)mil_lit","Average_MC/15000_miles(dollars)"]]
y = emission["CO2_emission_million_metric_tons"]
X_mean= X.mean()
X_stddev= X.std()

list_numerical=X.columns
list_numerical

from sklearn.model_selection import train_test_split
#Split the data set into train and test set with the first 70% of the data for training and the remaining 30% for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.preprocessing import StandardScaler
#standardization of numerical features
scaler = StandardScaler().fit(X_train[list_numerical]) 
X_train[list_numerical] = scaler.transform(X_train[list_numerical])
X_test[list_numerical] = scaler.transform(X_test[list_numerical])


#Apply lasso regression with arbitrarily alpha value of 1
reg = Lasso(alpha=1)
reg.fit(X_train, y_train)

#print R squared value for training and test set
print('R squared training set', round(reg.score(X_train, y_train)*100, 2))
print('R squared test set', round(reg.score(X_test, y_test)*100, 2))

#print MSE for training and test set
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
# Training data
pred_train = reg.predict(X_train)
mse_train = mean_squared_error(y_train, pred_train)
rsme_train=math.sqrt(mse_train)
mae_train=mean_absolute_error(y_train,pred_train)
print('RMSE of training set', round(rsme_train, 2))
print('MAE of training set', round(mae_train, 2))
# Test data
pred = reg.predict(X_test)
mse_test =mean_squared_error(y_test, pred)
rmse_test=math.sqrt(mse_test)
mae_test =mean_absolute_error(y_test, pred)
print('RMSE of test set', round(rmse_test, 2))
print('MAE of test set', round(mae_test, 2))

#Model Testing

def input_data (a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t):
    data = [[(a-X_mean[0])/X_stddev[0],(b-X_mean[1])/X_stddev[1],(c-X_mean[2])/X_stddev[2],(d-X_mean[3])/X_stddev[3],(e-X_mean[4])/X_stddev[4],(f-X_mean[5])/X_stddev[5],(g-X_mean[6])/X_stddev[6],(h-X_mean[7])/X_stddev[7],(i-X_mean[8])/X_stddev[8],(j-X_mean[9])/X_stddev[9],(k-X_mean[10])/X_stddev[10],(l-X_mean[11])/X_stddev[11],(m-X_mean[12])/X_stddev[12],(n-X_mean[13])/X_stddev[13],(o-X_mean[14])/X_stddev[14],(p-X_mean[15])/X_stddev[15],(q-X_mean[16])/X_stddev[16],(r-X_mean[17])/X_stddev[17],(s-X_mean[18])/X_stddev[18],(t-X_mean[19])/X_stddev[19]]]
    return data

s_data= input_data(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)
df = pd.DataFrame(s_data,columns=["Jet Fuel_avi_BTU", "Gasoline_avi_BTU", "LDV_SWB_road_BTU", "LDV_LWB_road_BTU","Combination_Truck_road_BTU","Bus_Road_BTU","Railways_BTU","Water_BTU","Natural_Gas_BTU","LDV_SWB_EFF","LDV_LWB_EFF","Passenger_Car_EFF","Domestic_EFF","Imported_EFF","Light_Truck_EFF","Passenger_Car_Age","Light_Truck_Age","Light_vehicle_Age","Demand_petroleum_transportation)mil_lit","Average_MC/15000_miles(dollars)"],dtype=float)

reg.predict(df)

#For model deployment
import pickle

with open('Lasso Regression Prediction.pkl','wb') as file:
    pickle.dump(reg, file)
