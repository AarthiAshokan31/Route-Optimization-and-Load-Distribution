#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 12:40:51 2021

@author: hariprasadrajendran
"""

import numpy as np
import pandas as pd
import datetime

#Importing the train dataset
train = pd.read_csv("/Users/hariprasadrajendran/Documents/Rutgers/AI in decision making/Project_RouteOptimization/train.csv")
train.head()
train.shape
a = train['vendor_id'].unique()

#Checking for the null values
is_NaN=train.isnull()
row_has_NaN=is_NaN.any(axis=1)
rows_withN = train[row_has_NaN]
print(rows_withN)

##printing the vendor ids
print(train['vendor_id'].value_counts())
print(train['passenger_count'].value_counts())
###dropping vendor id =2 & resetting the index
index_id=train[train['vendor_id']==2].index
train.drop(index_id, inplace=True)
train.shape
train=train.reset_index(drop=True)


##Feature Engineering

##changing trip duration to minutes
train['trip_duration'] = round((train['trip_duration'])/60)

##Converting Pickup & dropOff Datetime string to datetime
train["pickup_datetime"]=pd.to_datetime(train["pickup_datetime"],format='%Y-%m-%d %H:%M:%S')
train["dropoff_datetime"]=pd.to_datetime(train["dropoff_datetime"],format='%Y-%m-%d %H:%M:%S')

##Adding other columns
train['Year']=train['pickup_datetime'].dt.year
train['pickup_month']=train['pickup_datetime'].dt.month
train['pickup_date']=train['pickup_datetime'].dt.day
train['pickup_hour']=train['pickup_datetime'].dt.hour
train['pickup_minutes']=train['pickup_datetime'].dt.minute

print(type(train['pickup_month']))
print(max(train['pickup_date']))
##Calculating Haversine distance
#from sklearn.metrics.pairwise import haversine_distances
from math import radians

def hav_distance(train):
    lat_p = np.radians(train['pickup_latitude'])
    lat_d = np.radians(train['dropoff_latitude'])
    lat_diff = np.radians(train['dropoff_latitude']-train['pickup_latitude'])
    long_diff = np.radians(train['dropoff_longitude']-train['pickup_longitude'])
    p = np.sin(lat_diff/2)**2 + np.cos(lat_p) * np.cos(lat_d) * np.sin(long_diff/2)**2
    q = 2 * np.arctan2(np.sqrt(p), np.sqrt(1-p))
    r = q * 3956
    return r

train['Trip_Distance'] = hav_distance(train)
# 
import matplotlib.pyplot as plt

#plt.plot(train['dropoff_longitude'],train['dropoff_latitude'])

#Filtering out off boudary points - Boundary of NYC is (-75, -73, 40, 42)

def Filtering(df):
    boundary_nyc = (df.pickup_longitude >= -75) & (df.pickup_longitude <= -73) & \
                      (df.pickup_latitude >= 40) & (df.pickup_latitude <= 42) & \
                      (df.dropoff_longitude >= -75) & (df.dropoff_longitude <= -73) & \
                      (df.dropoff_latitude >= 40) & (df.dropoff_latitude <= 42)
    df =df[boundary_nyc]
    return df

print("Old size {}".format(len(train)))

train = Filtering(train)

print("New size {}".format(len(train)))
print(train.head())

###Plotting trips

longi = list(train.pickup_longitude)+ list(train.dropoff_longitude)
lati = list(train.pickup_latitude)+ list(train.dropoff_latitude)

plt.figure(figsize = (10,8))
plt.plot(longi,lati,'.',markersize=1)
plt.title("NYC Yellow Cab Trips")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

##Correlations between other variables and trip duration
print("Coorelation between Trip duration and other variables")
print(train.corr('pearson')['trip_duration'])

##Modeling#####
from sklearn import datasets, linear_model
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from sklearn.metrics import mean_squared_error, r2_score

#from sklearn.cross_validation import cross_val_score, cross_val_predict
#from sklearn import metrics

##dropping off some variables and splitting data 

from sklearn.model_selection import train_test_split

train, test = train_test_split(train, test_size=0.33, random_state=1111)
print(train.head())

indt_train = train.drop(['id','vendor_id','pickup_datetime','passenger_count','dropoff_datetime','trip_duration','store_and_fwd_flag'], axis=1)
dept_train = train['trip_duration']

indt_test = test.drop(['id','vendor_id','pickup_datetime','passenger_count','dropoff_datetime','trip_duration','store_and_fwd_flag'], axis=1)
dept_test = test['trip_duration']

# indt_train, indt_test, dept_train, dept_test = train_test_split(indt, dept, test_size=0.33, random_state=1111)
# #indt_train, indt_test, dept_train, dept_test = train_test_split(indt, dept, test_size=0.33)
# #indt_train, indt_val, dept_train, dept_val  = train_test_split(indt_train, dept_train, test_size=0.2)
# indt_train, indt_val, dept_train, dept_val  = train_test_split(indt_train, dept_train, test_size=0.2, random_state=2222)

##Linear Regression Model######

model_features = list(indt_train.columns)
print(model_features)

linreg = linear_model.LinearRegression(fit_intercept=True)
linreg.fit(indt_train, dept_train)
test_dept_predict_LP = linreg.predict(indt_test)
print("The intercept is:{}".format(linreg.intercept_))
print("Following are the estimated coefficients")
print(list(zip(linreg.coef_, model_features)))
#print(linreg.score(indt_train, dept_train))

##RootMean Square Error
import math
# LR_rmse = np.sqrt((abs(test_dept_predict_LP - dept_test)).mean())
# print("Linear Regression Results")
# print("RMSE: {}".format(LR_rmse))
### Mean absolute error ####
print((abs(test_dept_predict_LP - dept_test)).mean())

######XGBoost######

from xgboost import XGBRegressor
from xgboost import plot_importance
from matplotlib import pyplot

model_xgb = XGBRegressor()

XGBRegressor(base_score=0.5, booster='gbtree', objective='reg:squarederror', colsample_bylevel=1, colsample_bytree=1,
              learning_rate=0.3, max_depth=10, subsample=0.9,
              tree_method='exact', validate_parameters=1, verbosity=None)

model_xgb.fit(indt_train, np.log(dept_train + 1))
test_dept_predict_XGB = np.exp(model_xgb.predict(indt_test)) - 1
print(type(indt_test))

### Mean absolute Error ###
print((abs(test_dept_predict_XGB - dept_test)).mean())

print(test_dept_predict_XGB[1001], dept_test.iloc[1001])
print(test_dept_predict_LP[1001])

## Feature importance #####
plot_importance(model_xgb)
pyplot.show()


## Saving Model###
import pickle
filename = "XGBoost_model.sav"
pickle.dump(model_xgb, open(filename, 'wb'))


######XGBoost######

# train, val = train_test_split(train, test_size=0.2, random_state=1111)
# indt_train = train.drop(['id','vendor_id','pickup_datetime','passenger_count','dropoff_datetime','trip_duration','store_and_fwd_flag'], axis=1)
# dept_train = train['trip_duration']

# indt_val = val.drop(['id','vendor_id','pickup_datetime','passenger_count','dropoff_datetime','trip_duration','store_and_fwd_flag'], axis=1)
# dept_val = val['trip_duration']

# indt_train, indt_val, dept_train, dept_val = train_test_split(indt_train, dept_train, test_size=0.2, random_state=100)

# def rmse(dept_true, dept_pred):
#     assert len(dept_true) == len(dept_pred)
#     return np.square(np.log(dept_pred + 1) - np.log(dept_true + 1)).mean() **0.5



# #XGBoost parameters 
# params = {
#     'booster':            'gbtree',
#     'objective':          'reg:linear',
#     'learning_rate':      0.1,
#     'max_depth':          14,
#     'subsample':          0.9,
#     'colsample_bytree':   0.7,
#     'colsample_bylevel':  0.7,
#     'silent':             1,
#     'feval':              'rmse'
# }

# iterations = 1000

# ##Defining train and validation sets
# d_indt_train = xgb.DMatrix(indt_train, np.log(dept_train + 1))
# val = xgb.DMatrix(indt_val, np.log(dept_val+1))

# check = [(val,'eval'), (d_indt_train, 'train')]

# ##Train model

# model_xgb = xgb.train(params, d_indt_train, num_boost_round = iterations, evals=check, verbose_eval=True)

# ##Predictions#####
# xgb_pred = np.exp(model_xgb.predict(xgb.DMatrix(indt_test))) - 1
# ##Root mean square error
# rmse_xgb = np.sqrt((abs(xgb_pred - dept_test)).mean())
# print("RMSE XGB: {}".format(rmse_xgb))
# print('R2:{}'.format(r2_score(dept_test, xgb_pred)))

# print(mean_squared_error( xgb_pred, dept_test))
# print((abs(xgb_pred - dept_test)).mean())



