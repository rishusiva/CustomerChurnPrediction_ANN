import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv("Customer_churn.csv")

df.drop('customerID',axis=1,inplace=True)

#print(df.dtypes)

#print(df.TotalCharges.values)

df[pd.to_numeric(df.TotalCharges,errors='coerce').isnull()]

#print(df.iloc[488]['TotalCharges'])

df1=df[df.TotalCharges!=' ']

df1.TotalCharges=pd.to_numeric(df1.TotalCharges)
#print(df1.TotalCharges.dtype)

df1.replace('No internet service','No',inplace=True)
df1.replace('No phone service','No',inplace=True)

yes_no_columns=['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']
for columns in yes_no_columns:
    df1[columns].replace({'Yes':1,'No':0},inplace=True)

df1['gender'].replace({'Female':1,'Male':0},inplace=True)

df2=pd.get_dummies(data=df1,columns=['InternetService','Contract','PaymentMethod'])
