from Cleaning_dataset import *

#print(df2)

columns_scaling = ['tenure','MonthlyCharges','TotalCharges']

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

df2[columns_scaling]=scaler.fit_transform(df2[columns_scaling])

X = df2.drop('Churn',axis='columns')
y = df2['Churn']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=40)

#print(X_train.shape)
#print(X_test.shape)

