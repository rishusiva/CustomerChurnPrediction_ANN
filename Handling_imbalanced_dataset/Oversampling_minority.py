from Cleaning_dataset import *

count_class_0,count_class_1 = df2.Churn.value_counts()

df_class_0 = df2[df2['Churn']==0]
df_class_1 = df2[df2['Churn']==1]

df_class_1_new = df_class_1.sample(count_class_0,replace=True)

#print(df_class_1_new.shape)

df_new = pd.concat([df_class_0,df_class_1_new],axis=0)

#print(df_new.shape)

X = df_new.drop('Churn',axis=1)
y = df_new['Churn']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=40,stratify=y)

#print(y_train.value_counts())