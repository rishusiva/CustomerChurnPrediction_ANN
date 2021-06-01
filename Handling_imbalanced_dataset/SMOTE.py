from Cleaning_dataset import *

X = df2.drop('Churn',axis=1)
y = df2['Churn']

from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='minority')
X_smote,y_smote = smote.fit_resample(X,y)

#print(y_smote.value_counts())

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_smote,y_smote,test_size=0.2,random_state=40,stratify=y_smote)

#print(y_train.value_counts())