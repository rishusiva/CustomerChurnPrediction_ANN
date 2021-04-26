from Cleaning_dataset import *
from Splitting_dataset import *

import tensorflow as tf
from tensorflow import keras

#print(df2)

#print("Training the model with NN of 1 hidden layer")
#print("")
model =  keras.Sequential([
    keras.layers.Dense(20,input_shape=(26,),activation='relu'),
    keras.layers.Dense(15,activation='relu'),
    keras.layers.Dense(1,activation='sigmoid'),
    
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy']
             )

model.fit(X_train,y_train,epochs=100)

model.save("model.h5")

#print("The accuracy of the model on the train set is 83%")
#print("")

model.evaluate(X_test,y_test)

#print("The performance of the model is 77%")
#print("")
#print("Training the model without hidden layer")
#print("")
model =  keras.Sequential([
    keras.layers.Dense(20,input_shape=(26,),activation='relu'),
    keras.layers.Dense(1,activation='sigmoid'),
    
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy']
             )

model.fit(X_train,y_train,epochs=100)

model.save("model2.h5")

#print("The accuracy of the model on the train set is 82%")
#print("")

model.evaluate(X_test,y_test)

#print("The performance of the model on the train set is 78%")
#print("")

