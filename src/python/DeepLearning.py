
# coding: utf-8

# In[1]:


from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten, BatchNormalization
from keras.layers import LSTM, Embedding, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import pandas as pd
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras import regularizers
from sklearn.utils import resample,shuffle
from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score


# In[33]:


df = pd.read_csv('datamatrix_with_Class.csv') #Read data matrix
df["no_char"]= (df["no_char"] - df["no_char"].mean())/df["no_char"].std() #Standard Normazlization
df["no_word"]= (df["no_word"] - df["no_word"].mean())/df["no_word"].std() #Standard Normalization
CLASS = df.iloc[:,-1] #Class Column in seperate list
df = df[df.columns[df.columns!="CLASS"]] #Remvoing Class from the dataframe
print(df.shape) #Shape of the data matrix


# In[32]:


df


# In[26]:


df_matrix = df.values 


# In[9]:


le = preprocessing.LabelEncoder() #Label encoding 
le.fit(CLASS.ravel()) #flattens the dimension
labels = le.transform(CLASS.ravel())
labels = labels.reshape(10000,1) 


# In[13]:


data_matrix = np.append(df_matrix, labels, axis=1) #appends the data matrix and labels
df1 = pd.DataFrame(data_matrix) 
one = df1.loc[df1[9347] == 0] #Seggregate the data
two = df1.loc[df1[9347] == 1]
three = df1.loc[df1[9347] == 2]
four = df1.loc[df1[9347] == 3]


# In[14]:

#Splitting the data (First 80%, next 10%, last 10%)
train_one, validate_one, test_one = np.split(one, [int(.8*len(one)), int(.9*len(one))]) 
train_two, validate_two, test_two = np.split(two, [int(.8*len(two)), int(.9*len(two))])
train_three, validate_three, test_three = np.split(three, [int(.8*len(three)), int(.9*len(three))])
train_four, validate_four, test_four = np.split(four, [int(.8*len(four)), int(.9*len(four))])


# In[19]:

#Appending Train, Valid, Test, 
train = train_one.append([train_two, train_three, train_four])
valid = validate_one.append([validate_two, validate_three, validate_four])
test = test_one.append([test_two, test_three, test_four])
print(train.shape)
print(valid.shape)
print(test.shape)


# In[21]:

#splitting x and y
train_x = train[train.columns[train.columns!=9347] ]
print(train_x.shape)
train_y = train.iloc[:,-1]
print(train_y.shape)

valid_x = valid[valid.columns[valid.columns!=9347] ]
print(valid_x.shape)
valid_y = valid.iloc[:,-1]
print(valid_y.shape)

test_x = test[test.columns[test.columns!=9347] ]
print(test_x.shape)
test_y = test.iloc[:,-1]
print(test_y.shape)


# In[23]:

#Neural Network
model = Sequential()
model.add(Embedding(9347, 50, input_length=9348, trainable=True))
model.add(LSTM(100)) #LSTM Layer, used for sequential data
model.add(Dense(1, activation='sigmoid')) #Activation function (Can experiment using Tanh, Relu, Softmax)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #loss function, optimizor (can experiment using RMSprop, Adaboost, SGD) 
print(model.summary()) #Give the summary of the network


# In[27]:

#running the model
model.fit(train_x, train_y,validation_data=(valid_x, valid_y),epochs=30, batch_size=5) #Can experiment changing the number of epochs


# In[ ]:

#Predciting values for test data
pred  = model.predict(test_x)



# In[ ]:

#Identifying the class/label based on maximum probabailty for each instance
predlist = []
for i in range(len(pred)):
    maximum = max(pred[i])
    predlist.append(pred[i].index(maximum))
        

# In[ ]:

#Generating Perfromance metrics (Precision, Recall, F1-Score)
from sklearn.metrics import classification_report
print(classification_report(test_y, predlist))

#Generating Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(test_y, predlist))


