
# coding: utf-8

# In[19]:

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.datasets import load_boston
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, auc
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
print("Packages Loaded", flush = True)

# In[3]:

#For merging all files
import numpy as np
import pandas as pd
import os
os.chdir('/home/bduser/BigDataLabProjects/ICS/QCRI')
#%%
d1 = pd.read_csv('datamatrix.csv')
d1.drop(d1.columns[[0]], axis=1, inplace=True)
#Read data matrix
print("First CSV Loaded",flush = True)
#%%
#d2 = pd.read_csv('data_matrix100k.csv')
#d2.drop(d2.columns[[0]], axis=1, inplace=True)
#print("Second CSV Loaded",flush = True)
##%%
#d3 = pd.read_csv('data_matrix150k.csv')
#d3.drop(d3.columns[[0]], axis=1, inplace=True)
#print("Third CSV Loaded",flush = True)
##%%
#d4 = pd.read_csv('data_matrix182k.csv')
#d4.drop(d4.columns[[0]], axis=1, inplace=True)
#print("Fourth CSV Loaded",flush = True)
##%%
#frames = [d1, d2, d3, d4]
#x = pd.concat(frames)
#print("Data matrix generated",flush = True)
#print(x.shape,flush = True)
##%%
#print("Loading Class Labels",flush = True)
#c = pd.read_csv("Class.csv")
#print(c.shape, flush = True)
#%%
Class = pd.read_csv("Class.csv")
Class.drop(Class.columns[[0]], axis=1, inplace=True)
x = d1
c = Class
#%%
#df_matrix = df.values


#data_matrix = np.append(df_matrix, c, axis=1) #appends the data matrix and labels 

#x1 = pd.DataFrame(data_matrix)
#x1.to_csv('QCRIFulldata.csv')

#one = x1.loc[x1[18191] == 1] #Seggregate the data
#two = x1.loc[x1[18191] == 2]
#three = x1.loc[x1[18191] == 3]
#four = x1.loc[x1[18191] == 4]
#x = one.append([two, three, four])
#print(x.shape, flush = True)


#x = pd.DataFrame(data_matrix) 
#print("Starting Cross Fold", flush = True)
 
x = d1.head(5000)
x = x[x.columns[0:100]]
c = c.head(5000)
#c = pd.DataFrame(c)


#print(x['18191'], flash = True)
#print(type(x['18191'][0]), flash = True)
#print(type(x['18191']), flash = True)
#x['18191'] = x['18191'].astype('category')

#from sklearn import preprocessing 
#encoder = preprocessing.LabelEncoder()
#c = encoder.fit_transform(c)


#%%
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
bootstrap = [True, False]
max_features = ['auto', 'sqrt']

random_grid = {'n_estimators': n_estimators,'max_features': max_features, 'bootstrap':bootstrap}
rf = RandomForestClassifier(n_jobs=-1)
search_model = RandomizedSearchCV(estimator = rf, 
                           param_distributions = random_grid,  
                           cv = 3,  
                           n_jobs = -1,
                           verbose=51,
                           n_iter=50)
print("Parameter Tuning")
search_model.fit(x,c.values.ravel())

#%%
best_params = search_model.best_params_
print("Best Parameters Values: ",best_params,flush=True)
print("Best Score: ",search_model.best_score_,flush=True)
#%%
XX = x
YY = c

n_splits = 3
kf = KFold(n_splits=n_splits, shuffle=True)

con = []
acc = []
re = []
pr = []
f1 = []
auc_val = []
d = {1,2,3}
print("Cross-Fold Validation",flush = True)
for e in d:
    print("Round: ", e, flush = True)
    for train_index, val_index in kf.split(XX):
        print(train_index)
        print(val_index)
        auc1 = []

        print('Modelling', flush = True)
        model = RandomForestClassifier(**best_params,n_jobs=-1)
        #model = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,  cv = 2,  n_jobs = -1)

        model.fit(XX.iloc[train_index], YY.iloc[train_index].values.ravel())
        #best_random = model.best_estimator_
        print('Predicting', flush = True)

        pred = model.predict(XX.iloc[val_index])
        pred_prb = model.predict_proba(XX.iloc[val_index])
        print('Metrics', flush = True)
        for i in [1,2,3,4]:
         fpr, tpr, thresholds = metrics.roc_curve(YY.iloc[val_index], np.max(pred_prb, axis=1), pos_label= i)
         auc1.append(auc(fpr, tpr))
        
        
        matrix = confusion_matrix(YY.iloc[val_index], pred)

        acc1 = matrix.diagonal()/matrix.sum(axis=1)
        re1 = recall_score(YY.iloc[val_index], pred, average=None)
        pr1 = precision_score(YY.iloc[val_index], pred, average=None)
        f11 = f1_score(YY.iloc[val_index], pred, average=None)
        


        acc.append(acc1)
        re.append(re1)
        pr.append(pr1)
        f1.append(f11)
        con.append(matrix)
        auc_val.append(auc1)


# In[37]:

Accuracy = pd.DataFrame(acc, columns =['1', '2', '3', '4'])

Precision = pd.DataFrame(pr, columns =['1', '2', '3', '4'])

Recall = pd.DataFrame(re, columns =['1', '2', '3', '4'])

F1score = pd.DataFrame(f1, columns =['1', '2', '3', '4'])

AUC = pd.DataFrame(auc_val, columns =['1', '2', '3', '4'])
tin = []
fin =list(Accuracy.mean())
fin.append('Accuracy')
fin
gin = list(Precision.mean())
gin.append('Precision')
gin
din = list(Recall.mean())
din.append('Recall')
din
jam= list(F1score.mean())
jam.append('F1score')
jam
aam = list(AUC.mean())
aam.append('AUC')
aam
tin.append(fin)
tin.append(gin)
tin.append(din)
tin.append(jam)
tin.append(aam)
tin
print(tin, flush=True)
Results_Randomforest = pd.DataFrame(tin,columns =['1', '2', '3', '4', 'Metrics']) 
Results_Randomforest.to_csv('RandomForestoutput.csv')

