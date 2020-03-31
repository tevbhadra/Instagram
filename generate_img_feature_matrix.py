
# coding: utf-8
# Code written by Eswara Veerabhadra
# In[2]:

import pandas as pd
import numpy as np
import ast
import os
os.chdir('/home/bduser/BigDataLabProjects/qcri_instagram/src/python')
#%%
ww = pd.read_csv('../../data/instagram_20k.csv')
ww=ww.drop(ww.columns[0], axis=1)
#%%
ww.LabelsValuesFinal = ww.LabelsValuesFinal.apply(ast.literal_eval)
ww.LabelsFinal = ww.LabelsFinal.apply(ast.literal_eval)
#%%
no_features = 100
z = []
for x in ww.LabelsFinal:
    if (len(x)) != 0:
        z.append(x)
bha = []
for a in z:
  for b in a:
   bha.append(b)

bhav = np.array(bha)
arr = np.array(bha)
u, count = np.unique(arr, return_counts=True)
count_sort_ind = np.argsort(-count)[:no_features] #Number od elements
u[count_sort_ind]
ww.columns
ww['LabelsFinalcombo'] = " "
for i in range(len(ww.LabelsFinal)):
    ww['LabelsFinalcombo'][i] = set(zip(ww.LabelsFinal[i],ww.LabelsValuesFinal[i]))
zz = u[count_sort_ind]
for x in zz:
    ww[x] = ""
#Achieved
for row_number, row in ww.iterrows():
    for label,value in row.LabelsFinalcombo:
      try:
          if label  in ww.columns:
              ww.loc[row_number,label] = value
      except:
          continue
#ww.to_csv('Imageoutput.csv')
#%%
img_feature_matrix = ww[ww.columns[-no_features:]]
