
# coding: utf-8

# Temp1, Temp2 Temp3 e Temp4 são temperaturas medidas em diferentes partes da planta
# Target representa o estado da qualidade da amostra (temp1, temp2, temp3 e temp4) 

# In[20]:

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


#p1_data_test_df = pd.read_csv('p1_data_test.csv',header=0)
df = pd.read_csv('../p1_data_train.csv',header=0)

print len(df)
pct = int(len(df)*0.5)
print pct
new_df = df[df.index > pct]
new_df


# In[2]:

def get_outliers_index(df, columns, gama = 1.5):
    index_to_drop = []
    for column in columns:
        q2 = df[column].median()
        q3 = df[df[column] > q2][column].median()
        q1 = df[df[column] < q2][column].median()

        IQR = q3 - q1

        index_to_drop += list(df[(df[column] > q3 + gama*IQR) | (df[column] < q1 - gama*IQR)][column].index.values)

    return list(np.unique(index_to_drop))


# In[4]:

df.head()


index_to_drop = get_outliers_index(df,['Temp1','Temp2','Temp3','Temp4'])
print df.shape
print len(index_to_drop)
print index_to_drop
df = df.drop(df.index[index_to_drop])
print df.shape


# In[ ]:

data = {'name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
        'year': [2012, 2012, 2013, 2014, 2014],
        'reports': [4, 24, 31, 2, 3]}
df = pd.DataFrame(data, index = ['Cochice', 'Pima', 'Santa Cruz', 'Maricopa', 'Yuma'])
df
df.drop(df.index[[0,1,2]])


# In[192]:



