import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.decomposition import RandomizedPCA
matplotlib.style.use('ggplot')

def get_outliers_index(df, columns, gama = 1.5):
    index_to_drop = []
    for column in columns:
        q2 = df[column].median()
        q3 = df[df[column] > q2][column].median()
        q1 = df[df[column] < q2][column].median()
        IQR = q3 - q1
        index_to_drop += list(df[(df[column] > q3 + gama*IQR) | (df[column] < q1 - gama*IQR)][column].index.values)

    return list(np.unique(index_to_drop))

#loading CSV files
p1_data_test_df = pd.read_csv('p1_data_test.csv',header=0)
p1_data_train_df = pd.read_csv('p1_data_train.csv',header=0)

#removing the outliers
print "Removing outliers..."
gama = 1.5
index_to_drop = get_outliers_index(p1_data_train_df,['Temp1','Temp2','Temp3','Temp4'], gama = gama)
p1_data_train_df = p1_data_train_df.drop(p1_data_train_df.index[index_to_drop])


#getting de data traing, using 30% of the values
pct_train = 1
data_train_df = p1_data_train_df.sample(frac= pct_train)

data_train = data_train_df[['Temp1','Temp2', 'Temp3', 'Temp4']].values
target = data_train_df[['target']].values
l, c = np.shape(target)
target = np.reshape(target,(c,l))[0]

#training the models
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
classifyer = linear_model.LogisticRegression(C=1, max_iter = 5000, solver = 'lbfgs')

#extracting the most important features from the dataset
n_components = 4
pca = RandomizedPCA(n_components = n_components, whiten=True).fit(data_train)
data_train_pca = pca.transform(data_train)

for item in zip(pca.explained_variance_,pca.explained_variance_ratio_):
    print item

print "training the model..."
scaler = StandardScaler()
data_train = scaler.fit_transform(data_train_pca)

classifyer.fit(data_train, target)
#predicting the test values for all values of the train test
data_test_df = p1_data_train_df
data_test = data_test_df[['Temp1','Temp2', 'Temp3','Temp4']].values
data_test_pca = pca.transform(data_test)
target_test = data_test_df[['target']].values

print "predicting the values..."

data_test = scaler.fit_transform(data_test_pca)
predicted_values = classifyer.predict(data_test_pca)
print "unique predicted values:", np.unique(predicted_values)

print "Log reg score:", classifyer.score(data_test, target_test)


print "prediting values from the data_test sample"
index_to_drop = []#get_outliers_index(p1_data_test_df,['Temp1','Temp2', 'Temp3','Temp4'], gama = gama)

p1_data_test_df = p1_data_test_df.drop(p1_data_test_df.index[index_to_drop])
print "%d itens droped out" % len(index_to_drop)

data_test = p1_data_test_df[['Temp1','Temp2', 'Temp3','Temp4']].values

predicted_test_values = classifyer.predict(data_test)
print len(predicted_test_values)
print np.unique(predicted_test_values)

print "Classifications:"
print "1:", len(predicted_test_values[predicted_test_values == 1])
print "0:", len(predicted_test_values[predicted_test_values == 0])

del p1_data_test_df
del p1_data_train_df
