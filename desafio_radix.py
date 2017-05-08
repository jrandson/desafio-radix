import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.style.use('ggplot')

def get_outliers_index(df, columns, gama = 1.5):
    index_to_drop = []
    for column in columns:
        q2 = df[column].median()
        q3 = df[df[column] > q2][column].median()
        q1 = df[df[column] < q2][column].median()

        IQR = q3 - q1

        print len(index_to_drop)
        index_to_drop += list(df[(df[column] > q3 + gama*IQR) | (df[column] < q1 - gama*IQR)][column].index.values)

    return list(np.unique(index_to_drop))

#loading CSV files
p1_data_test_df = pd.read_csv('p1_data_test.csv',header=0)
p1_data_train_df = pd.read_csv('p1_data_train.csv',header=0)

#removing the outliers
print "Removing outliers..."
gama = 3
index_to_drop = get_outliers_index(p1_data_train_df,['Temp1','Temp2','Temp3','Temp4'], gama = gama)
p1_data_train_df = p1_data_train_df.drop(p1_data_train_df.index[index_to_drop])


#getting de data traing, using 30% of the values
pct_train = 0.9
count_train = int(len(p1_data_train_df)*pct_train)
count_test = int((1 - pct_train)*len(p1_data_train_df))

#data_train_df = p1_data_train_df[p1_data_train_df.index <= count_train]
data_train_df = p1_data_train_df.sample(frac=.3)

data_train = data_train_df[['Temp1','Temp2', 'Temp3', 'Temp4']].values
target = data_train_df[['target']].values
l, c = np.shape(target)
target = np.reshape(target,(c,l))[0]

#training the models
from sklearn.preprocessing import StandardScaler

from sklearn import linear_model
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

#classifyer = linear_model.LogisticRegression(C=1, max_iter = 1000, solver = 'lbfgs') #solver = 'sag, lbfgs'

#kernel = linea, poly, rbf
#, degree = 3
classifyer = svm.SVC(kernel = 'rbf',C = 1e2, gamma='auto', max_iter = -1)

#classifyer = DecisionTreeClassifier(min_samples_split = 50, random_state=0)

#classifyer = GaussianNB()

#classifyer = MLPClassifier(solver='lbfgs', alpha = 1e-5,hidden_layer_sizes = (4), random_state=1)

print "training the model..."
scaler = StandardScaler()
data_train = scaler.fit_transform(data_train)
classifyer.fit(data_train, target)

#predicting the test values for all values of the train test
#data_test_df = p1_data_train_df[p1_data_train_df.index > count_test]
data_test_df = p1_data_train_df
data_test = data_test_df[['Temp1','Temp2', 'Temp3','Temp4']].values
target_test = data_test_df[['target']].values

print "predicting the values..."
data_test = scaler.fit_transform(data_test)
predicted_values = classifyer.predict(data_test)
print np.unique(predicted_values)

#getting accuracy
rate = 0.0
for i in range(len(predicted_values)):
    if target_test[i] == predicted_values[i]:
        rate+= 1

accuracy = 1.0*rate / len(predicted_values)
print "Accuracy:", accuracy
#print "Score:", logreg.score(data_test, target_test)

#Accuracy logist regression: 0.892877711653
#Accuracy SVM: 0.896717220196
#Accuracy decision tree: 0.88769437512
#Accuracy decision Naive Bayes: 0.895373392206

print "prediting values from the data_test sample"
index_to_drop = get_outliers_index(p1_data_test_df,['Temp1','Temp2', 'Temp3','Temp4'], gama = gama)

print len(index_to_drop)

p1_data_test_df = p1_data_test_df.drop(p1_data_test_df.index[index_to_drop])

data_test = p1_data_test_df[['Temp1','Temp2', 'Temp3','Temp4']].values

predicted_test_values = classifyer.predict(data_test)
print len(predicted_test_values)
print np.unique(predicted_test_values)

print "1:", len(predicted_test_values[predicted_test_values == 1])
print "0:", len(predicted_test_values[predicted_test_values == 0])


del p1_data_test_df
del p1_data_train_df
