import pandas as pd
import numpy as np
import matplotlib
#matplotlib.style.use('ggplot')

#loading CSV files
p1_data_test_df = pd.read_csv('p1_data_test.csv',header=0)
p1_data_train_df = pd.read_csv('p1_data_train.csv',header=0)

def plot_bar():
    from matplotlib import pyplot as plt

    k = 1
    p1_data_test_df[::k].plot.bar()
    plt.title("Temperatura em diferentes locais da planta")
    plt.show()

def plot_lines():
    from matplotlib import pyplot as plt

    fig = plt.figure()
    k = 1
    ax1 = fig.add_subplot(2,2,1)
    ax1.plot(p1_data_test_df['Temp1'][::k])
    ax1.set_title('Temp1')

    ax1 = fig.add_subplot(2,2,2)
    ax1.plot(p1_data_test_df['Temp2'][::k])
    ax1.set_title('Temp2')

    ax1 = fig.add_subplot(2,2,3)
    ax1.plot(p1_data_test_df['Temp3'][::k])
    ax1.set_title('Temp3')

    ax1 = fig.add_subplot(2,2,4)
    ax1.plot(p1_data_test_df['Temp4'][::k])
    ax1.set_title('Temp4')

    plt.show()

#plt.plot(np.random.randn(100).cumsum(),'*-')
#plt.show()
def plot_():
    from matplotlib import pyplot as plt
    k = 1
    plt.plot(p1_data_test_df['Temp1'][::k], '*-',color='green',label='Temp1')
    plt.plot(p1_data_test_df['Temp2'][::k], '*-',color='blue',label='Temp2')
    plt.plot(p1_data_test_df['Temp3'][::k], '*-',color='red',label='Temp3')
    plt.plot(p1_data_test_df['Temp4'][::k], '*-',color='black',label='Temp4')
    plt.legend()
    plt.show()

from matplotlib import pyplot as plt
tmp1 = p1_data_test_df['Temp3']
tmp1.plot()
plt.show()
