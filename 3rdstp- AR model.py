

# Importing necessary libraries
import pandas as pd 
from matplotlib import pyplot as plt

#______________create AR model______________________

# ar model  autoreg used to import time series techniques to make data stationary
from statsmodels.tsa.api import AutoReg


# Read data from csv
dataset = pd.read_csv('sinewave.csv')
print (dataset)

#______________call AR model______________________

for i in range (1,10):

    ARMDEL =AutoReg(dataset,lags=i)
    ARMDEL_fit=ARMDEL.fit()
    ypredicted=ARMDEL_fit.predict(110,160)
    
    #Plot the data   
    plt.plot(dataset, color='b', label='original data') # Plot the csv file
    plt.legend()
    plt.plot(ypredicted, color='r', label ='AR predicted AVG') # Plot Rolling Mean
    plt.legend()
    plt.title('lags= ' + str(i), color ='r', fontsize = '20')
    #plt.text(0,0.5,'lags= ' + str(i))
    plt.show()



