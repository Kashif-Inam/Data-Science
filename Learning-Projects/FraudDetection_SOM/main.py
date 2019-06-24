''' Importing Liberaries '''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show


''' Importing the dataset '''
dataset = pd.read_csv('Credit_Card_Applications.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


''' Feature Scaling '''
sc = MinMaxScaler(feature_range = (0, 1))
x = sc.fit_transform(x)


''' Training the Self Organizing Map '''
                            #15 no of col  #default val
som = MiniSom(x= 10, y= 10, input_len= 15, sigma= 1.0, learning_rate= 0.5)
som.random_weights_init(x)                 # randomly initializing weights by giving data
som.train_random(data= x, num_iteration= 100)


''' Visualizing the results '''
bone()                                    # initializing the window of graphical representation
pcolor(som.distance_map().T)              # plotting trained mean-neuron distance to visualize and taking transpose as well
colorbar()                                # showing the colorbar to differentiate lower and higher values color
markers = ['o', 's']
colors = ['r', 'g']
for i, ex in enumerate(x):               # i is the index number, ex is the values in each index
    w = som.winner(ex)                   # finding the winning node for each customer data
    plot( w[0] + 0.5,                    # plotting the values of point x,y (w[0],w[1]), and adding with 0.5 to mark in the center
          w[1] + 0.5,
          markers[y[i]],                 # finding that the customer belongs to yes(0) category or no(1) category which is in y
          markeredgecolor = colors[y[i]],
          markerfacecolor = 'None',
          markersize = 10,
          markeredgewidth = 2)
show()


''' Finding the frauds '''
mappings = som.win_map(x)               # finding the mappings of each customer with respect to winning nodes plotted that which cutomer belongs to which mappings. And returns a dictionary
frauds = np.concatenate((mappings[(1, 6)], mappings[6, 4]), axis=0)            # finding the lists of each customer which belongs to that cells who involved in the fraud
frauds = sc.inverse_transform(frauds)   # we have the mean scaled values. But for the actual values we have to inverse them
print(frauds)                           # prints the data of customers who involved in fraud
