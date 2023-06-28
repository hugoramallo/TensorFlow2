import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

#read de file
df = pd.read_csv("FuelConsumption.csv")

# take a look at the dataset
df.head()
#lets define X and Y value for the linear regression, that is, train_x and train_y:
train_x = np.asanyarray(df[['ENGINESIZE']])
train_y = np.asanyarray(df[['CO2EMISSIONS']])

#First, we initialize the variables a and b, with any random guess, and then we define the linear function:
a = tf.Variable(20.0)
b = tf.Variable(30.2)

#formula linear regression
def h(x):
   y = a*x + b
   return y

#To find value of our loss, we use tf.reduce_mean(). This function finds the mean of a
# multidimensional tensor, and the result can have a different dimension.
def loss_object(y, train_y):
   return tf.reduce_mean(tf.square(y - train_y))
   # Below is a predefined method offered by TensorFlow to calculate loss function
   # loss_object = tf.keras.losses.MeanSquaredLogarithmicError()

#start training and run the graph
learning_rate = 0.01
train_data = []
loss_values = []
a_values = []
b_values = []
# steps of looping through all your data to update the parameters
training_epochs = 200

# train model
for epoch in range(training_epochs):
   with tf.GradientTape() as tape:
      y_predicted = h(train_x)
      loss_value = loss_object(train_y, y_predicted)
      loss_values.append(loss_value)

      # get gradients
      gradients = tape.gradient(loss_value, [b, a])

      # compute and adjust weights
      a_values.append(a.numpy())
      b_values.append(b.numpy())
      b.assign_sub(gradients[0] * learning_rate)
      a.assign_sub(gradients[1] * learning_rate)
      if epoch % 5 == 0:
         train_data.append([a.numpy(), b.numpy()])

#Lets plot the loss values to see how it has changed during the training:
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(loss_values, 'ro')

"""Lets visualize how the coefficient and intercept of line has changed to fit the data:
The green dots are the data points, the red lines are created using the a and b coefficients during training, 
and the black line is the line we use to model the relationship with the final/last coefficients."""

plt.scatter(train_x, train_y, color='green')
for a,b in zip(a_values[0:len(a_values)], b_values[0:len(b_values)]):
    plt.plot(train_x,a*train_x+b, color='red', linestyle='dashed')
plt.plot(train_x,a_values[-1]*train_x+b_values[-1], color='black')

final = mpatches.Patch(color='Black', label='Final')
estimates = mpatches.Patch(color='Red', label='Estimates')
data = mpatches.Patch(color='Green', label='Data Points')

plt.legend(handles=[data, estimates, final])

plt.show()