import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import tensorflow as tf
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 6)

#independent variable
X = np.arange(0.0, 5.0, 0.1)
X
##You can adjust the slope and intercept to verify the changes in the graph
a = 1
b = 0

#linear regression formula
Y= a * X + b
#plot
plt.plot(X, Y)
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

