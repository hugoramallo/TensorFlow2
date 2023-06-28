import tensorflow as tf
"""
if not tf.__version__ == '2.9.0':
    print(tf.__version__)
    raise ValueError('please upgrade to TensorFlow 2.9.0, or restart your Kernel (Kernel->Restart & Clear Output)')"""

#verify tensorflow is executing
tf.executing_eagerly()

#operations without eager execution mode
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

#verify if eager execution is working
print(tf.executing_eagerly())

#object type tensorflow
import numpy as np
a = tf.constant(np.array([1., 2., 3.]))
print(type(a))

#another tensor "b"
b = tf.constant(np.array([4.,5.,6.]))
c = tf.tensordot(a, b, 1)
print(type(c))

#execute tensorflow graph inside a tensorflow session
#creating the session
session = tf.compat.v1.Session()
output = session.run(c)
session.close()
print(output)






