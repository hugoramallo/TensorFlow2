#Tensorflow with Eager Execution
import tensorflow as tf
import numpy as np
#re-enable the eager execution
from tensorflow.python.framework.ops import enable_eager_execution
enable_eager_execution()

#run tensorflow operations and get the results inmediately
x = [[4]]
m = tf.matmul(x, x)
print("Result, {}".format(m))

#easy to inspect the results using print
a = tf.constant(np.array([1., 2., 3.]))
type(a)
print(a.numpy())

"Isn't this amazing? So from now on we can treat Tensors like ordinary python objects, " \
"work with them as usual, " \
"insert debug statements at any point or even use a debugger. So let's continue this example:"

b = tf.constant(np.array([4.,5.,6.]))
c = tf.tensordot(a, b,1) #multiplacmos a*b
type(c)
#Again, c is an tensorflow.python.framework.ops.EagerTensor object which can be directly read:
print(c.numpy())

#Dynamic Control Flow
def fizzbuzz(max_num):
  counter = tf.constant(0)
  max_num = tf.convert_to_tensor(max_num)
  for num in range(1, max_num.numpy()+1):
    num = tf.constant(num)
    if int(num % 3) == 0 and int(num % 5) == 0:
      print('FizzBuzz')
    elif int(num % 3) == 0:
      print('Fizz')
    elif int(num % 5) == 0:
      print('Buzz')
    else:
      print(num.numpy())
    counter += 1

fizzbuzz(15)