#Los tensorflow son especialmente buenos para trabajar con imágenes
#debido a que para los colores usamos 3 dimensiones RGB, TensorFlow se hace muy útil manejando esto
import tensorflow as tf

"""if not tf.__version__ == '2.9.0':
    print(tf.__version__)
    raise ValueError('please upgrade to TensorFlow 2.9.0')"""

"""Now we call the TensorFlow functions that construct new tf.Operation and tf.Tensor objects. 
As mentioned, each tf.
Operation is a node and each tf.Tensor is an edge in the graph."""
a = tf.constant([2], name = 'constant_a')
b = tf.constant([3], name = 'constant_b')

a
"""As you can see, it just shows the name, shape and type of the tensor in the graph. 
We will see it's value by running the TensorFlow code as shown below."""
tf.print(a.numpy()[0])

#Tensor flow static execution
#función de suma
@tf.function
def add(a,b):
    c = tf.add(a, b)
    #c = a + b is also a way to define the sum of the terms
    print(c)
    return c
result = add(a,b)
tf.print(result[0])

#Definiendo arrays multidimensionnales usando tensorflow

Scalar = tf.constant(2)
Vector = tf.constant([5,6,2])
Matrix = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
Tensor = tf.constant( [ [[1,2,3],[2,3,4],[3,4,5]] , [[4,5,6],[5,6,7],[6,7,8]] , [[7,8,9],[8,9,10],[9,10,11]] ] )

print ("Scalar (1 entry):\n %s \n" % Scalar) #un número

print ("Vector (3 entries) :\n %s \n" % Vector) #un vector

print ("Matrix (3x3 entries):\n %s \n" % Matrix) #una matriz

print ("Tensor (3x3x3 entries) :\n %s \n" % Tensor) #un tensor (3 dimensiones)

#devuelve la forma de nuestra estructura de datos
Scalar.shape
Tensor.shape

Matrix_one = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
Matrix_two = tf.constant([[2,2,2],[2,2,2],[2,2,2]])

@tf.function
def add():
    add_1_operation = tf.add(Matrix_one, Matrix_two)
    return add_1_operation


print ("Defined using tensorflow function :")
add_1_operation = add()
print(add_1_operation)
print ("Defined using normal expressions :")
add_2_operation = Matrix_one + Matrix_two
print(add_2_operation)

#Si queremos la multiplicación típica de dos matrices

Matrix_one = tf.constant([[2,3],[3,4]])
Matrix_two = tf.constant([[2,3],[3,4]])

@tf.function
def mathmul():
  return tf.matmul(Matrix_one, Matrix_two)


mul_operation = mathmul()

print ("Defined using tensorflow function :")
print(mul_operation)

#Creamos una variable contador
v = tf.Variable(0)

#Un método de incremento
@tf.function
def increment_by_one(v):
        v = tf.add(v,1)
        return v

#el bucle llama a la función para incrementar la variable en 1
for i in range(3):
    v = increment_by_one(v)
    print(v)

#Las operaciones son nodos que representan las operaciones matemáticas sobre los tensores en un gráfico


