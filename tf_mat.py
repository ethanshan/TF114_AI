import tensorflow as tf
import numpy as np
import time


# -------------------- Input data from matlab .mat file ---------------------
# Df 601x400
Df = np.loadtxt(open("/home/ethanshan/Codes/TF114_AI/data/Df.csv", "rb"), dtype=np.float , delimiter=",")
print("Df shape: " + str(Df.shape))

# exp_Input 400x1
exp_Input = np.loadtxt(open("/home/ethanshan/Codes/TF114_AI/data/exp_Input.csv", "rb"), delimiter=",")
print("exp_Input shape: " + str(exp_Input.shape))

# MAT_obs = 400x601
MAT_obs = np.loadtxt(open("/home/ethanshan/Codes/TF114_AI/data/MAT_obs.csv", "rb"), delimiter=",")
print("MAT_obs shape: " + str(MAT_obs.shape))

start_time = time.time()
result = Df@exp_Input

for i in range(1, 1):
    result = np.matmul(Df, exp_Input)
    print(result.shape)
end_time = time.time()

print("start time ", start_time, " \t end time: ", end_time)



m1 = [[1.0, 2.0], [3.0, 4.0]]

m2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

m3 = tf.constant([[1.0, 2.0], [3.0, 40]])

print(type(m1))
print(type(m2))
print(type(m3))

t1 = tf.convert_to_tensor(m1, dtype=tf.float32)
t2 = tf.convert_to_tensor(m2, dtype=tf.float32)
t3 = tf.convert_to_tensor(m3, dtype=tf.float32)

print(type(t1))
print(type(t2))
print(type(t3))


x = tf.constant([[1.0, 2.0, 3.0]])

y = tf.negative(x)

with tf.Session() as sess:
    result = sess.run(y)
    print(result)