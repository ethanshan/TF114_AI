import tensorflow as tf

sess = tf.InteractiveSession()

raw_data = [1., 2., 3., 4., 5., 6., 7., 8.]
spikes = tf.Variable([False]*len(raw_data), name='spikes')
sess.run(spikes.initializer)

saver = tf.train.Saver()

saver.restore(sess, "spikes.ckpt")

print("The spikes value: ", spikes.eval())

sess.close()