import tensorflow as tf

sess = tf.InteractiveSession()

raw_data = [1., 2., 3., 4., 5., 6., 7., 8.]
spikes = tf.Variable([False]*len(raw_data), name='spikes')
#spikes.initializer.run()
sess.run(spikes.initializer)

saver = tf.train.Saver()

for i in range(1, len(raw_data)):
    if (raw_data[i] - raw_data[i-1]) > 5:
        spikes_val = spikes.eval()
        spikes_val[i] = True
        updater = tf.assign(spikes, spikes_val)
        updater.eval()

spikes_val = spikes.eval()
spikes_val[1] = True
tf.assign(spikes, spikes_val).eval()

print("spikes data values: ", spikes.eval())
save_path = saver.save(sess, "spikes.ckpt")

print("spikes data saved in file: %s" % save_path)

sess.close()