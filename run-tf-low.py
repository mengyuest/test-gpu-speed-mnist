import tensorflow as tf
import numpy as np
import os, sys, time

print("Using GPU: "+sys.argv[1])

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

tf.logging.set_verbosity(tf.logging.WARN)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# parameters
learning_rate=0.0002
n_trainset=800
batch_size=4
height=120
width=160
n_train_iter=1000
n_print_interval=100
n_classes=50

# define the graph IO
image_pl = tf.placeholder(tf.float32, shape=(batch_size,height,width,3)) # NHWC
label_pl = tf.placeholder(tf.int32, shape=(batch_size)) # N

# define the graph logic (model)
def model(image):

    conv1 = tf.layers.conv2d(image, 8, 5, activation=tf.nn.relu)
    conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
    conv2 = tf.layers.conv2d(conv1, 8, 3, activation=tf.nn.relu)
    conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
    fc1 = tf.layers.flatten(conv2)
    fc1 = tf.layers.dense(fc1, 16)
    out = tf.layers.dense(fc1, n_classes)
    return out

# define the graph logic (loss)
prediction = model(image_pl)
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
    logits=prediction, labels=tf.cast(label_pl, dtype=tf.int32)))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

# generate random array as images
fake_img=np.random.rand(n_trainset, height, width, 3)
fake_lbl=np.random.randint(n_classes, size=n_trainset)

# control the training and timing logic
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    seq=np.random.choice(n_trainset,batch_size,replace=False)

    t1=time.time()
    for step in range(1,n_train_iter+1):   
        _,loss = sess.run([train_op, loss_op], feed_dict={image_pl: fake_img[seq], label_pl: fake_lbl[seq]})
        if step%n_print_interval==0:
            print("step:%5d  loss:%.4f  time:%.4f"%(step, loss, time.time()-t1))
            t1=time.time()
