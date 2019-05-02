import tensorflow as tf
import numpy as np
import cv2

sessions={}
sessions["train"] = ["data/train/%05d.npy"%(i) for i in range(60000)]
sessions["valid"] = ["data/test/%05d.npy"%(i) for i in range(10000)]

class ImageLoader(object):
    def __init__(self, args, data_split):

        self._batch_size = args.batch_size
        self._data_split = data_split

        self.image_dirs = sessions[data_split]
        if data_split=="train":
            label_dir = "data/train-labels.npy"
        else:
            label_dir = "data/test-labels.npy"
        self.label_vals = ["%d"%(x) for x in np.load(label_dir)]

        self._queue = tf.train.slice_input_producer(
            [tf.convert_to_tensor(self.image_dirs, dtype=tf.string),
             tf.convert_to_tensor(self.label_vals, dtype=tf.string)],
             shuffle=False,
             seed=1007#,
             #capacity=16
        )

        self.image, self.label, self.image_name = self.read_data()
    
    def dequeue(self, num_threads):
        return tf.train.batch([self.image, self.label, self.image_name],
                              batch_size = self._batch_size,
                              #capacity = 16,
                              num_threads = num_threads)

    def read_data(self):
        with tf.variable_scope("NORMALIZE_IMAGES"):
            image, label, image_name = tf.py_func(self.py_function, [self._queue], [tf.float32, tf.int32, tf.string])
            image_shape = (28,28,1)
            label_shape = (1, )
            image = tf.reshape(image, image_shape)
            label = tf.reshape(label, label_shape)
            image_name = tf.reshape(image_name, label_shape)

            image = (tf.cast(image, tf.float32))
            label = (tf.cast(label, tf.int32))
            image_name = (tf.cast(image_name, tf.string))
        return image, label, image_name
    
    def py_function(self, queue_data):
        image_name = self._data_split+"_"+queue_data[0].decode('utf-8').split("/")[-1].split(".")[0]
        image = np.load(queue_data[0].decode('utf-8'))
        label = int(queue_data[1].decode('utf-8'))
        return np.array(image,dtype=np.float32), np.array(label,dtype=np.int32), image_name


def load_data(args):

    tf.set_random_seed(args.seed)

    output_dic = dict()

    with tf.name_scope("IMAGE_LOADER"):
        train_reader = ImageLoader(args, "train")
        val_reader = ImageLoader(args, "valid")
        output_dic["train_batch"] = train_reader.dequeue(args.num_threads)
        output_dic["val_batch"] = val_reader.dequeue(args.num_threads)
    return output_dic