import os
import tensorflow as tf

from tensorflow.python.client import device_lib

def get_available_gpus(allowed_gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = allowed_gpu
    local_device_protos = device_lib.list_local_devices()
    return [str(x.name) for x in local_device_protos if x.device_type == 'GPU']


class ModelBuilder():
    def __init__(self, args):
        self.args = args
        self._batch_size = args.batch_size
        self._available_devices = get_available_gpus(args.gpus)
        with tf.name_scope("PLACE_HOLDER"):
            with tf.name_scope("IS_TRAIN"):
                self._is_training = tf.placeholder(tf.bool)
            with tf.name_scope("GT_CLASSIFICATION"):
                self._classification_gt = tf.placeholder(tf.float32, [None, 10])
            with tf.name_scope("IMG_INPUT"):
                self._img_input = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
        
        self._pred = None
        self._labels = None
        self._loss = None
        self._train_optim=None
        self._optimizer = None
        self.update_ops = None

        self._build_models()

    def _build_models(self):
        print("BUILDING THE MODEL")
        split_size = self._batch_size // len(self._available_devices)
        splits = [split_size, ] * (len(self._available_devices) - 1)
        splits.append(self._batch_size - split_size * (len(self._available_devices) - 1))

        img_input_split = tf.split(self._img_input, splits, axis = 0)
        labels_split = tf.split(self._classification_gt, splits, axis=0)

        tower_grads = []
        tower_loss = []

        tower_pred = []
        tower_logit = []
        tower_label = []

        self._optimizer = tf.train.AdamOptimizer(self.args.learning_rate)
        #loss_scale_manager = tf.contrib.mixed_precision.FixedLossScaleManager(self._loss_scale)
        #loss_scale_optimizer = tf.contrib.mixed_precision.LossScaleOptimizer(self._optimizer, loss_scale_manager)

        with tf.variable_scope(tf.get_variable_scope()):
            for gpu_idx, gpu in enumerate(self._available_devices):
                with tf.device(gpu):
                    with tf.name_scope("tower_%d"%(gpu_idx)) as scope:
                        print("TOWER "+str(gpu_idx))
                        tower_batch_size=img_input_split[gpu_idx].get_shape().as_list()[0]
                        cnn_model=CnnModel(self.args, scope, img_input_split[gpu_idx], self._is_training)
                        cnn_feature_map = cnn_model.return_features()
                        classifier_end_point = VideoClassifier(self.args, cnn_feature_map, self._is_training).get_logit()
                        out_preds = tf.nn.softmax(classifier_end_point)

                    tower_pred.append(tf.argmax(input=out_preds, axis=1))
                    tower_logit.append(out_preds)
                    tower_label.append(tf.argmax(input=labels_split[gpu_idx], axis=1))

                    self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    loss = self._get_loss(scope, classifier_end_point, labels_split[gpu_idx])

                    tf.get_variable_scope().reuse_variables()

                    grads = self._optimizer.compute_gradients(loss)

                    tower_grads.append(grads)
                    tower_loss.append(loss)

        with tf.device(self._available_devices[0]):
            self._classifier_pred = tf.concat(tower_pred, -1)
            self._classifier_logit = tf.concat(tower_logit, 0)
            self._classifier_labels = tf.concat(tower_label, -1)
            self._loss = tf.add_n(tower_loss, name="total_loss")

            averaged_grads = self._average_gradients(tower_grads)

            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            apply_gradient_op = self._optimizer.apply_gradients(averaged_grads, global_step=tf.train.get_global_step())

            variable_averages = tf.train.ExponentialMovingAverage(0.9999, tf.train.get_global_step())
            variables_averages_op = variable_averages.apply(tf.trainable_variables())

            with tf.control_dependencies(self.update_ops):
                self._train_optim = tf.group(apply_gradient_op, variables_averages_op)

    def _get_loss(self, scope, classifier_end_point, labels):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=classifier_end_point,
            labels=labels))

    
    def _average_gradients(self, tower_grads):
        average_grads = []

        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))

            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                if g is not None:
                    expanded_g = tf.expand_dims(g, 0)

                    # Append on a 'tower' dimension which we will average over below.
                    grads.append(expanded_g)

            if len(grads) > 0:
                # Average over the 'tower' dimension.
                grad = tf.concat(axis=0, values=grads)
                grad = tf.reduce_mean(grad, 0)

                # Keep in mind that the Variables are redundant because they are shared
                # across towers. So .. we will just return the first tower's pointer to
                # the Variable.
                v = grad_and_vars[0][1]
                grad_and_var = (grad, v)
                average_grads.append(grad_and_var)
        return average_grads

    #loss, pred = model.train_step(sess, images, gt_classes_hot)
    def train_step(self, sess, images, gt_classes_hot):
        # print(images)
        # print(gt_classes_hot)
        # exit()
        #print(self._loss, self._train_optim, self._classifier_pred, self._classifier_labels)
        
        loss, _, pred_logit, pred = sess.run(
            [
                self._loss,
                self._train_optim,
                self._classifier_logit,
                self._classifier_pred,
            ],
            feed_dict={
                self._img_input: images,
                self._classification_gt: gt_classes_hot,
                self._is_training: True,
            })
        
        return loss, pred_logit, pred
    
    #loss, logit, pred = model.validation_step(sess, images, gt_classes_hot)
    def validation_step(self, sess, images, gt_classes_hot):
        loss, pred_logit, pred = sess.run(
            [
                self._loss,
                self._classifier_logit,
                self._classifier_pred,
            ],
            feed_dict={
                self._img_input: images,
                self._classification_gt: gt_classes_hot,
                self._is_training: False,
            })
        
        return loss, pred_logit, pred

class CnnModel():
    def __init__(self, args, scope, img_input, _is_training):
        self.args = args
        with tf.variable_scope("bottom"):#("%s/bottom"%(scope)):
            conv1 = tf.layers.conv2d(img_input, 128, [5,5], [1,1], "SAME", name="conv2d_1")
            pool1 = tf.layers.max_pooling2d(conv1, [5,5], [1,1], "SAME", name="maxpool_1")
            conv2 = tf.layers.conv2d(pool1, 128, [3,3], [1,1], "SAME", name="conv2d_2")
            pool2 = tf.layers.max_pooling2d(conv2, [3,3], [1,1], "SAME", name="maxpool_2")
            conv3 = tf.layers.conv2d(pool2, 256, [3,3], [1,1], "SAME", name="conv2d_3")
            pool3 = tf.layers.max_pooling2d(conv3, [3,3], [1,1], "SAME", name="maxpool_3")
            conv4 = tf.layers.conv2d(pool3, 256, [3,3], [1,1], "SAME", name="conv2d_4")
            pool4 = tf.layers.max_pooling2d(conv4, [3,3], [2,2], "SAME", name="maxpool_4")
            conv5 = tf.layers.conv2d(pool4, 512, [3,3], [1,1], "SAME", name="conv2d_5")
            pool5 = tf.layers.max_pooling2d(conv5, [3,3], [2,2], "SAME", name="maxpool_5")
            conv6 = tf.layers.conv2d(pool5, 512, [3,3], [1,1], "SAME", name="conv2d_6")
            pool6 = tf.layers.max_pooling2d(conv6, [3,3], [2,2], "SAME", name="maxpool_6")
            conv7 = tf.layers.conv2d(pool6, 512, [3,3], [1,1], "SAME", name="conv2d_7")

            self._features = tf.layers.max_pooling2d(conv7, [3,3], [2,2],"SAME", name="maxpool_7")

    def return_features(self):
        return self._features

class VideoClassifier():
    def __init__(self, args, features, _is_training):
        
        self.args = args
        self._features = features
        self._is_training = _is_training
        self._num_class = args.num_classes

        with tf.variable_scope("CLASSIFIER"):
            end_point = tf.reshape(features, [features.shape[0],-1])

            mid_logit = tf.layers.dense(end_point, 64, 
             kernel_initializer=tf.variance_scaling_initializer(), name="dense_1")

            self.logit = tf.layers.dense(mid_logit, self._num_class, 
             kernel_initializer=tf.variance_scaling_initializer(), name="dense_2")

    def get_logit(self):
        return self.logit

#TODO
class CnnLiteModel():
    def __init__(self, args):
        a=1