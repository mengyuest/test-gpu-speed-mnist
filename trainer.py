import os,sys,time
import tensorflow as tf
import numpy as np
import argparse
import psutil


import data_loader
import model_builder

tf.logging.set_verbosity(tf.logging.WARN)
parser = argparse.ArgumentParser()

parser.add_argument("--mode",             type=str,   default = "train", help="")
parser.add_argument("--batch_size",       type=int,   default = 4,       help="")
parser.add_argument("--load_mode",        type=str,   default = "RAM",   help="")
parser.add_argument("--num_threads",      type=int,   default = 16,      help="")
parser.add_argument("--max_iters",        type=int,   default = 100,     help="")
parser.add_argument("--gpus",             type=str,   default = "0",     help="")
parser.add_argument("--print_interval",   type=int,   default = 10,      help="")
parser.add_argument("--val_interval",     type=int,   default = 5,       help="")
parser.add_argument("--max_epoch_val",    type=int,   default = 10,      help="")
parser.add_argument("--learning_rate",    type=float, default = 1e-4,    help="")
parser.add_argument("--arch",             type=str,   default = "cnn",   help="")
parser.add_argument("--seed",             type=int,   default = 1007,    help="")
parser.add_argument("--exp_dir",          type=str,   default = "exps/first", help="")

args=parser.parse_args()

args.num_classes=10

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("logs.txt", "a", 1)
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

sys.stdout = Logger()


def train(args):

    tf.set_random_seed(args.seed)    
    os.makedirs(args.exp_dir, exist_ok=True)

    data_dic = data_loader.load_data(args)
    train_batch = data_dic["train_batch"]
    val_batch = data_dic["val_batch"]

    # with tf.Graph().as_default():
    #     global_step=tf.Variable(0, name='global_step', trainable=False)
    #     incr_global_step = tf.assign(global_step, global_step+1)
    #     optim = tf.train.AdamOptimizer(args.learning_rate, 0.9)

    #     num_gpus = len(args.gpus.split(","))
    #     batch_per_gpu = args.batch_size // num_gpus
    #     batch_splits = [batch_per_gpu for x in range(num_gpus)]
    #     batch_splits[-1] = batch_per_gpu if args.batch_size % num_gpus == 0 else args.batch_size % num_gpus
    #     print("batch_splits: %s"%(batch_splits))

    #     if args.arch == "cnn":
    #         TheModel = mnist_model.CnnModel
    #     else:
    #         TheModel = mnist_model.CnnLiteModel

    #     losses = []

    #     loader = data_loader.DataLoader(args)
    #     with tf.variable_scope(tf.get_variable_scope()):
    #         for i,(idx,n_batch) in enumerate(zip(args.gpus.split(","), batch_splits)):
    #             with tf.device('/gpu:%s'%(idx)):
    #                 with tf.name_scope('gpu%s'%(idx)):
    #                     img_placeholder, lbl_placeholder = loader.load_train_batch(n_batch)
    #                     model = TheModel(args, img_placeholder, lbl_placeholder)
    #                     losses.append(model.total_loss)
    #                     tf.get_variable_scope().reuse_variables()
    #             if i == 0:
    #                 vars_to_restore = 1 #TODO save selecting values
        
    #     loss = tf.stack(axis = 0, values = losses)
    #     loss = tf.reduce_mean(loss, 0)
    #     train_op = optim.minimize(loss)

    model = model_builder.ModelBuilder(args)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    
    coord=tf.train.Coordinator()
    

    #from tensorflow.python import debug as tf_debug

    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)


    #with tf_debug.LocalCLIDebugWrapperSession(tf.Session(config = config)) as sess:
    with tf.Session(config = config) as sess:
        sess.run(tf.global_variables_initializer())

        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        
        process = psutil.Process(os.getpid())

        sess.graph.finalize()

        for step in range(args.max_iters):
            t_load_start=time.time()
            data = sess.run(train_batch)
            # try:
            #     data = sess.run(train_batch)
            # except Exception as e:
            #     import traceback
            #     traceback.print_exc()
            #     print("ERRR",e)
            images, labels, image_names = data[0]
            print(images.shape, labels, image_names)
            memory = process.memory_percent()
            gt_classes_hot = np.eye(args.num_classes)[labels]

            t_train_start = time.time()
            loss, logits, pred = model.train_step(sess, images, gt_classes_hot)
            
            t_end = time.time()
            if step % args.val_interval == 0 and step != 0:
                validate(sess, model, val_batch, args.max_epoch_val)
            print("[%s] RAM:%4.2f%% loss: %.4f   total: %.4fs  load: %.4fs   train: %.4fs"%(step, memory, loss, t_end-t_load_start, t_train_start-t_load_start, t_end-t_train_start))
            print("pred:%s\ngt:  %s\n"%(pred, labels))

        coord.request_stop()
        coord.join(threads)

def validate(sess, model, val_batch, max_epoch_val):
    
    losses=[]
    preds=[]
    #logits=[]

    process = psutil.Process(os.getpid())

    for step in range(max_epoch_val):
        t_load_start = time.time()
        data = sess.run([val_batch])
        images, labels, image_names = data[0]
        memory = process.memory_percent()
        gt_classes_hot = np.eye(args.num_classes)[labels]
        
        t_valid_start=time.time()
        loss, logit, pred = model.validation_step(sess, images, gt_classes_hot)

        losses.append(loss)
        #logits.append(logit)
        preds.append(pred)
        
        t_end = time.time()
        print("[%s] RAM:%4.2f%% loss: %.4f   total: %.4fs  load: %.4fs   valid: %.4fs"%(step, memory, loss, t_end-t_load_start, t_valid_start-t_load_start, t_end-t_valid_start))
        print("pred:%s\ngt:  %s\n"%(pred, labels))
    
    loss_av = np.average(losses)
    acc_av = np.sum(pred==labels)/labels.shape[0]
    print("Av_Loss: %.4f   Precision: %.4f"%(loss_av, acc_av))

if __name__ == '__main__':
    #if args.mode=="train":
    #    train(args)
    #else:
    #    test(args)
    train(args)