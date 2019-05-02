import os,sys,time
import tensorflow as tf
import numpy as np
import argparse
import psutil


import data_loader
import model_builder

tf.logging.set_verbosity(tf.logging.ERROR)
parser = argparse.ArgumentParser()

parser.add_argument("--mode",                type=str,   default = "train", help="")
parser.add_argument("--batch_size",          type=int,   default = 32,       help="")
parser.add_argument("--load_mode",           type=str,   default = "RAM",   help="")
parser.add_argument("--num_threads",         type=int,   default = 16,      help="")
parser.add_argument("--max_iters",           type=int,   default = 100,    help="")
parser.add_argument("--gpus",                type=str,   default = "0",     help="")
parser.add_argument("--print_interval",      type=int,   default = 10,     help="")
parser.add_argument("--print_interval_test", type=int,   default = 10,      help="")
parser.add_argument("--val_interval",        type=int,   default = 99,     help="")
parser.add_argument("--max_epoch_val",       type=int,   default = 50,     help="")
parser.add_argument("--learning_rate",       type=float, default = 1e-4,    help="")
parser.add_argument("--arch",                type=str,   default = "cnn",   help="")
parser.add_argument("--seed",                type=int,   default = 1007,    help="")
parser.add_argument("--exp_dir",             type=str,   default = "exps/first", help="")
parser.add_argument("--log_mode",            type=str,   default = 'a',     help="")

args=parser.parse_args()

args.num_classes=10

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("logs.txt", args.log_mode, 1)
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

sys.stdout = Logger()

report_train_load_time=[]
report_train_model_time=[]
report_train_total_time=[]
report_test_load_time=[]
report_test_model_time=[]
report_test_total_time=[]


def train(args):

    tf.set_random_seed(args.seed)    
    os.makedirs(args.exp_dir, exist_ok=True)

    data_dic = data_loader.load_data(args)
    train_batch = data_dic["train_batch"]
    val_batch = data_dic["val_batch"]

    model = model_builder.ModelBuilder(args)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    
    coord=tf.train.Coordinator()
    

    load_time_sum=0
    train_time_sum=0
    total_time_sum=0

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
            data = sess.run([train_batch])

            images, labels, image_names = data[0]
            labels=np.array([x[0] for x in labels])

            memory = process.memory_percent()
            gt_classes_hot = np.eye(args.num_classes)[labels]

            t_train_start = time.time()
            loss, logits, pred = model.train_step(sess, images, gt_classes_hot)
            
            t_end = time.time()
            
            load_time_sum  += t_train_start-t_load_start
            train_time_sum += t_end-t_train_start
            total_time_sum += t_end-t_load_start
            
            if step%args.print_interval==args.print_interval-1:
                print("[TRAIN %04d] RAM:%4.2f%% loss: %.4f   total: %.4fs  load: %.4fs   train: %.4fs"\
                    %(step, memory, loss, total_time_sum/args.print_interval, 
                    load_time_sum/args.print_interval, train_time_sum/args.print_interval))
                print("pred:%s\ngt:  %s"%(pred, labels))
                
                if step>args.print_interval:
                    report_train_load_time.append(load_time_sum/args.print_interval)
                    report_train_model_time.append(train_time_sum/args.print_interval)
                    report_train_total_time.append(total_time_sum/args.print_interval)
                load_time_sum=0
                train_time_sum=0
                total_time_sum=0
                

            if step % args.val_interval == 0 and step != 0:
                validate(sess, model, val_batch, args.max_epoch_val)

        coord.request_stop()
        coord.join(threads)

def validate(sess, model, val_batch, max_epoch_val):
    
    losses=[]
    preds=[]
    #logits=[]

    load_time_sum=0
    valid_time_sum=0
    total_time_sum=0

    process = psutil.Process(os.getpid())

    for step in range(max_epoch_val):
        t_load_start = time.time()
        data = sess.run([val_batch])
        images, labels, image_names = data[0]
        memory = process.memory_percent()

        labels=np.array([x[0] for x in labels])
        gt_classes_hot = np.eye(args.num_classes)[labels]
        
        t_valid_start=time.time()
        loss, logit, pred = model.validation_step(sess, images, gt_classes_hot)

        losses.append(loss)
        #logits.append(logit)
        preds.append(pred)
        
        t_end = time.time()

        load_time_sum  += t_valid_start-t_load_start
        valid_time_sum += t_end-t_valid_start
        total_time_sum += t_end-t_load_start

        if step%args.print_interval_test==args.print_interval_test-1:
            print("[TEST  %04d] RAM:%4.2f%% loss: %.4f   total: %.4fs  load: %.4fs   valid: %.4fs"%\
                (step, memory, loss, total_time_sum/args.print_interval_test, load_time_sum/args.print_interval_test, valid_time_sum/args.print_interval_test))
            print("pred:%s\ngt:  %s"%(pred, labels))
            if step>args.print_interval_test:
                report_test_load_time.append(load_time_sum/args.print_interval_test)
                report_test_model_time.append(valid_time_sum/args.print_interval_test)
                report_test_total_time.append(total_time_sum/args.print_interval_test)
            load_time_sum=0
            valid_time_sum=0
            total_time_sum=0
    
    loss_av = np.average(losses)
    acc_av = np.sum(pred==labels)/labels.shape[0]
    print("Av_Loss: %.4f   Precision: %.4f"%(loss_av, acc_av))

def mean(l):
    return sum(l)/len(l)

if __name__ == '__main__':

    train(args)
    print("TIMING(secs) BATCH:%3d GPU:%5s THREADS:%2d TRAIN: %.4f %.4f %.4f TEST: %.4f %.4f %.4f"%\
        (args.batch_size, args.gpus, args.num_threads,
        mean(report_train_total_time),
        mean(report_train_load_time),
        mean(report_train_model_time),
        mean(report_test_total_time),
        mean(report_test_load_time),
        mean(report_test_model_time)))