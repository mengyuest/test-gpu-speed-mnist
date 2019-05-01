import gzip
import matplotlib.pyplot as plt
import numpy as np
import time
import os

image_size=28
dataset_dir="data/"
t1=time.time()


def data_preproc(split, num_samples, input_path, save_name, mode):
    f=gzip.open(input_path)
    
    if mode=="image":
        offset=16
        buf_size = image_size * image_size
    else:
        offset=8
        buf_size = 1
    f.read(offset)
    buf=f.read(buf_size * num_samples)
    data=np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    if mode=="image":
        data=data.reshape(num_samples, image_size, image_size, 1)
    else:
        data=data.reshape(num_samples, 1)
    np.save(save_name, data)
    os.makedirs(dataset_dir+split, exist_ok=True)
    if mode=="image":
        for i in range(num_samples):
            np.save(dataset_dir+split+"/%05d.npy"%(i), data[i])
    else:
        with open(dataset_dir+split+"/list.txt", "w") as f:
            for i in range(num_samples):
                f.write("%d\n"%(data[i,0]))

task_list=[("train", 60000,  "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "train-images.npy", "train-labels.npy"),
           ("test",  10000,  "t10k-images-idx3-ubyte.gz",  "t10k-labels-idx1-ubyte.gz",  "test-images.npy",  "test-labels.npy")]

for name, num_samples, img_path, lbl_path, img_output, lbl_output in task_list:
    data_preproc(name, num_samples, dataset_dir+img_path, dataset_dir+img_output, "image")
    data_preproc(name, num_samples, dataset_dir+lbl_path, dataset_dir+lbl_output, "label")

print("finished in %.4f secs"%(time.time()-t1))
