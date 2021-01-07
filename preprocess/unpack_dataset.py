import os
import glob
import numpy as np
import tensorflow as tf
import sys
import argparse
import cv2
import random
from tqdm import tqdm


def error(msg):
    print('Error: ' + msg)
    exit(1)


def write_batch_arrs(buffer_lists, args, f_writer =None):
    for item in buffer_lists:
        color, uv, mask, basis,  view_index = item
        
        np.save(os.path.join(args.outDir, "image", "image_%d.npy" % view_index), color)
        np.save(os.path.join(args.outDir, "UV", "uv_%d.npy" % view_index), uv)
        cv2.imwrite(os.path.join(args.outDir, "mask", "mask_%d.png" % view_index), mask)

        basis_count = 5

        basis_lists = np.split(basis, 5, axis = -1)
        for i in range(basis_count):
            np.save(os.path.join(args.outDir, "basis", "basis%d_%d.npy" % (i,view_index)), basis_lists[i])


        if f_writer is not None:
            f_writer.write("%d\n" % (view_index))
        
    return []
    
def parse_tfrecord_tf_light_index(record, args):
    features = tf.parse_single_example(record, features={
        'shape': tf.FixedLenFeature([3], tf.int64),
        'color_basis': tf.FixedLenFeature([], tf.string),
        'uv': tf.FixedLenFeature([], tf.string),
        'mask': tf.FixedLenFeature([], tf.string),
        'index': tf.FixedLenFeature([], tf.string)
        })
    
    shape_color_basis = features["shape"]
    shape_uv = tf.convert_to_tensor([shape_color_basis[0],shape_color_basis[1], 2], dtype=tf.int64)
    shape_mask = tf.convert_to_tensor([shape_color_basis[0],shape_color_basis[1], 1], dtype=tf.int64)
   
    color_basis = tf.decode_raw(features['color_basis'], tf.float16)
    uv = tf.decode_raw(features['uv'], tf.float32)
    mask = tf.decode_raw(features['mask'], tf.uint8)

    idx = tf.decode_raw(features['index'], tf.int32)
    
    return tf.reshape(color_basis, shape_color_basis), tf.reshape(uv, shape_uv), tf.reshape(mask, shape_mask), idx



def load_data(data_dir):
    print("Load data from %s" % data_dir)
    batch_size = 1

    tfr_files = sorted(glob.glob(os.path.join(data_dir, '*.tfrecords')))
    assert(len(tfr_files) >= 1)

    dataset = tf.data.TFRecordDataset(filenames=tfr_files)
    dataset = dataset.map(lambda x : parse_tfrecord_tf_light_index(x, args), num_parallel_calls=16)
 
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size = batch_size)

    iterator = dataset.make_one_shot_iterator()

    batch_color_basis, batch_uv, batch_mask, batch_indices = iterator.get_next()

   
    return batch_color_basis, batch_uv, batch_mask, batch_indices
    
def create_dataset(args):

    batch_color_basis, batch_uv, batch_mask, batch_indices = load_data(args.dataDir)

    dataset_size = args.count

    batch_color = batch_color_basis[...,0:3]
    batch_basis = batch_color_basis[...,3:]
  
    memory = args.memory
    buffer_max_size = int ( float(memory) * 1024 / (512 * 512 * (2 * 4.0 + 18 * 2.0 + 1 * 1.0) / 1024.0/ 1024.0) )
    
    print("[Info] Buffer max size: %d" % buffer_max_size)


    fetches = {
        "color" :batch_color[0],
        "basis": batch_basis[0],
        "uv": batch_uv[0],
        "mask": batch_mask[0],
        "index" : batch_indices[0],
    }
   
    buffer_lists = []
   
    f_writer = open(os.path.join(args.outDir, 'log.txt'), 'w+')
    
    os.makedirs(os.path.join(args.outDir, "image"), exist_ok=True)
    os.makedirs(os.path.join(args.outDir, "basis"), exist_ok=True)
    os.makedirs(os.path.join(args.outDir, "mask"), exist_ok=True)
    os.makedirs(os.path.join(args.outDir, "UV"), exist_ok=True)
    
    with tf.Session() as sess:
        for idx in range(args.count):
            try:
                results = sess.run(fetches)
            except tf.errors.OutOfRangeError:
                break
                    
            view_idx = results["index"]
                  
            color = results["color"]
            uv = results["uv"]
            mask = results["mask"]
            basis = results["basis"]
          

            assert(color.dtype == np.float16) 
            assert(uv.dtype == np.float32)
            assert(mask.dtype == np.uint8)
            assert(basis.dtype == np.float16)

            item = [color, uv, mask, basis, view_idx]
            buffer_lists.append(item)

            if len(buffer_lists) >= buffer_max_size:
                buffer_lists = write_batch_arrs(buffer_lists, args, f_writer)


        if len(buffer_lists) != 0:
            buffer_lists = write_batch_arrs(buffer_lists, args, f_writer)

    f_writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataDir', type=str,  required=True)
    parser.add_argument('--outDir', type=str,  required=True)

    parser.add_argument('--count', type=int,  required=True)
  
    parser.add_argument('--memory', type=int, default=4) # 8GB    

    args,unknown = parser.parse_known_args()

    if len(unknown) > 0:
        print(unknown)
        exit(-1)
    
   
    if not os.path.exists(args.outDir):
        try:
            os.mkdir(args.outDir)
        except FileNotFoundError:
            print("[Warining] Create directory %s recursively." % args.outDir)
            os.makedirs(args.outDir)

    create_dataset(args)
   