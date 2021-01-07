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

class TFRecordExporter:
    def __init__(self, tfrecord_dir, expected_images, print_progress=True, progress_interval=10, prefix=""):
        self.tfrecord_dir       = tfrecord_dir
        self.tfr_prefix         = os.path.join(self.tfrecord_dir, os.path.basename(self.tfrecord_dir)) + "_" + str(prefix)
        self.expected_images    = expected_images
        self.cur_images         = 0
        self.shape              = None
        
        self.tfr_writer = None
        self.print_progress     = print_progress
        self.progress_interval  = progress_interval

        if self.print_progress:
            print('Creating dataset "%s"' % tfrecord_dir)
        if not os.path.isdir(self.tfrecord_dir):
            os.makedirs(self.tfrecord_dir)
        assert os.path.isdir(self.tfrecord_dir)

    def close(self):
        if self.print_progress:
            print('%-40s\r' % 'Flushing data...', end='', flush=True)
        if self.tfr_writer is not None:
            self.tfr_writer.close()
        self.tfr_writer = None
        if self.print_progress:
            print('%-40s\r' % '', end='', flush=True)
            print('Added %d images.' % self.cur_images)


    def add_numpy(self, color_basis, uv, mask, light_index, view_index):
        if self.print_progress and self.cur_images % self.progress_interval == 0:
            print('%d / %d\r' % (self.cur_images, self.expected_images), end='', flush=True)
        if self.shape is None:
            self.shape = color_basis.shape
            assert self.shape[0] == self.shape[1]
            tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
            
            tfr_file  = self.tfr_prefix + '.tfrecords'
            self.tfr_writer = tf.python_io.TFRecordWriter(tfr_file, tfr_opt)
            
        assert(color_basis.dtype == np.float16)
        assert(uv.dtype == np.float32)
        assert(mask.dtype == np.uint8)

        view_index = np.dtype('int32').type(view_index)
        light_index = np.dtype('int32').type(light_index)

        ex = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=color_basis.shape)),
                "color_basis": tf.train.Feature(bytes_list=tf.train.BytesList(value=[color_basis.tostring()])),
                "uv": tf.train.Feature(bytes_list=tf.train.BytesList(value=[uv.tostring()])),
                "mask": tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask.tostring()])),
                'index': tf.train.Feature(bytes_list=tf.train.BytesList(value=[view_index.tostring()])),
                'light_index': tf.train.Feature(bytes_list=tf.train.BytesList(value=[light_index.tostring()]))
                }))
        self.tfr_writer.write(ex.SerializeToString())

        self.cur_images += 1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

def write_batch_arrs(buffer_lists, tfr, args, f_writer =None):
    if args.shuffle:
        random.shuffle(buffer_lists)

    for item in buffer_lists:
        color_basis, uv, mask, light_index, view_index = item
        tfr.add_numpy(color_basis,uv,mask, light_index, view_index)
        if f_writer is not None:
            f_writer.write("%d %d\n" % (light_index, view_index))
        
    return []


def load_mask_img(name):
    img = cv2.imread(name, -1)
    img = img[...,np.newaxis] 
    return img

def create_dataset(args):
    tfrecord_dir = args.outDir

    data_dir = args.dataDir
    
    
    if args.image_mask_dir is None:
        mask_data_dir = os.path.join(data_dir, args.mask_name)
        img_data_dir = os.path.join(data_dir, args.image_name)
    else:
        mask_data_dir = os.path.join(args.image_mask_dir, args.mask_name)
        img_data_dir = os.path.join(args.image_mask_dir, args.image_name)

    uv_data_dir = os.path.join(data_dir, args.uv_name)
    basis_data_dir = os.path.join(data_dir, args.basis_name)

    if args.indices is None:
        begin, end = args.begin, args.end
        print('Loading data from "%s", index range: [%d-%d]' % (data_dir, begin, end))
        indices = list(range(begin, end))
    else:
        indices = []
        with open(args.indices, 'r') as f:
            for line in f:
                lst = line.split()
                if len(lst) != 1:
                    continue
                indices.append(int(lst[0]))
        print('Loading data from "%s", index from: %s, number: %d' % (data_dir, args.indices, len(indices)))

    if args.shuffle:
        random.shuffle(indices)

    with open(os.path.join(args.outDir, 'shuffle_indices' + args.suffix +  '.txt'), 'w+') as f:
        for i in indices:
            f.write("%d\n" % i)


    dataset_size = len(indices)
    print("dataset_size: %d" % dataset_size)

    if dataset_size > 5000 and args.num_parts == 1:
        print("[Info] Too many data #%d. Split into 2 parts" % dataset_size)
        args.num_parts = 2
    
    if dataset_size % args.num_parts != 0:
        dataset_size -= dataset_size % args.num_parts

    memory = args.memory
    buffer_lists = []
    buffer_max_size = int ( float(memory) * 1024 / (512 * 512 * (2 * 4.0 + 18 * 2.0 + 1 * 1.0) / 1024.0/ 1024.0) )
  
    print("[Info] Buffer max size: %d" % buffer_max_size)

    count_warnings = 0
    assert(dataset_size % args.num_parts == 0)
    part_dataset_size = int(dataset_size / args.num_parts)

    f_writer = open(os.path.join(args.outDir,'log' + args.suffix +  '.txt'), 'w+')

    if args.dumpDir is not None:
        dump_basis_dir = os.path.join(args.dumpDir, "basis")
        dump_image_dir = os.path.join(args.dumpDir, "image")
        dump_mask_dir = os.path.join(args.dumpDir, "mask")
        dump_uv_dir = os.path.join(args.dumpDir, "UV")
        os.makedirs(dump_basis_dir, exist_ok=True)
        os.makedirs(dump_image_dir, exist_ok=True)
        os.makedirs(dump_mask_dir, exist_ok=True)
        os.makedirs(dump_uv_dir, exist_ok=True)

    for part_id in range(args.num_parts):
        part_indices = indices[part_id * part_dataset_size : (part_id + 1) * part_dataset_size]
        with TFRecordExporter(tfrecord_dir, part_dataset_size, prefix=str(part_id) + args.suffix) as tfr:
            for idx in part_indices: 
               
                if os.path.exists(os.path.join(img_data_dir, 'image_%d.npy' % idx)):
                    color = np.load( os.path.join(img_data_dir, 'image_%d.npy' % idx) )
                else:
                    if count_warnings < 10:
                        print("[Warning] No color.")
                    count_warnings += 1
                    color = np.zeros((512,512,3),dtype=np.float16)

                if os.path.exists(os.path.join(mask_data_dir, 'mask_%d.png' % idx)):
                    mask = load_mask_img(os.path.join(mask_data_dir, 'mask_%d.png' % idx) )
            
                else:
                    if count_warnings < 10:
                        print("[Warning No mask.]")
                    count_warnings += 1
                    mask = np.ones((512,512,1), dtype=np.uint8)*255

                uv = np.load( os.path.join(uv_data_dir, 'uv_%d.npy' % idx)) 

                basis = []
                for k in args.basis_lists:
                    if args.real and not os.path.exists(os.path.join(basis_data_dir, 'basis%d_%d.npy' % (k, idx))):
                        tmp_basis_data_dir = os.path.join(data_dir, "basis2")
                        basis.append(np.load(os.path.join(tmp_basis_data_dir, 'basis%d_%d.npy' % (k, idx))))
                    else:
                        basis.append(np.load(os.path.join(basis_data_dir, 'basis%d_%d.npy' % (k, idx))))
                basis = np.concatenate(basis, axis = -1)
                

                assert(color.dtype == np.float16) 
                assert(basis.dtype == np.float16) 
                assert(uv.dtype == np.float32)    
                assert(mask.dtype == np.uint8)    


                if args.dumpDir is not None:
                    np.save(os.path.join(dump_image_dir, "image_%d.npy" % idx), color)
                    np.save(os.path.join(dump_uv_dir, "uv_%d.npy" % idx), uv)
                    for kk in range(5):
                        np.save(os.path.join(dump_basis_dir, "basis%d_%d.npy" % (kk, idx)), basis[..., 3 * kk : 3 * (kk+1)])
                    cv2.imwrite(os.path.join(dump_mask_dir, "mask_%d.png" % idx), mask)


                color_basis = np.concatenate([color, basis], axis = -1)
                if args.rescale is not None:
                    color_basis *= args.rescale

                light_idx = -1
                view_idx = idx
                item = [color_basis, uv, mask, light_idx, view_idx]
                buffer_lists.append(item)

                if len(buffer_lists) >= buffer_max_size:
                    buffer_lists = write_batch_arrs(buffer_lists, tfr, args,f_writer)


            if len(buffer_lists) != 0:
                buffer_lists = write_batch_arrs(buffer_lists, tfr, args,f_writer)

    f_writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataDir', type=str,  required=True)
    parser.add_argument('--image_mask_dir', type=str,  default=None)

    parser.add_argument('--outDir', type=str,  required=True)
    parser.add_argument('--indices', type=str,  default=None)
    parser.add_argument('--suffix', type=str,  default="")
    parser.add_argument('--dumpDir', type=str,  default=None)

    parser.add_argument('--begin', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)

    parser.add_argument('--memory', type=int, default=None)   
    parser.add_argument("--basis_lists", type=str, default="0,1,2,3,4")
    parser.add_argument('--rescale', type=float, default=None)

    parser.add_argument("--shuffle", dest = "shuffle", action = "store_true")
    parser.add_argument("--no-shuffle", dest = "shuffle", action = "store_false")
    parser.set_defaults(shuffle=True)


    parser.add_argument('--image_name', type=str,  default='image')
    parser.add_argument('--mask_name', type=str,  default='mask')
    parser.add_argument('--basis_name', type=str,  default='basis')
    parser.add_argument('--uv_name', type=str,  default='UV')

    parser.add_argument('--num_parts', type=int,  default=1)


    args,unknown = parser.parse_known_args()

    if len(unknown) > 0:
        print(unknown)
        exit(-1)
    
    args.basis_lists = [int(i) for i in args.basis_lists.split(',')]
  
    if not os.path.exists(args.outDir):
        try:
            os.mkdir(args.outDir)
        except FileNotFoundError:
            print("[Warining] Create directory %s recursively." % args.outDir)
            os.makedirs(args.outDir)


    create_dataset(args)
   