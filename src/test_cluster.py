import argparse
import os
import numpy as np
import cv2
import tensorflow as tf
import sys
from tqdm import tqdm
import collections

from IO_mapper import LogIOMapper
from neural_render_aug import create_test_model
from util import set_random_seed, initial_logger, restore_model


parser = argparse.ArgumentParser()
parser.add_argument('--dataDir', type=str,  required=True)
parser.add_argument('--logDir', type=str,  required=True)
parser.add_argument('--indexDir', type=str,  required=True)
parser.add_argument('--real_indices', type=str,  default=None)


parser.add_argument('--checkpoint', type=str,  default=None)
parser.add_argument('--checkpoint_step', type=str,  default=None)

parser.add_argument('--max_steps', type=int, default=50000)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--seed', type=int, default=195829826)


parser.add_argument("--shuffle", dest = "shuffle", action = "store_true")
parser.add_argument("--no-shuffle", dest = "shuffle", action = "store_false")
parser.set_defaults(shuffle=False)

parser.add_argument('--rescale_output', type=float, default=1.0)
parser.add_argument('--rescale_input', type=float, default=1.0)

parser.add_argument('--texture_size', type=int, default=512)
parser.add_argument('--texture_channels', type=int, default=30)
parser.add_argument('--texture_levels', type=int, default=4)
parser.add_argument("--texture_init", type=str, default="glorot_uniform", choices=["normal", "glorot_uniform", "zeros", "ones","uniform"])

parser.add_argument("--mipmap", dest = "mipmap", action = "store_true")
parser.add_argument("--no-mipmap", dest = "mipmap", action = "store_false")
parser.set_defaults(mipmap=True)

parser.add_argument("--basis_lists", type=str, default="0,1,2,3,4")
parser.add_argument('--data_max_val', type=float, default=None)
parser.add_argument("--activation", type=str, default="tanh", choices=["none", "tanh"])

parser.add_argument("--gamma", dest = "gamma", action = "store_true")
parser.add_argument("--no-gamma", dest = "gamma", action = "store_false")
parser.set_defaults(gamma=True)

parser.add_argument("--HDR", dest = "HDR", action = "store_true")
parser.add_argument("--no-HDR", dest = "HDR", action = "store_false")
parser.set_defaults(HDR=False)



parser.add_argument('--ngf', type=int,  default=64)
parser.add_argument('--resnet_res_count', type=int,  default=9)
parser.add_argument('--resnet_conv_count', type=int,  default=2)
parser.add_argument('--resnet_padding', type=str,  default="SYMMETRIC", choices=['REFLECT', 'CONSTANT', 'SYMMETRIC'])


args,unknown = parser.parse_known_args()

if len(unknown) != 0:
    print(unknown)
    exit(-1)


def load_data_from_file(idx):
    uv = np.load(os.path.join(args.dataDir, "UV", "uv_%d.npy" % idx))
    uv = uv.astype(np.float32)

    basis = []
    for i in range(5):
        t = np.load(os.path.join(args.dataDir, "basis", "basis%d_%d.npy" % (i, idx)))
        basis.append(t)
    basis = np.concatenate(basis, axis = -1)
    basis = basis.astype(np.float32) # [512, 512, 15]


    batch_uv = uv[np.newaxis, ...]
    batch_basis = basis[np.newaxis, ...]
    
    return batch_uv, batch_basis


def test_once(vertex_idx, uv, basis, output, saver, sess):
    fetches = {
        "output" : output
    }

    test_indices = args.INVERSE_NEARBY_INDICES[vertex_idx]

    restore_model(sess, saver, os.path.join(args.checkpoint, "log_cluster_%d" % vertex_idx), args)

    for test_idx in tqdm(test_indices):
        if args.real_indices is not None:
            _uv, _basis = load_data_from_file(args.test_indices_mapper[test_idx])
        else:
            _uv, _basis = load_data_from_file(test_idx)
        
      
        img = sess.run(fetches, feed_dict = {uv: _uv, basis : _basis})["output"]
     
        img = img[0]
        
        weight = None
        for i in range(3):
            if vertex_idx == args.NEARBY_INDICES[test_idx][i]:
                weight = args.WEIGHTS[test_idx][i]
                break
        
        if test_idx not in args.IMGS:
            args.IMGS[test_idx] = img * weight
        else:
            args.IMGS[test_idx] += img * weight


def load_indices(args):
    # NEARBY_INDICES: test_view_idx ==> [v0, v1, v2]
    # INVERSE_NEARBY_INDICES vertex_idx ==> lst of test_view_idx
   
    args.NEARBY_INDICES = []
    args.WEIGHTS = []

    args.INVERSE_NEARBY_INDICES = {}
    args.IMGS = {}


    if args.real_indices is not None:
        real_indices = []
        with open(args.real_indices, "r") as f:
            for line in f:
                real_indices.append(int(line.split()[0]))
        print("[Info] Load Real_indices." ) 
        args.test_indices_mapper = real_indices

    with open(args.indexDir, "r") as f:
        idx = 0
        for line in f:
            lst = line.split()
            v1, v2, v3 = [int(i) for i in lst[:3]]
            w2, w3 = [float(i) for i in lst[3:5]]
            w1 = 1.0 - w2 - w3

            args.NEARBY_INDICES.append([v1, v2, v3])
            args.WEIGHTS.append([w1, w2, w3])

            if v1 not in args.INVERSE_NEARBY_INDICES:
                args.INVERSE_NEARBY_INDICES[v1] = [idx]
            else:
                args.INVERSE_NEARBY_INDICES[v1].append(idx)
            
            if v2 not in args.INVERSE_NEARBY_INDICES:
                args.INVERSE_NEARBY_INDICES[v2] = [idx]
            else:
                args.INVERSE_NEARBY_INDICES[v2].append(idx)

            if v3 not in args.INVERSE_NEARBY_INDICES:
                args.INVERSE_NEARBY_INDICES[v3] = [idx]
            else:
                args.INVERSE_NEARBY_INDICES[v3].append(idx)

            idx += 1

Dataset = collections.namedtuple("Dataset", "iterator, color, uv, mask, basis, index")

if __name__ == '__main__':
    set_random_seed(args)

    logger = initial_logger(args, dump_code=False)
    args.logger = logger
    
    args.basis_lists = [int(i) for i in args.basis_lists.split(',')]
    args.mapper = LogIOMapper(args.data_max_val)

    load_indices(args)
    

    # define graph
    uv = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, 512, 512, 2])
    basis = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, 512, 512, 15])
  
    dataset = Dataset(uv=uv, basis=basis, iterator= None, color = None, mask =None,index=None)
    output = create_test_model(dataset, args)
    output *= args.rescale_output

    saver = tf.train.Saver(var_list=[var for var in tf.global_variables()]) 
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    sess = tf.Session(graph=tf.get_default_graph(), config=config)

    count = 0
    for i in args.INVERSE_NEARBY_INDICES:
        test_once(i, uv, basis, output, saver, sess)
        count += 1


    for idx in tqdm(args.IMGS):
        final_img = args.IMGS[idx]
        if args.real_indices is not None:
            final_idx = args.test_indices_mapper[idx]
        else:
            final_idx = idx

        if args.HDR:
            np.save(os.path.join(args.logDir, "output_%d.npy" % final_idx), final_img)
        else:
            final_img = final_img[...,::-1]
            final_img = np.clip(final_img, 0, 1)

            if args.gamma:
                final_img = final_img ** (1.0/2.2)

            final_img = np.clip(final_img * 255.0, 0, 255)
            cv2.imwrite(os.path.join(args.logDir, "output_%d.png" % final_idx), final_img)
