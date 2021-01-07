import argparse
import os
import sys
import numpy as np
import cv2
import tensorflow as tf

from tqdm import tqdm
from ops import resnet_cyclegan
from util import initial_logger, restore_model

parser = argparse.ArgumentParser()
parser.add_argument('--uvDir', type=str,  required=True)
parser.add_argument('--logDir', type=str,  required=True)

parser.add_argument('--imageDir', type=str,  default=None)

parser.add_argument('--checkpoint', type=str,  required=True)
parser.add_argument('--checkpoint_step', type=str,  default=None)

parser.add_argument('--begin', type=int, default=0)
parser.add_argument('--end', type=int, default=360)

args,unknown = parser.parse_known_args()

if len(unknown) != 0:
    print(unknown)
    exit(-1)


def create_test_model(uv, args):
    with tf.variable_scope("mask_branch"):
        reconstruct_mask = resnet_cyclegan(uv, output_channles = 1,  activation="sigmoid", prefix="mask", args=args)
    return reconstruct_mask


if __name__ == '__main__':
    # setting up logging
    logger = initial_logger(args, dump_code=False)
    args.logger = logger

    args.resolution = 512
    args.ngf = 32
    args.resnet_conv_count = 2
    args.resnet_res_count = 9
    args.resnet_padding = 'SYMMETRIC'
    args.batch_size = 1


    # build network
    logger.info('------Build Network Structure------')
    tf_uv = tf.placeholder(dtype=tf.float32, shape=[1, args.resolution, args.resolution, 2])
    output = create_test_model(tf_uv, args)

    
    # session initial
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    saver = tf.train.Saver(var_list=[var for var in tf.global_variables()])


    # initial variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    restore_model(sess, saver, args.checkpoint, args)

    logger.info('---Start testing...---')

    if args.imageDir is None:
        mask_log_dir = args.logDir
        apply_mask = False
    else:
        mask_log_dir = os.path.join(args.logDir, "mask")
        os.makedirs(mask_log_dir, exist_ok=True)
        apply_mask = True

    for step in tqdm(range(args.begin, args.end), file=sys.stdout):
        np_uv = np.load(os.path.join(args.uvDir, "uv_%d.npy" % step))
        np_uv = np_uv.astype(np.float32)
        np_uv = np_uv[np.newaxis, ...]
        out = sess.run(output, feed_dict = {tf_uv: np_uv})

        mask = out[0]

        if apply_mask:
            img = cv2.imread(os.path.join(args.imageDir, "output_%d.png" % step))
            img = img.astype(np.float32) / 255.0

            mask = np.clip(mask, 0.0, 1.0)
            out = img * mask
            cv2.imwrite(os.path.join(args.logDir, "output_%d.png" % step), out * 255)

        mask = np.clip(mask * 255.0, 0, 255)
        cv2.imwrite(os.path.join(mask_log_dir, "mask_%d.png" % step), mask)
