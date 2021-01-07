import argparse
import os
import numpy as np
import cv2
import tensorflow as tf
import sys
import collections
from tqdm import tqdm
import data_mask

from IO_mapper import LogIOMapper
from ops import resnet_cyclegan, GDSummaryWriter, create_train_op
from loss import BinaryLoss
from util import set_random_seed, initial_logger, restore_model

parser = argparse.ArgumentParser()
parser.add_argument('--dataDir', type=str,  required=True)
parser.add_argument('--logDir', type=str,  required=True)

parser.add_argument('--summaryDir', type=str,  default=None)

parser.add_argument('--checkpoint', type=str,  default=None)

parser.add_argument('--max_steps', type=int, default=50000)
parser.add_argument('--save_freq', type=int, default=10000)
parser.add_argument('--display_freq', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--seed', type=int, default=195829826)

parser.add_argument('--lr', type=float, default=0.0002)

parser.add_argument('--max_to_keep', type=int, default=100)
parser.add_argument("--shuffle", dest = "shuffle", action = "store_true")
parser.add_argument("--no-shuffle", dest = "shuffle", action = "store_false")
parser.set_defaults(shuffle=True)


args,unknown = parser.parse_known_args()

if len(unknown) != 0:
    print(unknown)
    exit(-1)


def create_model(dataset, args):
    Model = collections.namedtuple("Model", "train_op, summary_op, loss, vars, output")

    with tf.variable_scope("mask_branch"):
        reconstruct_mask = resnet_cyclegan(dataset.uv, output_channles = 1,  activation="sigmoid", prefix="mask", args=args)

    loss = BinaryLoss(reconstruct_mask, dataset.mask)
    
    tf_vars = tf.trainable_variables()

    train_op = create_train_op(args.lr, 0.9, 0.999, loss, tf_vars, "mask")

     # visualize
    summary_writer = GDSummaryWriter(args.batch_size)
    with tf.name_scope("tensorboard_visualize"):
        # scalar
        summary_writer.add_scalar("loss", loss)
    
        summary_writer.add_image("Mask_GT", dataset.mask, channels=1)
        summary_writer.add_image("Mask_recon", reconstruct_mask, channels=1)
        summary_writer.add_image("UV", dataset.uv, channels=2)

    summary_op = tf.summary.merge(summary_writer.lists)

    return Model(train_op = train_op,
        summary_op = summary_op,
        loss = loss,
        vars = tf_vars,
        output = reconstruct_mask)


if __name__ == '__main__':
    set_random_seed(args)

    # setting up logging
    logger = initial_logger(args)
    args.logger = logger
   
    args.ngf = 32
    args.resnet_conv_count = 2
    args.resnet_res_count = 9
    args.resnet_padding = 'SYMMETRIC'


    # load data and preprocess
    
    dataset = data_mask.load_data(args)

    # build network
    logger.info('------Build Network Structure------')
    model = create_model(dataset, args)

    # session initial
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)

    saver = tf.train.Saver(max_to_keep = args.max_to_keep, var_list=[var for var in tf.global_variables() if 'Adam' not in var.name and 'train_op' not in var.name], save_relative_paths=True)

    if args.summaryDir is None:
        args.summaryDir = args.logDir
    train_writer = tf.summary.FileWriter(args.summaryDir, sess.graph)

    # initial variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    if args.checkpoint is not None:
        restore_model(sess, saver, args.checkpoint)

    logger.info('---Start training...---')

    for step in tqdm(range(0, args.max_steps), file=sys.stdout):
        def should(freq):
            return freq > 0 and (step % freq == 0 or step == args.max_steps - 1)

        fetches = {
            'loss' : model.loss,
            'output' : model.output
        }

        if should(args.display_freq):
            fetches["summary"] = model.summary_op

    
        fetches["train_op"] = model.train_op
        
        results = sess.run(fetches)

        # display
        if should(args.display_freq):
            summary = results["summary"]
            _loss = results["loss"]

            train_writer.add_summary(summary, step)
            train_writer.flush()
            logger.info("Iter {:06d}, loss = {:04f}".format(step,_loss))


        if should(args.save_freq):
            saver.save(sess, args.logDir + r'/{}.tfmodel'.format(step))
            saver.save(sess, args.logDir + r'/latest_model.tfmodel')


    logger.info('---Finished Training.---')
    saver.save(sess, args.logDir + r'/{}.tfmodel'.format(step))
    saver.save(sess, args.logDir + r'/latest_model.tfmodel')

 