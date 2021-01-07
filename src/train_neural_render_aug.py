import argparse
import os
import tensorflow as tf
import sys
from tqdm import tqdm

from IO_mapper import LogIOMapper
from util import set_random_seed, initial_logger, restore_model
from neural_render_aug import create_model
from data_min_size import load_data

parser = argparse.ArgumentParser()
parser.add_argument('--dataDir', type=str,  required=True)
parser.add_argument('--logDir', type=str,  required=True)

parser.add_argument('--summaryDir', type=str,  default=None)
parser.add_argument('--augDataDir', type=str,  default=None)
parser.add_argument('--checkpoint', type=str,  default=None)

parser.add_argument('--augRescale', type=float, default=0.32)
parser.add_argument('--rescale_input', type=float, default=1.0)
parser.add_argument('--rescale_output', type=float, default=1.0)



parser.add_argument('--start_step', type=int, default=0)
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
parser.add_argument("--loss", type=str, default="l1", choices=["l1", "l2", "l1_l1grad", "l2_l1grad", "l1_l2grad", "l2_l2grad"])


parser.add_argument('--texture_size', type=int, default=512)
parser.add_argument('--texture_channels', type=int, default=30)
parser.add_argument('--texture_levels', type=int, default=4)
parser.add_argument("--texture_init", type=str, default="glorot_uniform", choices=["normal", "glorot_uniform", "zeros", "ones","uniform"])
parser.add_argument("--mipmap", dest = "mipmap", action = "store_true")
parser.add_argument("--no-mipmap", dest = "mipmap", action = "store_false")
parser.set_defaults(mipmap=True)

parser.add_argument("--LDR", dest = "LDR", action = "store_true")
parser.add_argument("--no-LDR", dest = "LDR", action = "store_false")
parser.set_defaults(LDR=True)

parser.add_argument("--basis_lists", type=str, default="0,1,2,3,4")
parser.add_argument('--data_max_val', type=float, default=1.024)
parser.add_argument('--keep_max_val', type=float, default=1.0)

parser.add_argument("--activation", type=str, default="tanh", choices=["none", "tanh"])

parser.add_argument('--ngf', type=int,  default=64)
parser.add_argument('--resnet_res_count', type=int,  default=9)
parser.add_argument('--resnet_conv_count', type=int,  default=2)
parser.add_argument('--resnet_padding', type=str,  default="SYMMETRIC", choices=['REFLECT', 'CONSTANT', 'SYMMETRIC'])


parser.add_argument("--clip_grad", dest = "clip_grad", action = "store_true")
parser.add_argument("--no-clip_grad", dest = "clip_grad", action = "store_false")
parser.set_defaults(clip_grad=False)

args,unknown = parser.parse_known_args()

if len(unknown) != 0:
    print(unknown)
    exit(-1)



if __name__ == '__main__':
    # setting
    set_random_seed(args)
    logger = initial_logger(args)
    args.logger = logger

    args.basis_lists = [int(i) for i in args.basis_lists.split(',')]
    args.mapper = LogIOMapper(args.data_max_val)

    if args.augDataDir is not None:
        args.with_aug = True
    else:
        args.with_aug =  False


    # load data and preprocess
    dataset = load_data(args,rescale=args.rescale_input)

    if args.with_aug:
        aug_dataset = load_data(args, path = args.augDataDir, rescale=args.augRescale)
    else:
        aug_dataset = None


    # build network
    logger.info('------Build Network Structure------')
    model = create_model(dataset, aug_dataset, args)

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
    end_step = args.start_step + args.max_steps
    for step in tqdm(range(args.start_step, end_step), file=sys.stdout):
        def should(freq):
            return freq > 0 and (step % freq == 0 or step == end_step - 1)

        fetches = {
            'loss' : model.loss,
            'output' : model.output
        }

        if should(args.display_freq):
            fetches["summary"] = model.summary_op

        # update
        fetches["train_op"] = model.train_op
        if args.with_aug:
            fetches["aug_train_op"] = model.aug_train_op
     
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

