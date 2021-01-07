import random
import numpy as np
import tensorflow as tf
import os
import logging
import json
import glob
import zipfile

def set_random_seed(args):
    if args.seed == None:
        args.seed = random.randint(0, 2 ** 31 - 1)
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    print("random seed: " + str(args.seed))

def get_var_list(name_list):
    layer_vars = []
    for name in name_list:
        layer_vars.append([var for var in tf.global_variables() if name in var.name])
    layer_vars = [var for layer in layer_vars for var in layer]
    return layer_vars

def initial_logger(args, dump_code=True):
    if not os.path.exists(args.logDir):
        os.makedirs(args.logDir)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(args.logDir + r'/training_log.txt')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)

    if dump_code:
        logger.info("Dump current source code...")
        current_path = os.path.dirname(os.path.abspath(__file__))
        dump_files = glob.glob(os.path.join(current_path, '*.py'))
        dump_files = dump_files + glob.glob(os.path.join(os.path.join(os.path.dirname(current_path),"preprocess"), '*.py'))
        zf = zipfile.ZipFile(os.path.join(args.logDir, 'source_code.zip'), mode='w')
        try:
            for f in dump_files:
                zf.write(f)
        finally:
            zf.close()

    logger.info('----Settings:-------')
    args_train = vars(args)
    for key in args_train:
        logger.info('---{}:{}----'.format(key, args_train[key]))
    with open(args.logDir + r'/params.json', 'w') as f:
        json.dump(args_train, f)

    return logger

def restore_model(sess, saver, checkpoint, args):
    if checkpoint is not None:
        args.logger.info('Restore {}'.format(checkpoint))
        if args.checkpoint_step is not None:
            ckpt = os.path.join(checkpoint, "%s.tfmodel" % args.checkpoint_step)
        else:
            ckpt = tf.train.latest_checkpoint(checkpoint)
        print('Restore {}.'.format(ckpt))
        saver.restore(sess, ckpt)
        return True
    else:
        args.logger.info('[ERROR] Restore failed {}.'.format(checkpoint))
        return False