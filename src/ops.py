#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import tensorflow as tf
import math


def resnet_cyclegan(inputs, output_channles, activation, prefix, args):
    def lrelu(x, leak=0.2, name="lrelu", alt_relu_impl=False):
        with tf.variable_scope(name):
            if alt_relu_impl:
                f1 = 0.5 * (1 + leak)
                f2 = 0.5 * (1 - leak)
                return f1 * x + f2 * abs(x)
            else:
                return tf.maximum(x, leak * x)

    def instance_norm(x):
        with tf.variable_scope("instance_norm"):
            epsilon = 1e-5
            mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
            scale = tf.get_variable('scale', [x.get_shape()[-1]],
                                    initializer=tf.truncated_normal_initializer(
                                        mean=1.0, stddev=0.02
            ))
            offset = tf.get_variable(
                'offset', [x.get_shape()[-1]],
                initializer=tf.constant_initializer(0.0)
            )
            out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset

            return out

    def general_conv2d(inputconv, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02,
                   padding="VALID", name="conv2d", do_norm=True, do_relu=True,
                   relufactor=0):
        with tf.variable_scope(name):
            conv = tf.contrib.layers.conv2d(
                inputconv, o_d, f_w, s_w, padding,
                activation_fn=None,
                weights_initializer=tf.truncated_normal_initializer(
                    stddev=stddev
                ),
                biases_initializer=tf.constant_initializer(0.0)
            )
            if do_norm:
                conv = instance_norm(conv)

            if do_relu:
                if(relufactor == 0):
                    conv = tf.nn.relu(conv, "relu")
                else:
                    conv = lrelu(conv, relufactor, "lrelu")

            return conv

    def general_deconv2d(inputconv, outshape, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1,
                     stddev=0.02, padding="VALID", name="deconv2d",
                     do_norm=True, do_relu=True, relufactor=0):
        with tf.variable_scope(name):

            conv = tf.contrib.layers.conv2d_transpose(
                inputconv, o_d, [f_h, f_w],
                [s_h, s_w], padding,
                activation_fn=None,
                weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                biases_initializer=tf.constant_initializer(0.0)
            )

            if do_norm:
                conv = instance_norm(conv)

            if do_relu:
                if(relufactor == 0):
                    conv = tf.nn.relu(conv, "relu")
                else:
                    conv = lrelu(conv, relufactor, "lrelu")

            return conv

    def build_resnet_block(inputres, dim, name="resnet", padding="REFLECT"):
        with tf.variable_scope(name):
            out_res = tf.pad(inputres, [[0, 0], [1, 1], [
                1, 1], [0, 0]], padding)
            out_res = general_conv2d(
                out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c1")
            out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
            out_res = general_conv2d(
                out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c2", do_relu=False)

            return tf.nn.relu(out_res + inputres)

    ngf = args.ngf
    n_downsampling = args.resnet_conv_count
    n_resblock = args.resnet_res_count

    with tf.variable_scope("unet_cyclegan_%s" % prefix):
        f = 7
        ks = 3
        padding = args.resnet_padding

        pad_input = tf.pad(inputs, [[0, 0], [ks, ks], [
                ks, ks], [0, 0]], padding)

        # first 1x1 conv
        o_c1 = general_conv2d(
                pad_input, ngf, f, f, 1, 1, 0.02, name="c1")

        layers = [o_c1]

        # down sampling
        for i in range(n_downsampling):
            mult = int(2 ** i)
            o_c_tmp = general_conv2d(
                layers[-1], ngf * mult * 2, ks, ks, 2, 2, 0.02, "SAME", "c%d" % (i+2) )

            layers.append(o_c_tmp)


        # res-block
        res_mul = int(2 ** (n_downsampling - 1))
        for i in range(n_resblock):
            o_r_tmp = build_resnet_block(layers[-1], ngf * 2 * res_mul, "r%d" % (i+1), padding)
            layers.append(o_r_tmp)

        # up sampling
        for i in range(n_downsampling):
            mult = int(2 ** (n_downsampling - i))
            o_c_tmp = general_deconv2d(
            layers[-1], [args.batch_size, int(512/mult), int(512/mult), int(ngf * mult / 2)], int(ngf * mult / 2), ks, ks, 2, 2, 0.02,
            "SAME", "c%d" % (i + 2 + n_downsampling))
            layers.append(o_c_tmp)

        # last 1x1 conv
        o_c6 = general_conv2d(layers[-1], output_channles, f, f, 1, 1,
                                    0.02, "SAME", "c%d" % (2 + n_downsampling * 2),
                                    do_norm=False, do_relu=False)

        if activation == "tanh":
            out_gen = tf.tanh(o_c6, "t1") # [-1,1]
        elif activation == 'sigmoid':
            out_gen = tf.sigmoid(o_c6)
        elif activation == 'none':
            out_gen = o_c6

        return out_gen

def avg_downsample(img):
    shape_len = len(img.get_shape().as_list())
    if shape_len == 3:
        no_batch = True
        batch_img = img[tf.newaxis, ...]
    elif shape_len == 4:
        no_batch = False
        batch_img = img

    channels = batch_img.get_shape().as_list()[-1]
    np_kernel = np.array((1.0, 1.0), dtype=np.float32)
    np_kernel = np.outer(np_kernel, np_kernel)
    np_kernel /= np.sum(np_kernel)

    np_kernel = np_kernel[:, :, np.newaxis, np.newaxis]
    kernel = tf.constant(np_kernel, dtype=tf.float32) * tf.eye(channels, dtype=tf.float32)
    batch_result =  tf.nn.conv2d(input=batch_img, filter=kernel, strides=[1, 2, 2, 1], padding="SAME")

    if no_batch:
        return batch_result[0]
    else:
        return batch_result

def create_train_op(lr, beta1, beta2, loss, vars, prefix="", args=None):
    if len(vars) == 0:
        return tf.no_op()

    with tf.name_scope("solver_op_%s" % prefix):
        solver = tf.train.AdamOptimizer(learning_rate=lr, beta1 = beta1, beta2 = beta2)
        grad = solver.compute_gradients(loss, vars)
        if args is not None and args.clip_grad:
            clipped_grad = [(tf.clip_by_value(g, -1., 1.), var) for g, var in grad]
            grad = clipped_grad

    with tf.name_scope("train_op_%s" % prefix):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = solver.apply_gradients(grad)
        return train_op

def compute_number_of_parameters(var_lists=tf.trainable_variables(), verbose=False):
    total_parameters = 0
    for variable in var_lists:
        shape = variable.get_shape()
        
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value

        if verbose:
            print("%s: %d" % (variable.name, variable_parameters))

        total_parameters += variable_parameters
    
    return total_parameters

def quantize(img, keep_max_val):
    with tf.name_scope("quantize"):
        # 1. rescale [0,M] to [0,M/K]
        K = keep_max_val
        img = img / K

        # 2. clamp [0, M/K] to [0,1]
        img = tf.clip_by_value(img, 0.0, 1.0)

        # 2. convert linear [0,1] to srgb color space [0,1]
        img = img ** (1.0/2.2)

        # 3. quantize to [0, 255] uint8
        img = tf.cast(img * 255, tf.uint8)

        # 4. convert back to float32, [0,1]
        img = tf.cast(img, tf.float32) / 255.0

        # 5. convert back to linear color space [0,1]
        img = img ** (2.2)

        # 6. convert back to [0, K]
        img = img * K

        return img

def to_uint8(img):
    with tf.name_scope('to_uint8'):
        img = tf.clip_by_value(img * 255.0, 0.0, 255.0)
        img_out = tf.cast(img, tf.uint8)
        return img_out

def to_srgb_uint8(img):
    with tf.name_scope('to_srgb_uint8'):
        img_out = img ** (1.0 / 2.2)
        img_out = tf.cast(tf.minimum(255.0, img_out * 255), tf.uint8)
        return img_out

def to_srgb(img):
    with tf.name_scope('to_srgb'):
        return img ** (1.0/2.2)

def to_linear(img):
    with tf.name_scope('to_linear'):
        return img ** (2.2)

def tileBatch(input_batch, nCol):
	input_shape = input_batch.get_shape().as_list()
	nImg = input_shape[0]
	nRow = (nImg - 1) // nCol + 1
	output_rows = []
	for r in range(0, nRow):
		output_row = []
		for c in range(0, nCol):
			output_row.append(input_batch[r*nCol+c,:,:,:])
		output_rows.append(tf.concat(output_row, axis=1))
	output = tf.concat(output_rows, axis=0)
	return output[tf.newaxis,...]

class GDSummaryWriter():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.nCol_dict = {
            1024:32,512:32,256:16,128:16,100:10,96:12,32:4,64:8,48:6,24:6,16:4,12:4,8:4,4:2,2:1,1:1,10:5, 20:5, 25:5, 15:3, 5:1
        }
        self.lists = []

    def add_scalar(self, name, value):
        self.lists.append(tf.summary.scalar(name, value))

    def add_image(self, name, image, rescale_factor=1.0, minmax_norm=False, linear=False, channels=None, batch = None):
        if image is None:
            print("[Warning] `%s` is None" % name)
            return 

        if batch is None:
            batch_size = self.batch_size
        else:
            batch_size = batch
        
        if channels is None:
            channels = image.get_shape().as_list()[-1]
            if channels is None:
                raise ValueError("Invalid channels of Tensor [%s] " % image.name)

        if channels > 3:
            if channels % 3 != 0:
                total_channels = (int(channels / 3) + 1 ) * 3
                image = tf.concat([image, tf.zeros_like(image[...,0:(total_channels-channels)])], axis=-1)
            else:
                total_channels = channels

            _split_image = tf.split(image, int(total_channels/3), axis=-1) # K x [B, H, W, 3]
            image = tf.concat(_split_image, axis=0) # [B * K, H, W, 3]
        elif channels < 3:
            total_channels = 3
            if channels == 1:
                image = tf.concat([image, image, image], axis=-1)
            elif channels == 2:
                image = tf.concat([image, tf.zeros_like(image[...,0:1])], axis=-1)


        color_batch = tf.concat(image * rescale_factor, axis=0)

        if minmax_norm:
            color_batch =  (color_batch - tf.reduce_min(color_batch)) / (tf.reduce_max(color_batch) - tf.reduce_min(color_batch))

        if linear:
            color_tiled = to_uint8(tileBatch(tf.clip_by_value(color_batch,0,1), self.nCol_dict[batch_size]))
        else:
            color_tiled = to_srgb_uint8(tileBatch(tf.clip_by_value(color_batch,0,1), self.nCol_dict[batch_size]))

        self.lists.append(tf.summary.image(name, color_tiled))
