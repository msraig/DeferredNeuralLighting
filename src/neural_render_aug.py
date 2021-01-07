#!/usr/bin/env python
# -*- coding: utf-8 -*-

from neural_texture import create_neural_texture, sample_texture

import tensorflow as tf
import collections

from ops import resnet_cyclegan
from ops import GDSummaryWriter
from ops import quantize
from loss import compute_loss
from ops import compute_number_of_parameters
from ops import create_train_op


def select_k_basis_from_5(basis, choices):
    total_number = int (basis.get_shape().as_list()[-1] / 3)
    with tf.name_scope("select_k_basis_from_5"):
        basis_lst = tf.split(basis, total_number, axis = -1) # [B, H, W, 15] => 5 x [B, H, W, 3]
        res_basis_lists = []
        for idx in choices:
            res_basis_lists.append(basis_lst[idx])
        return tf.concat(res_basis_lists, axis = -1)


def neural_render(neural_texture, uv, index, _basis, args):
    # 0. select k basis from 5
    basis_count = len(args.basis_lists)
    basis = select_k_basis_from_5(_basis, args.basis_lists)

    # 1. neural render part
    with tf.variable_scope("neural_renderer"):
        with tf.variable_scope("sample_texture"):
            sampled_texture = sample_texture(neural_texture, uv, args)

        with tf.variable_scope("multipy_basis_and_sampled_texture"):
            assert(args.texture_channels % int(basis_count * 3) == 0)
           
            n_times = int(args.texture_channels / (basis_count * 3))
            reduced_basis = tf.concat([basis] * n_times, axis = -1, name="concat_more_times")

            mapped_reduced_basis = args.mapper.map_input(reduced_basis)
            mapped_sampled_texture = args.mapper.map_texture(sampled_texture)

            multiplied_texture = mapped_reduced_basis + mapped_sampled_texture
           

    # 2. neural renderer
    with tf.variable_scope("unet"):
        output = resnet_cyclegan(multiplied_texture, 3, args.activation, 'render', args)
        reconstruct = args.mapper.map_output(output)

    return sampled_texture, reduced_basis, multiplied_texture, output, reconstruct


Model = collections.namedtuple("Model", "train_op, aug_train_op, summary_op, loss, vars, output")

def create_model(dataset, aug_dataset, args):
    # define network
    with tf.variable_scope("OffsetNetwork"):
        with tf.variable_scope("create_neural_texture"):
            neural_texture = create_neural_texture(args)
           
        sampled_texture, reduced_basis, multiplied_texture, output, reconstruct = neural_render(neural_texture, dataset.uv,  dataset.index, dataset.basis, args)

    if args.with_aug:
        with tf.variable_scope("OffsetNetwork", reuse=True):
            aug_sampled_texture, aug_reduced_basis, aug_multiplied_texture, aug_output, aug_reconstruct = neural_render(neural_texture, aug_dataset.uv, aug_dataset.index, aug_dataset.basis, args)


    # loss and train_op
    loss = tf.zeros(shape=(), dtype=tf.float32)
    loss_aug = tf.zeros(shape=(), dtype=tf.float32)

    if args.LDR:
        target_image = quantize(dataset.color * dataset.mask, args.keep_max_val)
    else:
        target_image = dataset.color * dataset.mask

    target = args.mapper.map_input(target_image)

    loss += compute_loss(output, target, args.loss)

    if args.with_aug:
        aug_target = args.mapper.map_input(aug_dataset.color * aug_dataset.mask)
        loss_aug += compute_loss(aug_output, aug_target, args.loss)

 
    tf_vars = tf.trainable_variables()
    print("[info] Pameters: #%d, Variables: #%d" % (compute_number_of_parameters(tf_vars), len(tf_vars) ))
   

    train_op = create_train_op(args.lr, 0.9, 0.999, loss, tf_vars, "all", args)

    if args.with_aug:
        aug_train_op = create_train_op(args.lr, 0.9, 0.999, loss_aug, tf_vars, "aug_all", args)
    else:
        aug_train_op = None


    # visualize
    summary_writer = GDSummaryWriter(args.batch_size)
    with tf.name_scope("tensorboard_visualize"):
        # scalar
        summary_writer.add_scalar("loss", loss)
        if args.with_aug:
            summary_writer.add_scalar("loss(aug)", loss_aug)

        # image
        summary_writer.add_image("image(GT)", dataset.color * dataset.mask, rescale_factor=args.rescale_output)
        summary_writer.add_image("image(recon)", reconstruct, rescale_factor=args.rescale_output)
        summary_writer.add_image("sampled_texture", sampled_texture, channels=args.texture_channels)
        summary_writer.add_image("basis", reduced_basis, channels=args.texture_channels)
        summary_writer.add_image("multiplied_texture", multiplied_texture, channels=args.texture_channels)

        if args.with_aug:
            summary_writer.add_image("aug_image(GT)", aug_dataset.color * aug_dataset.mask, rescale_factor=args.rescale_output)
            summary_writer.add_image("aug_image(recon)", aug_reconstruct, rescale_factor=args.rescale_output)
            summary_writer.add_image("aug_sampled_texture", aug_sampled_texture, channels=args.texture_channels)
            summary_writer.add_image("aug_basis", aug_reduced_basis, channels=args.texture_channels)
            summary_writer.add_image("aug_multiplied_texture", aug_multiplied_texture, channels=args.texture_channels)

        for i in range(args.texture_levels):
            summary_writer.add_image('neural_texture_level_%d(0-2)' % i, tf.clip_by_value(neural_texture[i][...,:3][tf.newaxis, ...],0,1), channels=3, batch=1)

    summary_op = tf.summary.merge(summary_writer.lists)

    return Model(train_op = train_op,
        aug_train_op = aug_train_op,
        summary_op = summary_op,
        loss = loss,
        vars = tf_vars,
        output = reconstruct)

def create_test_model(dataset, args):
    basis = dataset.basis * args.rescale_input

    with tf.variable_scope("OffsetNetwork"):
        with tf.variable_scope("create_neural_texture"):
            neural_texture = create_neural_texture(args)
           
        sampled_texture, reduced_basis, multiplied_texture, output, reconstruct = neural_render(neural_texture, dataset.uv,  dataset.index, basis, args)
 
    return reconstruct

