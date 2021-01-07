import tensorflow as tf

def compute_loss(outputs, targets, method="l1"):
    if method == 'l1':
        return L1Loss(outputs, targets)
    elif method == 'l2':
        return L2Loss(outputs, targets)
    elif method == 'l1_l1grad':
        return L1Loss(outputs, targets) + L1GradLoss(outputs, targets)
    elif method == 'l2_l1grad':
        return L2Loss(outputs, targets) + L1GradLoss(outputs, targets)
    elif method == 'l1_l2grad':
        return L1Loss(outputs, targets) + L2GradLoss(outputs, targets)
    elif method == 'l2_l2grad':
        return L2Loss(outputs, targets) + L2GradLoss(outputs, targets)
    else:
        raise NotImplementedError("Unknown loss function: %s" % method)
    

def L1Loss(outputs, targets, weight = 1.0):
    with tf.name_scope("L1_loss"):
        diff = tf.abs(outputs - targets)
        return tf.reduce_mean(diff) * weight

def L2Loss(outputs, targets,  weight = 1.0):
    with tf.name_scope("L2_loss"):
        return tf.reduce_mean(tf.squared_difference(outputs, targets)) * weight

def L1GradLoss(outputs, targets, weight=1.0):
    with tf.name_scope("L1_gradient_loss"):
        dy_true, dx_true = tf.image.image_gradients(outputs)
        dy_pred, dx_pred = tf.image.image_gradients(targets)
        return L1Loss(dy_pred, dy_true, weight) + L1Loss(dx_pred, dx_true, weight)

def L2GradLoss(outputs, targets, weight=1.0):
    with tf.name_scope("L2_gradient_loss"):
        dy_true, dx_true = tf.image.image_gradients(outputs)
        dy_pred, dx_pred = tf.image.image_gradients(targets)
        return L2Loss(dy_pred, dy_true, weight) + L2Loss(dx_pred, dx_true, weight)

def BinaryLoss(outputs, targets):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=outputs, labels = targets))

