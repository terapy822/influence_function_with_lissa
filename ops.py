import tensorflow as tf


def log_clip(x):
    return tf.log(tf.clip_by_value(x, 1e-10, x))