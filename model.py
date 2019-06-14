import tensorflow as tf
import numpy as np

from ops import log_clip

class LogisticRegression(object):
    def __init__(self, sess, weight_decay):
        self.sess = sess
        self.x, self.y, self.loss, self.hvp, self.train_grad, self.test_grad, self.w_assign_op, self.w_ph, self.u = self._build(weight_decay)

    def _build(self, wd):
        x = tf.placeholder(tf.float32, [None, 784], 'input_image')
        y = tf.placeholder(tf.float32, [None, 1], 'grand_truth')
        w = tf.Variable(tf.zeros([784]), name='w')
        logits = tf.matmul(x, tf.reshape(w, [-1, 1]))
        preds = tf.nn.sigmoid(logits)
        train_loss = -tf.reduce_mean(y * log_clip(preds) + (1 - y) * log_clip(1 - preds)) + tf.nn.l2_loss(w) * wd
        test_loss = -tf.reduce_mean(y * log_clip(preds) + (1 - y) * log_clip(1 - preds))

        w_ph = tf.placeholder(tf.float32, w.get_shape(), name='w_placeholder')
        w_assign_op = tf.assign(w, w_ph)

        # hessian vector product
        u = tf.placeholder(tf.float32, w.get_shape())
        first_grad = tf.gradients(train_loss, w)[0]
        elemwise_prod = first_grad * u
        hvp = tf.gradients(elemwise_prod, w)[0]

        # gradient
        train_grad = tf.gradients(train_loss, w)[0]
        test_grad = tf.gradients(test_loss, w)[0]

        return x, y, test_loss, hvp, train_grad, test_grad, w_assign_op, w_ph, u

    def get_inverse_hvp_lissa(self, v, x, y, scale=10, num_samples=5, recursion_depth=1000, print_iter=100):

        inverse_hvp = None

        for i in range(num_samples):
            print('Sample iteration [{}/{}]'.format(i+1, num_samples))
            cur_estimate = v
            permuted_indice = np.random.permutation(range(len(x)))

            for j in range(recursion_depth):

                x_sample = x[permuted_indice[j]:permuted_indice[j]+1]
                y_sample = y[permuted_indice[j]:permuted_indice[j]+1]

                # get hessian vector product
                hvp = self.sess.run(self.hvp, feed_dict={self.x: x_sample,
                                                         self.y: y_sample,
                                                         self.u: cur_estimate})

                # update hv
                cur_estimate = v + cur_estimate - hvp / scale

                if (j % print_iter == 0) or (j == recursion_depth - 1):
                    print("Recursion at depth {}: norm is {}".format(j, np.linalg.norm(cur_estimate)))

            if inverse_hvp is None:
                inverse_hvp = cur_estimate / scale
            else:
                inverse_hvp = inverse_hvp + cur_estimate / scale

        inverse_hvp = inverse_hvp / num_samples
        return inverse_hvp
