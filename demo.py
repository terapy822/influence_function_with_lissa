import tensorflow as tf
from sklearn import linear_model
import numpy as np

from utils import get_2class_mnist, visualize_result
from model import LogisticRegression as LR

EPOCH = 10
BATCH_SIZE = 100
NUM_A, NUM_B = 1, 7
TEST_INDEX = 5
WEIGHT_DECAY = 0.01
OUTPUT_DIR = 'result'
SAMPLE_NUM = 50 * 2

if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = get_2class_mnist(NUM_A, NUM_B)
    train_sample_num = len(x_train)

    # prepare sklearn model to train w
    C = 1.0 / (train_sample_num * WEIGHT_DECAY)
    sklearn_model = linear_model.LogisticRegression(C=C, solver='lbfgs', tol=1e-8, fit_intercept=False)

    # prepare tensorflow model to compute influence function
    tf_model = LR(tf.Session(), weight_decay=WEIGHT_DECAY)

    # train
    sklearn_model.fit(x_train, y_train.ravel())
    print('LBFGS training took %s iter.' % sklearn_model.n_iter_)

    # assign W into tensorflow model
    w_opt = sklearn_model.coef_.ravel()
    tf_model.sess.run(tf_model.w_assign_op, feed_dict={tf_model.w_ph: w_opt})

    # calculate original loss
    feed_dict_test = {tf_model.x: x_test[TEST_INDEX: TEST_INDEX+1], tf_model.y: y_test[TEST_INDEX: TEST_INDEX+1]}
    test_loss_ori = tf_model.sess.run(tf_model.loss, feed_dict=feed_dict_test)

    # get test loss gradient
    test_grad = tf_model.sess.run(tf_model.test_grad, feed_dict=feed_dict_test)

    # get inverse hvp (s_test)
    print('Calculating s_test ...')
    s_test = tf_model.get_inverse_hvp_lissa(test_grad, x_train, y_train, scale=10)
    # s_test = tf_model.sess.run(tf_model.inverse_hessian, feed_dict={tf_model.x: x_train, tf_model.y: y_train}) @ test_grad

    # get train loss gradient and estimate loss diff
    loss_diff_approx = np.zeros(train_sample_num)
    for i in range(train_sample_num):
        train_grad = tf_model.sess.run(tf_model.train_grad, feed_dict={tf_model.x: x_train[i: i+1],
                                                                       tf_model.y: y_train[i: i+1]})
        loss_diff_approx[i] = np.asscalar(train_grad.T @ s_test) / train_sample_num
        if i % 100 == 0:
            print('[{}/{}] Estimated loss diff: {}'.format(i+1, train_sample_num, loss_diff_approx[i]))

    # get high and low loss diff indice
    sorted_indice = np.argsort(loss_diff_approx)
    sample_indice = np.concatenate([sorted_indice[-int(SAMPLE_NUM/2):], sorted_indice[:int(SAMPLE_NUM/2)]])

    # calculate true loss diff
    loss_diff_true = np.zeros(SAMPLE_NUM)
    for i, index in zip(range(SAMPLE_NUM), sample_indice):
        print('[{}/{}]'.format(i+1, SAMPLE_NUM))

        # get minus one dataset
        x_train_minus_one = np.delete(x_train, index, axis=0)
        y_train_minus_one = np.delete(y_train, index, axis=0)

        # retrain
        C = 1.0 / ((train_sample_num - 1) * WEIGHT_DECAY)
        sklearn_model_minus_one = linear_model.LogisticRegression(C=C, fit_intercept=False, tol=1e-8, solver='lbfgs')
        sklearn_model_minus_one.fit(x_train_minus_one, y_train_minus_one.ravel())
        print('LBFGS training took {} iter.'.format(sklearn_model_minus_one.n_iter_))

        # assign w on tensorflow model
        w_retrain = sklearn_model_minus_one.coef_.T.ravel()
        tf_model.sess.run(tf_model.w_assign_op, feed_dict={tf_model.w_ph: w_retrain})

        # get retrain loss
        test_loss_retrain = tf_model.sess.run(tf_model.loss, feed_dict=feed_dict_test)

        # get true loss diff
        loss_diff_true[i] = test_loss_retrain - test_loss_ori

        print('Original loss       :{}'.format(test_loss_ori))
        print('Retrain loss        :{}'.format(test_loss_retrain))
        print('True loss diff      :{}'.format(loss_diff_true[i]))
        print('Estimated loss diff :{}'.format(loss_diff_approx[index]))

    visualize_result(loss_diff_true, loss_diff_approx[sample_indice])

