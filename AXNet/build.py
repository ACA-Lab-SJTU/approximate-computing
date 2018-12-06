import os

import numpy as np
import tensorflow as tf
from utilities import load_data, draw


class Model(object):
    def __init__(self, sess, benchmark, a_net, c_net, error_bound, error_func, learning_rate):
        assert sum(a_net[1:-1]) == c_net[-1] - 2

        self.a_net = a_net
        self.c_net = c_net
        self.error_func = error_func
        self.learning_rate = learning_rate
        self.sess = sess
        self.error_bound = error_bound
        self.benchmark = benchmark
        X0, Y0, X1, Y1 = load_data(benchmark)

        self.train_data = np.hstack((X0, Y0))
        self.test_data = np.hstack((X1, Y1))
        np.random.shuffle(self.train_data)
        np.random.shuffle(self.test_data)

        self.result_dir = self._make_dir('/result/')
        self.weight_dir = self.result_dir
        with tf.variable_scope('data'):
            self.inputs = tf.placeholder(tf.float32, shape=[None, a_net[0]], name='inputs')
            self.label = tf.placeholder(tf.float32, shape=[None, a_net[-1]], name='label')
            self.clf_label = tf.placeholder(tf.float32, shape=[None, 2])

        with tf.variable_scope('clf'):
            clayer = self.inputs
            for i in range(1, len(c_net) - 1):
                clayer = tf.layers.dense(inputs=clayer, units=c_net[i], activation=tf.nn.relu,
                                         kernel_initializer=tf.random_normal_initializer(0.0, 0.1))
            clayer = tf.layers.dense(inputs=clayer, units=c_net[-1],
                                     kernel_initializer=tf.random_normal_initializer(0.0, 0.1))
            split_layers = tf.split(clayer, a_net[1:-1] + [2], axis=1)
            print(split_layers)
            self.split_layers = split_layers[:-1]
            self.cout = split_layers[-1]
            self.clf_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.clf_label, logits=self.cout,
                                                        name='clf_loss'))
            self.clf_if_flag = tf.arg_max(self.cout, 1)


        with tf.variable_scope('acc'):
            alayer = self.inputs
            for i in range(1, len(a_net) - 1):
                alayer = tf.layers.dense(inputs=alayer, units=a_net[i], activation=tf.nn.relu,
                                         kernel_initializer=tf.random_normal_initializer(0.0, 0.1))
                alayer = tf.multiply(alayer, self.split_layers[i - 1])
            alayer = tf.layers.dense(inputs=alayer, units=a_net[-1],
                                     kernel_initializer=tf.random_normal_initializer(0.0, 0.1))
            self.prediction_op = alayer
            self.accuracy = error_func(self.label, alayer)
            self.loss = tf.losses.mean_squared_error(labels=self.label, predictions=alayer)

        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss + self.clf_loss)
        self.train_op_single = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        self.clf_dense_1 = [v for v in tf.trainable_variables() if v.name == 'clf/dense_1/kernel:0'][0]
        self.clf_dense_0 = [v for v in tf.trainable_variables() if v.name == 'clf/dense/kernel:0'][0]
        tf.global_variables_initializer().run()

    def next_batch(self, batch_size=128, train=True):
        if train:
            batch = self.train_data[np.random.choice(list(range(len(self.train_data))), batch_size)]
        else:
            batch = self.test_data[np.random.choice(list(range(len(self.test_data))), batch_size)]
        x_data = batch[:, :-self.a_net[-1]]
        x_data = np.reshape(x_data, (batch_size, self.a_net[0]))
        y_data = batch[:, -self.a_net[-1]:]
        y_data = np.reshape(y_data, (batch_size, self.a_net[-1]))
        return x_data, y_data


    def create_clf_data_label(self, batch_size=128, train=True, final=False):
        if not final:
            batch = self.next_batch(batch_size, train=train)
            accu = self.sess.run(self.accuracy, feed_dict={self.inputs: batch[0], self.label: batch[1]})
            if self.a_net[-1] != 1:
                re = np.all(accu < self.error_bound, 1).reshape([batch_size, 1])
            else:
                re = accu < self.error_bound
            return batch[0], batch[1], np.concatenate((~re, re),1).astype('float32')
        else:
            batch = self.test_data
            x_data = batch[:, :-self.a_net[-1]]
            x_data = np.reshape(x_data, (self.test_data.shape[0], self.a_net[0]))
            y_data = batch[:, -self.a_net[-1]:]
            y_data = np.reshape(y_data, (self.test_data.shape[0], self.a_net[-1]))
            accu = self.sess.run(self.accuracy, feed_dict={self.inputs: x_data, self.label: y_data})
            if self.a_net[-1] != 1:
                re = np.all(accu < self.error_bound, 1).reshape([self.test_data.shape[0], 1])
            else:
                re = accu < self.error_bound
            return x_data, y_data, np.concatenate((~re, re),1).astype('float32')

    def train(self, epoch, single=False):
        test_clf_loss = []
        test_loss = []
        test_err = []
        test_true_invocation = []
        test_clf_invocation = []
        saver = tf.train.Saver()
        for i in range(epoch):
            if i % 100 == 0:
                x, y, clf_y = self.create_clf_data_label(train=False)
                l, cl, a, clf_if = self.sess.run([self.loss, self.clf_loss, self.accuracy, self.clf_if_flag],
                                                 feed_dict={self.inputs: x, self.label: y, self.clf_label: clf_y})
                err = np.mean(np.mean(a,1) * clf_if)  # Error of those samples labeled Safe-to-approximate.
                clf_y = np.argmax(clf_y, 1)
                true_invocation = np.sum(clf_y) / clf_y.shape[0]
                clf_invocation = np.sum(clf_if) / clf_if.shape[0]

                print(
                    'Iter %04d, Loss: %.6f, ClfLoss: %.4f, AllLoss: %.5f, TrueIVC: %.3f, ClfIVC: %.3f, MRE: %f' % (
                        i, l, cl, l + cl, true_invocation, clf_invocation, err))

                test_clf_loss.append(cl)
                test_loss.append(l)
                test_err.append(err)
                test_true_invocation.append(true_invocation)
                test_clf_invocation.append(clf_invocation)

            if i % 10000 == 9999:
                saver.save(self.sess, self.weight_dir + '/model.ckpt', i)
                print('checkpoint save: ', i)

            else:
                if single:
                    x, y, clf_y = self.create_clf_data_label(train=True)
                    self.sess.run(self.train_op_single, feed_dict={self.inputs: x, self.label: y, self.clf_label: clf_y})
                else:
                    x, y, clf_y = self.create_clf_data_label(train=True)
                    self.sess.run(self.train_op, feed_dict={self.inputs: x, self.label: y, self.clf_label: clf_y})

        saver = tf.train.Saver()
        saver.save(self.sess, self.weight_dir + '/model.ckpt')
        self._draw(test_clf_loss, test_loss, test_err, test_true_invocation, test_clf_invocation)
        self._record()

    def _record(self):
        with open(self.result_dir + '/result.txt', 'w') as f:
            re = []
            ab = []
            tr = []
            ci = []
            x, y, clf_y = self.create_clf_data_label(train=False, final=True)
            clf_if, accu = self.sess.run([self.clf_if_flag, self.accuracy],
                                         feed_dict={self.inputs: x, self.label: y})
            clf_y = np.argmax(clf_y, 1)

            tr.append(np.sum(clf_y) / clf_y.size)
            ci.append(np.sum(clf_if) / clf_if.size)
            re.append(np.sum([1 if clf_y[i] == clf_if[i] else 0 for i in range(clf_if.size)]) / clf_if.size)
            ab.append(np.mean(np.multiply(clf_if, np.mean(accu, 1))))

            f.write("Now we have tail!!" + '\n')
            f.write("True invocation: \t\t\t\t" + str(np.mean(tr)) + '\n')
            f.write("CLF invocation: \t\t\t\t" + str(np.mean(ci)) + '\n')
            f.write("Final CLF accuracy: \t\t\t" + str(np.mean(re)) + '\n')
            f.write("Error in CLF's prediction: \t\t" + str(ab[0]) + '\n')

    def _draw(self, test_clf_loss, test_loss, test_err, test_true_invocation, test_clf_invocation):
        if self.keep_tail:
            total_loss = [test_clf_loss[i] + test_loss[i] for i in range(len(test_loss))]
            draw([test_clf_loss, test_loss, total_loss], ['clf loss', 'acc loss', 'total loss'], 'Test Loss',
                 dir=self.result_dir)
            draw([test_true_invocation, test_clf_invocation], ['true', 'clf'], 'Invocation', self.result_dir)
        else:
            draw([test_loss], ['acc loss'], 'Test Loss', self.result_dir)
            draw([test_true_invocation], ['true'], 'Invocation', self.result_dir)

    def restore(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)
        print('begin record')
        self._record()


    def _make_dir(self, dirname=None):
        assert dirname is not None
        path = os.getcwd()
        if not os.path.exists(path + '/weight'):
            os.makedirs(path + '/weight')
        if not os.path.exists(path + dirname + self.benchmark):
            os.makedirs(path + dirname + self.benchmark)
        if len(os.listdir(path + dirname + self.benchmark)) == 0:
            os.makedirs(path + dirname + self.benchmark + '/000')
            f = open(path + dirname + self.benchmark + '/000/' + '0: a%s, c%s, eb%f, lr%f, keeptail %s' % (
                str(self.a_net), str(self.c_net), self.error_bound, self.learning_rate, str(self.keep_tail)), 'w')
            f.close()
            return path + dirname + self.benchmark + '/000'
        else:
            num = len(os.listdir(path + dirname + self.benchmark))
            os.makedirs(path + dirname + self.benchmark + '/%03d' % num)
            f = open(
                path + dirname + self.benchmark + '/%03d/' % num + str(num) + ': a%s, c%s, eb%f, lr%f, keeptail %s' % (
                    str(self.a_net), str(self.c_net), self.error_bound, self.learning_rate, str(self.keep_tail)), 'w')
            f.close()
            return path + dirname + self.benchmark + '/%03d' % num
