import os
import time

import tensorflow as tf
import numpy as np

setattr(tf.contrib.rnn.GRUCell, '__deepcopy__', lambda self, _: self)
setattr(tf.contrib.rnn.BasicLSTMCell, '__deepcopy__', lambda self, _: self)
setattr(tf.contrib.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)


class Seq2SeqModel:
    def __init__(self,
                 xseq_len,
                 yseq_len,
                 x_vocab_size,
                 y_vocab_size,
                 emb_dim,
                 num_layers,
                 ckpt_path,
                 metadata,
                 lr=0.001,
                 epochs=10000,
                 model_name='chatbot_model',
                 batch_size=64,
                 hook=None
                 ):
        self.xseq_len = xseq_len
        self.yseq_len = yseq_len
        self.ckpt_path = ckpt_path
        self.epochs = epochs
        self.model_name = model_name
        self.batch_size = batch_size
        self.metadata = metadata
        self.sess = None
        self.hook = hook

        tf.reset_default_graph()

        print('starting building model')
        # 对输入（问题）进行编码，有多少个（xseq_len）输入单词就有多少个placeholder
        self.enc_ip = [tf.placeholder(shape=[None, ], dtype=tf.int64,
                                      name='encode_input_x{}'.format(t))
                       for t in range(xseq_len)]
        # 对输入（回答）进行编码，有多少个（yseq_len）输入单词就有多少个placeholder
        self.labels = [tf.placeholder(shape=[None, ], dtype=tf.int64,
                                      name='encode_input_y{}'.format(t))
                       for t in range(yseq_len)]

        # 解码即为在y前加上开始符号GO
        self.dec_ip = [tf.zeros_like(self.enc_ip[0], dtype=tf.int64, name='GO')] + self.labels[:-1]

        # LSTM网络
        self.keep_pro = tf.placeholder(tf.float32, name='keep_pro')
        basic_cell = tf.contrib.rnn.BasicLSTMCell(emb_dim, state_is_tuple=True)
        basic_cell = tf.contrib.rnn.DropoutWrapper(basic_cell, output_keep_prob=self.keep_pro)
        # 多层LSTM
        stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([basic_cell] * num_layers, state_is_tuple=True)

        # 解码seq
        with tf.variable_scope('decoder') as scope:
            # train
            self.decode_outputs, self.decode_states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
                self.enc_ip, self.dec_ip, stacked_lstm, x_vocab_size, y_vocab_size, emb_dim)

            # 测试时需要共享参数
            scope.reuse_variables()
            # test 设置feed_previous=True 则解码输入第一个为GO，其他的的解码输入为前一个解码的输出
            self.decode_outputs_test, self.decode_states_test = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
                self.enc_ip, self.dec_ip, stacked_lstm, x_vocab_size, y_vocab_size, emb_dim,
                feed_previous=True)

        # 最后一层，计算loss
        # 每一个label对应的权重
        loss_weights = [tf.ones_like(label, dtype=tf.float32) for label in self.labels]
        self.loss = tf.contrib.legacy_seq2seq.sequence_loss(self.decode_outputs, self.labels, loss_weights,
                                                            y_vocab_size,
                                                            name='loss')
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss, global_step=self.global_step,
                                                                          name='train_op')
        print('finish building model')

    def get_feed(self, x, y, keep_prob):
        """
        feed X和Y，定义了多少个enc_ip 就对应多少个feed_dict 里的key, Y(label)同理
        :param X:
        :param Y:
        :param keep_prob:
        :return:
        """
        feed_dict = {self.enc_ip[t]: x[t] for t in range(self.xseq_len)}
        feed_dict.update({self.labels[t]: y[t] for t in range(self.yseq_len)})
        feed_dict[self.keep_pro] = keep_prob

        return feed_dict

    def batch_gen(self, x, y, batch_size, epoch):
        """
        batch 生成
        :param x:
        :param y:
        :param batch_size:
        :param epoch:
        :return:
        """
        for ep in range(epoch):
            # print('epoch: {}'.format(ep))
            for i in range(0, len(x), batch_size):
                yield x[i: i + batch_size].T, y[i: i + batch_size].T, ep

    def restore_last_session(self):
        """
        重载模型
        :return:
        """
        saver = tf.train.Saver()
        sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt.model_checkpoint_path)
            self.sess = sess
        return saver

    def test_step(self, x, y, batch_size=None):
        """

        :param batch_size: 
        :param x:
        :param y:
        :return:
        """
        if batch_size is None:
            batch_size = self.batch_size

        batches = self.batch_gen(x, y, batch_size, 1)
        losses = []
        for batch in batches:
            test_x, test_y, _ = batch
            feed_dict = self.get_feed(test_x, test_y, keep_prob=1.)
            loss_v, dec_op_v = self.sess.run([self.loss, self.decode_outputs_test], feed_dict)
            losses.append(loss_v)
            dec_op_v = np.array(dec_op_v).transpose([1, 0, 2])
            dec_op_v = np.argmax(dec_op_v, axis=2)
            self.decode_to_text(test_x.T, dec_op_v, test_y.T)
        return np.mean(losses)

    def decode_to_text(self, input, output, y=None):
        """
        解码为文字
        :return:
        """
        replies = []
        if y is not None:
            for ii, oi, yi in zip(input, output, y):
                q = self.decode(sequence=ii, lookup=self.metadata['idx2w'], separator=' ')
                a = self.decode(sequence=yi, lookup=self.metadata['idx2w'], separator=' ')
                decoded = self.decode(sequence=oi, lookup=self.metadata['idx2w'], separator=' ').split(' ')

                print('q : [{0}]; a : [{1}], y: [{2}]'.format(q, ' '.join(decoded), a))
                replies.append(decoded)
        else:
            for ii, oi in zip(input, output):
                q = self.decode(sequence=ii, lookup=self.metadata['idx2w'], separator=' ')
                decoded = self.decode(sequence=oi, lookup=self.metadata['idx2w'], separator=' ').split(' ')

                print('q : [{0}]; a : [{1}]'.format(q, ' '.join(decoded)))
                replies.append(decoded)

        return replies

    def decode(self, sequence, lookup, separator=''):
        """
        解码为文字
        :param sequence:
        :param lookup:
        :param separator:
        :return:
        """
        return separator.join([lookup[element] for element in sequence if element])

    def train(self, train_set, test_set):
        """
        训练
        :param train_set: train_set[0], rain_set[1]
        :param test_set: test_set[0], test_set[1]
        :return:
        """
        if not os.path.exists(self.ckpt_path):
            os.mkdir(self.ckpt_path)

        # 尝试重载已训练模型
        saver = self.restore_last_session()
        if self.sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

        # saver = tf.train.Saver()
        current_epoch = 0
        batches = self.batch_gen(train_set[0], train_set[1], self.batch_size, self.epochs)
        for batch in batches:
            x, y, epoch = batch
            is_next_ep = epoch != current_epoch
            # training
            feed_dict = self.get_feed(x, y, 0.5)
            _, step = self.sess.run([self.train_op, self.global_step], feed_dict=feed_dict)
            if step % 100 == 0:
                loss, = self.sess.run([self.loss, ], feed_dict=feed_dict)
                print('{} epoch: {} step: {} train loss: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), epoch, step, loss))

            if step % 500 == 0:
                test_loss = self.test_step(test_set[0], test_set[1])
                print('{} epoch: {} step: {} test loss: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), epoch, step, test_loss))

            if is_next_ep and epoch % 10 == 0:
                saver.save(self.sess, os.path.join(self.ckpt_path, self.model_name + '.ckpt'), global_step=step)
            elif is_next_ep:
                current_epoch = epoch

            if self.hook is not None and is_next_ep and epoch % 500 == 0:
                self.hook()
                
        if self.hook is not None:
            self.hook()
