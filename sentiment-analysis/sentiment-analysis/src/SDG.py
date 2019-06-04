import numpy as np
import tensorflow as tf


# input: sampled set of sentence representation of last time step
# output: one-dimensional selection distribution vector
# n_steps: max_bag_size
# input_size: max  sentence representation vector size

class SDG(object):
    def __init__(self, sess, n_steps, input_size, output_size, cell_size, batch_size=32,lr=0.01, repr=None):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.td_error = tf.placeholder(tf.float32, None, "td_error")
        self.sess = sess

        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.pred)
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def add_input_layer(self,):
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # (batch*n_step, in_size)
        # Ws (in_size, cell_size)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        # bs (cell_size, )
        bs_in = self._bias_variable([self.n_steps, self.cell_size])
        # l_in_y = (batch * n_steps, cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # reshape l_in_y ==> (batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)

    def add_output_layer(self):
        # shape = (batch * steps, cell_size)
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.n_steps,self.output_size])
        # shape = (batch * steps, output_size)
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.nn.sigmoid(tf.matmul(l_out_x, Ws_out) + bs_out)

    # def compute_cost(self):
    #     losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
    #         [tf.reshape(self.pred, [-1], name='reshape_pred')],
    #         [tf.reshape(self.ys, [-1], name='reshape_target')],
    #         [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
    #         average_across_timesteps=True,
    #         softmax_loss_function=self.ms_error,
    #         name='losses'
    #     )
    #     with tf.name_scope('average_cost'):
    #         self.cost = tf.div(
    #             tf.reduce_sum(losses, name='losses_sum'),
    #             self.batch_size,
    #             name='average_cost')
    #         tf.summary.scalar('cost', self.cost)

    def learn(self,X_train, td):
        #s = s[np.newaxis, :]
        feed_dict = {self.xs: X_train, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    # s is the input selection distribution vector
    def deform(self, s):     # to do: maybe a one-step reward should be returned.
        feed_dict = {self.xs: s}
        s_ = self.sess.run(self.pred, feed_dict)
        return s_



    def sample(self, s, repr, labels):
        # s: numpy array of probabilities
        # repr should be the
        # s = s[np.newaxis, :]
        select_index=[]
        sess = tf.Session()
        with sess.as_default():
            #print s.eval().size
            s_v = s.eval()
        for i in range(s.get_shape()[0]):
            sample = np.random.uniform(0,1)
            if sample <= s_v[i]:
               select_index.append(i)
        #select_index = np.arange(50)
        print("select_index lenth:",len(select_index))
        select_repr = tf.nn.embedding_lookup(repr,(select_index))
        select_label = tf.nn.embedding_lookup(labels,(select_index))
        sess1 = tf.Session()
        select_repr = sess1.run(select_repr)
        select_label = sess1.run(select_label)
        #print("####selected repr:", select_repr)
        #print("####selected label:", select_label)
        return select_repr, select_label


    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)
