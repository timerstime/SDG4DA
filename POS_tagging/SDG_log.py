import numpy as np
import tensorflow as tf


# input: sampled set of sentence representation of last time step
# output: one-dimensional selection distribution vector
# n_steps: max_bag_size
# input_size: max  sentence representation vector size

class SDG(object):
    def __init__(self, sess, n_steps, input_size, output_size, cell_size, batch_size=32,lr=0.01, repr=None):
        self.n_steps = n_steps #n_steps here means the number of action choices
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.td_error = tf.placeholder(tf.float32, None, "td_error")
        self.xs = tf.placeholder(tf.float32,[batch_size, None, input_size], "state_size")
        self.sess = sess
        self.xs_input = tf.reshape(self.xs, [-1,input_size])
        self.pred = tf.layers.dense(
            inputs=self.xs_input,
            units=2,
            activation=tf.sigmoid,  #logistic activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        print("shape of pred:", self.pred)
        self.all_act_prob = tf.nn.softmax(self.pred, name="act_prob", dim=1)
        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.all_act_prob)
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)



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
        print("learning...number is:", self.n_steps)
        feed_dict = {self.xs: X_train, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    # s is the input selection distribution vector
    def deform(self, s):     # to do: maybe a one-step reward should be returned.
        feed_dict = {self.xs: s}
        s_ = self.sess.run(self.all_act_prob, feed_dict)
        return s_



    def sample(self, s, repr, labels):
        # s: numpy array of probabilities
        # repr should be the
        # s = s[np.newaxis, :]
        select_index=[]
        sess = tf.Session()
        with sess.as_default():
            print("#######",s.eval().size)
            print("sampling...number:", self.n_steps)
            s_v = s.eval()
        for i in range(s.get_shape()[0]):
            sample = np.random.uniform(0,1)
            if sample <= s_v[i][0]:
               select_index.append(i)
        if len(select_index)==0:
           select_index = np.arange(500)
        print("select_index lenth:",len(select_index))

        select_repr=[]
        select_label=[]

        for item in select_index:
            select_repr.append(repr[item])
            select_label.append(labels[item])

        return select_repr, select_label,len(select_index)


    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)











































