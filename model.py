# shared
import tensorflow as tf


class Seq2Seq:
    logits = None
    outputs = None
    cost = None
    train_op = None

    def __init__(self, vocab_size, n_hidden=128, n_output=2, n_layers=3, output_keep_prob=0.80):
        self.learning_late = 0.001

        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.enc_input = tf.placeholder(tf.float32, [None, None, self.vocab_size])
        self.targets = tf.placeholder(tf.float32, [None, n_output])

        self.weights = tf.Variable(tf.random_normal([self.n_hidden, self.vocab_size]), name="weights")
        self.bias = tf.Variable(tf.zeros([self.vocab_size]), name="bias")
        self.W1 = tf.Variable(tf.random_normal([self.vocab_size, n_output]))
        self.B1 = tf.Variable(tf.random_normal([n_output]))
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.enc_input = tf.transpose(self.enc_input, [1, 0, 2])
        enc_cell = tf.nn.rnn_cell.MultiRNNCell([self.cell(output_keep_prob)
                                                for _ in range(self.n_layers)])
        outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, self.enc_input, dtype=tf.float32)

        self.enc_ret = tf.reshape(enc_states[-1][-1], [-1, self.n_hidden])
#        self.enc_ret = enc_states[-1][-1]

        # rnn돌린 것을 활성화해준다.
        self.logit1 = tf.matmul(self.enc_ret, self.weights) + self.bias
        self.logits = tf.nn.softmax(tf.matmul(self.logit1, self.W1) + self.B1)

        self.cost = -tf.reduce_sum(self.logits * tf.log(tf.clip_by_value(self.targets, 1e-10, 1.0))) - tf.reduce_sum(
            (1 - self.logits) * tf.log(tf.clip_by_value((1 - self.targets), 1e-10, 1.0)))
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_late).minimize(self.cost,
                                                                                          global_step=self.global_step)

        tf.summary.scalar('cost', self.cost)
        self.saver = tf.train.Saver(tf.global_variables())

    def cell(self, output_keep_prob):
        cells = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden)
        cells = tf.nn.rnn_cell.DropoutWrapper(cells, output_keep_prob=output_keep_prob)
        return cells

    def train(self, session, enc_input, targets):
        return session.run([self.train_op, self.logits],
                           feed_dict={self.enc_input: enc_input, self.targets: targets})

    def predict(self, session, enc_input):
        return session.run(self.outputs,
                           feed_dict={self.enc_input: enc_input, })

    def write_logs(self, session, writer, enc_input, targets):
        merged = tf.summary.merge_all()

        summary = session.run(merged, feed_dict={self.enc_input: enc_input,
                                                 self.targets: targets})

        writer.add_summary(summary, self.global_step.eval())
