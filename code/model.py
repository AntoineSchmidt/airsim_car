import numpy as np
import tensorflow as tf # TF 1.15
import tensorflow_probability as tfp # TFP 0.8


# tensorflow model
class Model():
    NOISE = [0.1, 0.1] # throttle, steering

    def ppo_loss(out_control, control, probability, advantage):
        mvn = tfp.distributions.MultivariateNormalDiag(loc=out_control, scale_diag=Model.NOISE)
        ratio = tf.math.exp(mvn.log_prob(control) - probability)
        return -tf.reduce_mean(tf.minimum(ratio * advantage, tf.clip_by_value(ratio, clip_value_min=0.8, clip_value_max=1.2) * advantage))

    def __init__(self, il_lr=1e-3, rl_lr=1e-4):
        # setup session, dynamically grow the memory used on the GPU
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.config)

        # placeholder, input
        self.image = tf.placeholder(tf.float32, shape=(None, 72, 128, 3), name="image")
        self.speed = tf.placeholder(tf.float32, shape=(None, 1), name="speed")

        # placeholder, label
        self.control = tf.placeholder(tf.float32, shape=(None, 2), name="control")  # throttle, steering
        self.value = tf.placeholder(tf.float32, shape=(None, 1), name="value")

        # additional placeholder, ppo
        self.probability = tf.placeholder(tf.float32, shape=(None, 1), name="probability")
        self.advantage = tf.placeholder(tf.float32, shape=(None, 1), name="advantage")

        # network convolutional layer
        layer = self.image
        layer = tf.layers.conv2d(inputs=layer, filters=3, kernel_size=9, strides=4, padding="same", activation=tf.nn.relu)
        layer = tf.layers.conv2d(inputs=layer, filters=4, kernel_size=5, strides=2, padding="same", activation=tf.nn.relu)
        layer = tf.layers.MaxPooling2D(pool_size=2, strides=2, padding="same")(layer)
        layer = tf.contrib.layers.flatten(layer)

        # add speed information
        layer = tf.concat([layer, self.speed], axis=-1)

        # dense layer
        layer_control = tf.layers.dense(inputs=layer, units=10, activation=tf.nn.relu)
        layer_value = tf.layers.dense(inputs=layer, units=10, activation=tf.nn.relu)

        # network output
        self.out_control = tf.layers.dense(inputs=layer_control, units=2, activation=tf.nn.tanh) # throttle, steering
        self.out_value = tf.layers.dense(inputs=layer_value, units=1)

        # sample action for ppo
        mvn = tfp.distributions.MultivariateNormalDiag(loc=self.out_control, scale_diag=Model.NOISE)
        self.out_control_sampled = mvn.sample()
        self.out_control_sampled_prob = mvn.log_prob(self.out_control_sampled)

        # setup imitation trainer
        self.il_loss = tf.losses.mean_squared_error(labels=self.control, predictions=self.out_control)
        self.il_optimizer = tf.train.AdamOptimizer(learning_rate=il_lr).minimize(self.il_loss)

        # setup reinforcement trainer
        self.rl_loss_control = Model.ppo_loss(self.out_control, self.control, self.probability, self.advantage)
        self.rl_loss_value = tf.losses.mean_squared_error(labels=self.value, predictions=self.out_value)
        self.rl_optimizer = tf.group(tf.train.GradientDescentOptimizer(learning_rate=rl_lr).minimize(self.rl_loss_control),\
                                     tf.train.AdamOptimizer(learning_rate=rl_lr).minimize(self.rl_loss_value))

        # setup model saver
        self.saver = tf.train.Saver(max_to_keep=5)

        # initialize variables
        self.sess.run(tf.global_variables_initializer())

    # load trained model
    def load(self, file_name=None):
        if file_name is None: file_name = tf.train.latest_checkpoint("./model/")
        else: file_name = "./model/" + file_name
        print("Loading:", file_name)
        return self.saver.restore(self.sess, file_name)

    # save trained model
    def save(self, file_name, step):
        return self.saver.save(self.sess, "./model/" + file_name, global_step=step)