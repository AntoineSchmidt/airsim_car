import tensorflow as tf


# handel tensorboard file
class TensorboardWriter:
    def __init__(self, store_dir):
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.writer = tf.summary.FileWriter(store_dir)

        self.loss = tf.placeholder(tf.float32, name="loss")
        tf.summary.scalar("loss", self.loss)
        self.performance = tf.summary.merge_all()

    # write data to tensorboard file
    def write_episode_data(self, episode, values):
        summary = self.sess.run(self.performance, feed_dict={ self.loss: values["loss"] })
        self.writer.add_summary(summary, episode)
        self.writer.flush()

    # close tensorboard writer
    def close_session(self):
        self.writer.close()
        self.sess.close()