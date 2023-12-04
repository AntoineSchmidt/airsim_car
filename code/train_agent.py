import numpy as np
import matplotlib.pyplot as plt

from data import *
from model import *
from tensorboard_writer import *


# train parameter
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.05
STEERING_CORRECTION = 0.1    

# initialize model
agent = Model()

# setup tensorboard files
tensorboard_train = TensorboardWriter("tensorboard/train")
tensorboard_valid = TensorboardWriter("tensorboard/valid")

# load data
data = [load_folder("./data/{}".format(i)) for i in [3, 1, 2, 0]]
speed = np.concatenate([data[i][0] for i in range(len(data))])
control = np.concatenate([data[i][1] for i in range(len(data))])
images_left = np.concatenate([data[i][2] for i in range(len(data))])
images_center = np.concatenate([data[i][3] for i in range(len(data))])
images_right = np.concatenate([data[i][4] for i in range(len(data))])

# correct steering for left and right images
control_left = control + np.array([0, STEERING_CORRECTION])
control_right = control - np.array([0, STEERING_CORRECTION])

# validation and training data split
valid_size = int(len(speed) * VALIDATION_SPLIT)

speed_train = np.concatenate([speed[valid_size:]] * 3)
control_train = np.concatenate([control_left[valid_size:], control[valid_size:], control_right[valid_size:]])
images_train = np.concatenate([images_left[valid_size:], images_center[valid_size:], images_right[valid_size:]])

speed_valid = np.concatenate([speed[:valid_size]] * 3)
control_valid = np.concatenate([control_left[:valid_size], control[:valid_size], control_right[:valid_size]])
images_valid = np.concatenate([images_left[:valid_size], images_center[:valid_size], images_right[:valid_size]])

# clip control for tanh, reduce acceleration
control_train = np.clip((control_train * np.array([0.9, 1.0])) - np.array([0.03, 0]), -1.0, 1.00)
control_valid = np.clip((control_valid * np.array([0.9, 1.0])) - np.array([0.03, 0]), -1.0, 1.00)

# check data
print(min(speed), max(speed), np.mean(speed))
print(min(control[:, 0]), max(control[:, 0]), np.mean(control[:, 0]))
print(min(control[:, 1]), max(control[:, 1]), np.mean(control[:, 1]))
#for i in range(3):
#    plt.imshow(images_center[234, :, :, i])
#    plt.show()

# validation dict
valid_dict = { agent.image: images_valid, agent.speed: speed_valid, agent.control: control_valid }


# sample a balanced minibatch
indices_p_p = np.arange(len(speed_train))[(control_train[:, 0] >= 0) * (control_train[:, 1] > 0)]
indices_p_n = np.arange(len(speed_train))[(control_train[:, 0] >= 0) * (control_train[:, 1] <= 0)]
indices_n_p = np.arange(len(speed_train))[(control_train[:, 0] < 0) * (control_train[:, 1] > 0)]
indices_n_n = np.arange(len(speed_train))[(control_train[:, 0] < 0) * (control_train[:, 1] <= 0)]
def sample_minibatch():
    indices = np.random.choice(indices_p_p, BATCH_SIZE // 4)
    indices = np.append(indices, np.random.choice(indices_p_n, BATCH_SIZE // 4))
    indices = np.append(indices, np.random.choice(indices_n_p, BATCH_SIZE // 4))
    indices = np.append(indices, np.random.choice(indices_n_n, BATCH_SIZE // 4))
    return images_train[indices], speed_train[indices], control_train[indices]


# train agent
batch = 0
while True:
    batch += 1

    images_batch, speed_batch, control_batch = sample_minibatch()
    data = { agent.image: images_batch, agent.speed: speed_batch, agent.control: control_batch }
    _, loss = agent.sess.run([agent.il_optimizer, agent.il_loss], feed_dict=data)
    tensorboard_train.write_episode_data(batch, { "loss": loss })
    print(batch, loss)

    # validate agent
    if batch % 1000 == 0:
        loss = agent.sess.run(agent.il_loss, feed_dict=valid_dict)
        tensorboard_valid.write_episode_data(batch, { "loss": loss })

        agent.save("agent.ckpt", step=batch)
        print("validation loss:", loss)