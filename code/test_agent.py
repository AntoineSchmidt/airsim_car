import airsim
import numpy as np

from data import *
from model import *
from reward import *


print("Connecting to AirSim...")
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()

print("Loading Model...")
agent = Model()
agent.load()
agent.load("agent.ckpt-10000")

EPISODES = 10
MAX_STEPS = 2000

# evaluate model
reward_list = []
for i in range(EPISODES):
    reward = 0

    # initialize position
    compute_reward(client)

    speed = np.zeros((1, 1))
    image = np.zeros((1, 72, 128, 3), dtype=np.uint8)
    for step in range(MAX_STEPS):
        # retrieve data
        data = data_live(client)
        if data:
            image = np.roll(image, 1, axis=-1)
            speed[0, 0], image[0, :, :, 0] = data
        else: continue

        # predict action
        control = agent.sess.run(agent.out_control, feed_dict={agent.image: image, agent.speed: speed})

        # execute action
        car_controls.throttle = float(control[0, 0])
        car_controls.steering = float(control[0, 1])
        client.setCarControls(car_controls)

        # update reward
        reward_step, crashed, _ = compute_reward(client)
        reward += reward_step

        print("{:2d}/{:4d}: {:5.2f} {:5.2f}, {:6.3f}".format(i, step, car_controls.throttle, car_controls.steering, reward_step))

        if crashed: break

    # finish episode
    client.reset()
    print(i, reward)
    reward_list.append(reward)

# show rewards
print(reward_list)
print("mean:", np.mean(reward_list), "std:", np.std(reward_list))

# restore to original state
client.reset()
client.enableApiControl(False)