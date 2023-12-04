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

DISCOUNT = 0.98
END_SKIP = 150
EPOCHS = 1
MAX_STEPS = 1000
DATAPOINTS = 5000

trace = {
    "image": np.zeros((DATAPOINTS, 72, 128, 3), dtype=np.uint8),
    "speed": np.zeros((DATAPOINTS, 1)),
    "value": np.zeros((DATAPOINTS, 1)),
    "control": np.zeros((DATAPOINTS, 2)),
    "probability": np.zeros((DATAPOINTS, 1)),
    "reward": np.zeros((DATAPOINTS, 1)),
}

positions = [ # starting positions
    airsim.Pose(airsim.Vector3r( 0,0,-1), airsim.utils.to_quaternion(0,0, 0)),
    airsim.Pose(airsim.Vector3r(85,0,-1), airsim.utils.to_quaternion(0,0, 0)),
    airsim.Pose(airsim.Vector3r( 0,0,-1), airsim.utils.to_quaternion(0,0, 1.5)),
    airsim.Pose(airsim.Vector3r( 0,0,-1), airsim.utils.to_quaternion(0,0, 3.5)),
    airsim.Pose(airsim.Vector3r( 0,0,-1), airsim.utils.to_quaternion(0,0, 5))
]
positions_edges = [ # min x,y / max x,y
    (np.array([   5, -120]), np.array([ 80,  -5])),
    (np.array([  90, -120]), np.array([120,  -5])),
    (np.array([   5,    5]), np.array([120, 120])),
    (np.array([-120,    5]), np.array([ -5, 120])),
    (np.array([-120, -120]), np.array([ -5,  -5]))
]

# reinforce model
index = 0
episode = 0
while True:
    # initialize position
    client.reset()
    positions_id = np.random.choice(len(positions))
    client.simSetVehiclePose(positions[positions_id], True, vehicle_name="Agent")
    compute_reward(client, positions_edges[positions_id])

    speed = np.zeros((1, 1))
    image = np.zeros((1, 72, 128, 3), dtype=np.uint8)
    for step in range(min(MAX_STEPS, DATAPOINTS - index)):
        # retrieve data
        data = data_live(client)
        if data:
            image = np.roll(image, 1, axis=-1)
            speed[0, 0], image[0, :, :, 0] = data
        else: continue

        # predict action
        control, probability, value = agent.sess.run([agent.out_control_sampled, agent.out_control_sampled_prob, agent.out_value], feed_dict={agent.image: image, agent.speed: speed})
        control, probability, value = control[0], probability, value[0, 0]

        # execute action
        car_controls.throttle = float(np.clip(control[0], -1, 1))
        car_controls.steering = float(np.clip(control[1], -1, 1))
        client.setCarControls(car_controls)

        # update reward
        reward, crashed, crashed_animal = compute_reward(client, positions_edges[positions_id])

        print("{:4d}: {:5.2f} {:5.2f} {:5.2f}, {:6.3f}".format(step, car_controls.throttle, car_controls.steering, value, reward))

        # save learning data
        trace["image"][index + step] = image[0]
        trace["speed"][index + step] = speed[0, 0]
        trace["value"][index + step] = value
        trace["control"][index + step] = control
        trace["probability"][index + step] = probability
        trace["reward"][index + step] = reward

        if crashed:
            break

    # prepare data
    if step > 50: # drop too short episodes
        for i in range(index, index + step)[::-1]: # calculate on-policy state values
            trace["reward"][i] += DISCOUNT * trace["reward"][i + 1]

        index += step + 1
        if not crashed or crashed_animal: # bypass last state value estimation
            index -= min(step, END_SKIP)

        # train model
        if index >= DATAPOINTS - END_SKIP:
            print("Improving policy")

            data = {
                agent.image: trace["image"][:index],
                agent.speed: trace["speed"][:index],
                agent.control: trace["control"][:index],
                agent.probability: trace["probability"][:index],
                agent.value: trace["reward"][:index],
                agent.advantage: trace["reward"][:index] - trace["value"][:index] * int(min(10, episode) / 10) # ignore baseline the first 10 episodes
            }

            # train and save policy
            for _ in range(EPOCHS):
                agent.sess.run([agent.rl_optimizer], feed_dict=data)
            agent.save("agent_ppo.ckpt", step=episode)

            index = 0
            episode += 1

# restore to original state
client.reset()
client.enableApiControl(False)