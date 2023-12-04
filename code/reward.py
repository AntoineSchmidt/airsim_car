import numpy as np


# reward algorithm, rewarding travelled distance
last_position = None
def compute_reward(client, edges=None):
    global last_position

    position = client.simGetVehiclePose().position
    position = np.array([position.x_val, position.y_val])

    if edges is not None:
        position = np.clip(position, edges[0], edges[1])

    reward = 0
    if last_position is not None:
        speed = client.getCarState().speed
        if 10 <= speed:
            #reward = np.linalg.norm(position - last_position) # total distance
            reward = np.max(np.abs(position - last_position)) # straight distance, streets are parallel to x or y axis
            reward *= np.sin((min(20, speed) - 10) * np.pi / 10) # max reward factor at 15 speed
            reward /= 2

    last_position = position

    if client.simGetCollisionInfo().has_collided:
        if client.simGetCollisionInfo().object_id == 0: # crash with animal
            return reward, True, True
        return -5, True, False

    return reward, False, False