import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import FlattenObservation
import matplotlib.pyplot as plt
import json

class DroneDeliveryEnv(gym.Env):
    def __init__(self, distance_matrix, weights, battery_capacity=1000, base_power=10, alpha=2, velocity=10):
        super(DroneDeliveryEnv, self).__init__()

        self.distance_matrix = np.array(distance_matrix)
        self.weights = np.array(weights)
        self.n_nodes = len(weights)
        self.depot = 0

        self.battery_capacity = battery_capacity
        self.base_power = base_power
        self.alpha = alpha
        self.velocity = velocity

        self.action_space = spaces.Discrete(self.n_nodes)
        self.observation_space = spaces.Dict({
            "location": spaces.Discrete(self.n_nodes),
            "battery": spaces.Box(low=0, high=self.battery_capacity, shape=(1,), dtype=np.float32),
            "visited": spaces.MultiBinary(self.n_nodes)
        })

        self.reset()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_node = self.depot
        self.battery = self.battery_capacity
        self.visited = np.zeros(self.n_nodes, dtype=np.int8)
        self.visited[self.depot] = 1
        self.total_energy = 0
        self.route = [self.depot]
        self.energy_per_hop = []
        return self._get_obs(), {}

    def _get_obs(self):
        return {
            "location": self.current_node,
            "battery": np.array([self.battery], dtype=np.float32),
            "visited": self.visited.copy()
        }

    def _energy_cost(self, from_node, to_node):
        distance = self.distance_matrix[from_node][to_node]
        weight = self.weights[from_node]
        return (self.base_power + self.alpha * weight) * (distance / self.velocity)

    def step(self, action):
        done = False
        info = {}

        if action == self.current_node or self.visited[action]:
            reward = -100
            return self._get_obs(), reward, done, False, info

        energy = self._energy_cost(self.current_node, action)
        return_energy = self._energy_cost(action, self.depot)

        if self.battery < energy + return_energy:
            reward = -100
            return self._get_obs(), reward, done, False, info

        self.battery -= energy
        self.total_energy += energy
        self.energy_per_hop.append((self.current_node, action, energy))
        self.current_node = action
        self.visited[action] = 1
        self.route.append(action)

        if np.all(self.visited[1:]):
            energy_back = self._energy_cost(self.current_node, self.depot)
            if self.battery >= energy_back:
                self.battery -= energy_back
                self.total_energy += energy_back
                self.energy_per_hop.append((self.current_node, self.depot, energy_back))
                self.route.append(self.depot)
                reward = -self.total_energy
                done = True
            else:
                reward = -100
        else:
            reward = -energy

        return self._get_obs(), reward, done, False, info

if __name__ == "__main__":
    # Read test case from file
    distance_matrix = []
    weights = []
    with open("test_case_15_nodes.txt") as file:
        lines = file.readlines()
        parsing_matrix = False
        for line in lines:
            if line.strip().startswith("Distance Matrix"):
                parsing_matrix = True
                continue
            elif line.strip().startswith("Weights"):
                parsing_matrix = False
                continue
            elif parsing_matrix and line.strip():
                row = list(map(int, line.strip().split()))
                distance_matrix.append(row)
            elif not parsing_matrix and line.strip():
                weights = list(map(int, line.strip().split()))

    base_env = DroneDeliveryEnv(distance_matrix, weights)
    wrapped_env = FlattenObservation(base_env)
    vec_env = make_vec_env(lambda: DroneDeliveryEnv(distance_matrix, weights), n_envs=1)

    model = PPO("MultiInputPolicy", vec_env, verbose=1, n_steps=256, batch_size=64, gae_lambda=0.95, gamma=0.99)
    model.learn(total_timesteps=20000)

    obs, _ = base_env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, _, _ = base_env.step(action)

    print("\nOptimal Route:")
    print(" -> ".join([f"L{n}" for n in base_env.route]))

    print("\nEnergy Consumption (per hop):")
    for from_node, to_node, energy in base_env.energy_per_hop:
        print(f"L{from_node} -> L{to_node}: {energy:.2f} units")

    print(f"\nTotal Energy Used: {base_env.total_energy:.2f} units")

    coordinates = np.random.rand(base_env.n_nodes, 2) * 100
    for i in range(len(base_env.route) - 1):
        start = coordinates[base_env.route[i]]
        end = coordinates[base_env.route[i+1]]
        plt.plot([start[0], end[0]], [start[1], end[1]], 'bo-')
        plt.text(start[0], start[1], f"L{base_env.route[i]}", fontsize=9)
    plt.text(coordinates[base_env.route[-1]][0], coordinates[base_env.route[-1]][1], f"L{base_env.route[-1]}", fontsize=9)
    plt.title("Drone Route Visualization")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()



