import gym
import numpy as np

def load_data():
    actions = np.load("actions.npy", allow_pickle=True)
    states = np.load("states.npy", allow_pickle=True)
    print(len(actions))
    print(len(states))

if __name__ == "__main__":
    load_data()
