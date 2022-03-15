import gym
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    actions = np.load("actions.npy", allow_pickle=True)
    states = np.load("states.npy", allow_pickle=True)
    print(f"Tenho {len(actions)} ações")
    print(f"Tenho {len(states)} estados")

    fig, ax = plt.subplots(3, 1)
    for i in range(3):
        ax[i].plot(actions[:, i])
    plt.show()

    plt.imshow(states[1000, :, :, :])
    plt.show()

if __name__ == "__main__":
    load_data()
