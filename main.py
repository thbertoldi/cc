import gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image

def load_data():
    actions = np.load("actions.npy", allow_pickle=True)
    states = np.load("states.npy", allow_pickle=True)
    print(f"Tenho {len(actions)} ações")
    print(f"Tenho {len(states)} estados")
    # _, ax = plt.subplots(3, 1)
    # for i in range(3):
    #     ax[i].plot(actions[:, i])
    # plt.show()
    plt.imshow(states[300, :, :, :])
    plt.show()
    plt.imshow(states[600, :, :, :])
    plt.show()
    return (torch.from_numpy(actions), states)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(64, 512)
        self.fc2 = nn.Linear(512, 3)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.activation(x)
        return x

if __name__ == "__main__":
    metadata = load_data()
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(2):
        for i, _ in enumerate(metadata):
            action, state = metadata[0][i], metadata[1][i]
            state = np.transpose(state)
            print(f"Got size {len(state)}")
            output = net(torch.FloatTensor(state))
            print(f"Out tamanho {len(output)}")
            print(f"Out {output}")
            # action[0], action[1], action[2] = (action[0] + 1)/2, action[1], action[2]
            loss = criterion(output[-1], action)
            print("aplicou loss")
            loss.backward()
            optimizer.step()
            running_loss = loss.item()
            if i % 200 == 199:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

    torch.save(net.state_dict(), "resp.pth")
