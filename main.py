import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

SHOW_IMAGES = False
BATCH_SIZE = 128 # tamanho do batch para treinamento
GAME_START = 100 # pulando frames estranhos


def load_data():
    actions = np.load("actions.npy", allow_pickle=True)
    states = np.load("states.npy", allow_pickle=True)

    print(f"Tenho {len(actions)} ações")
    print(f"Tenho {len(states)} estados")
    if SHOW_IMAGES: # dados segundo o relatório
        _, ax = plt.subplots(3, 1)
        for i in range(3):
            ax[i].plot(actions[:, i])

        plt.show()
        plt.imshow(states[300, :, :, :])
        plt.show()
        plt.imshow(states[600, :, :, :])
        plt.show()
    return actions, states


def preprocessing(metadata):
    actions = metadata[0][GAME_START:]
    states = metadata[1][GAME_START:].astype("double")
    states = states / 255.0 # normaliza valores de rgb

    return states, actions


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding="same")
        self.conv3 = nn.Conv2d(64, 64, 2, padding="same")
        self.fcy = nn.Linear(1600, 4)
        self.fcz = nn.Linear(1600, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)

        y = self.fcy(x)
        y = torch.reshape(y, shape=(-1, 2, 2))
        y = F.softmax(y, dim=2)

        z = self.fcz(x)
        z = torch.sigmoid(z) * 2 - 1

        return (z, y)


if __name__ == "__main__":
    metadata = load_data()

    net = Net()
    cross_entropy_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.6)
    torch.autograd.set_detect_anomaly(True)
    all_actions = []
    all_outputs = []
    losses = []

    states, actions = preprocessing(metadata)

    # training loop
    for epoch in range(1):
        running_loss = 0.0
        for i in range(len(states) - BATCH_SIZE - 1):
            action, state = (actions[i : i + BATCH_SIZE], states[i : i + BATCH_SIZE])

            state = torch.tensor(state).float()
            state = state.permute(0, 3, 1, 2)

            target_steer = torch.tensor(action[:, 0])
            target_forward_backward = torch.tensor(action[:, 1:])
            target_forward_backward = F.one_hot(
                target_forward_backward.long(), num_classes=2
            )

            # optimzation
            optimizer.zero_grad()
            steer, forward_backward = net(state)
            steer = steer.double()
            forward_backward = forward_backward.double()

            loss = cross_entropy_loss(
                forward_backward.double(), target_forward_backward.double()
            ) + mse_loss(target_steer, steer)

            loss.backward()
            optimizer.step()
            running_loss = loss.item()

            all_actions.append(
                [
                    target_steer[0].detach().numpy(),
                    target_forward_backward[0].detach().numpy(),
                ]
            )
            all_outputs.append(
                [steer[0].detach().numpy(), forward_backward[0].detach().numpy()]
            )

            losses.append(running_loss)

            if i % 10 == 9:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss:.3f}")

                print(f"Action: {target_steer[0]}, output: {steer[0]}")
                print(
                    f"Action: {torch.argmax(target_forward_backward[0, 0])}, output: {torch.argmax(forward_backward[0, 0])}"
                )
                print(
                    f"Action: {torch.argmax(target_forward_backward[0, 1])}, output: {torch.argmax(forward_backward[0, 1])}"
                )

            if i % 500 == 499:
                break

    all_actions = np.array(all_actions)
    all_outputs = np.array(all_outputs)

    plt.figure(0)
    plt.plot(losses)

    all_steer = [action[0] for action in all_actions]
    all_target_steer = [output[0] for output in all_outputs]

    _, ax = plt.subplots(2, 1)

    ax[0].plot(all_steer)
    ax[1].plot(all_target_steer)

    all_forward_backward = np.array([list(action[1:]) for action in all_actions])
    all_target_forward_backward = np.array([list(output[1:]) for output in all_outputs])

    for i in range(2):
        _, ax = plt.subplots(2, 1)

        ax[0].plot(all_forward_backward[:, 0, i])
        ax[1].plot(all_target_forward_backward[:, 0, i])

    plt.show()

    torch.save(net.state_dict(), "resp.pth")
