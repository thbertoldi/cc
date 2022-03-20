import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

SHOW_IMAGES = False
BATCH_SIZE = 10


def load_data():
    actions = np.load("actions.npy", allow_pickle=True)
    actions = actions.astype(np.longlong)
    # actions = actions.astype(np.float32)

    states = np.load("states.npy", allow_pickle=True)
    # states = states.astype(np.float32)
    print(f"Tenho {len(actions)} ações")
    print(f"Tenho {len(states)} estados")
    if SHOW_IMAGES:
        _, ax = plt.subplots(3, 1)
        for i in range(3):
            ax[i].plot(actions[:, i])
        plt.show()
        plt.imshow(states[300, :, :, :])
        plt.show()
        plt.imshow(states[600, :, :, :])
        plt.show()
    # return (torch.from_numpy(actions.astype(np.F)), states)
    return actions.astype(np.float32), states


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 8)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = torch.reshape(x, shape=(BATCH_SIZE, 4, 2))
        x = F.softmax(x, dim=2)
        # print("antes", x.size())
        #print(x.size())

        return x

def toCrossFormat(x, n_classes):
    y = np.zeros(shape=(4, n_classes))
    for i in range(4):
        y[i, int(round(x[i]))] = 1
    return y

# x = [0, 1, 1], y = [[1, 0], [0, 1], [0, 1]]


if __name__ == "__main__":
    metadata = load_data()
    print(len(metadata))
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.5)
    torch.autograd.set_detect_anomaly(True)
    all_actions = []
    all_outputs = []
    losses = []
    for epoch in range(1):
        running_loss = 0.0
        for i in range(len(metadata[0])-BATCH_SIZE-1):
            action, state = metadata[0][i:i+BATCH_SIZE], metadata[1][i:i+BATCH_SIZE]
            # action, state = metadata[0:1][i], metadata[1:1+1][i]

            state = np.transpose(state)
            state = np.reshape(state, newshape=(BATCH_SIZE, 3, 96, 96))

            state = np.array(state)
            optimizer.zero_grad()
            output = net(torch.Tensor(state))
            # print("Output is", output.size())
            # print(output)
            old_action = action.copy()
            action = []
            for i in range(BATCH_SIZE):
                i_action = [
                    1 if old_action[i, 0] > 0 else 0,
                    1 if old_action[i, 0] < 0 else 0,
                    old_action[i, 1],
                    old_action[i, 2],
                ]
                i_action = toCrossFormat(i_action, 2)
                
                action.append(i_action)

            action = np.array(action)
            # action = torch.Tensor(action, requires_grad=True)
            action = torch.tensor(action, requires_grad=True)
            #print(action.size())

            # output = torch.reshape(output, (1, *output.size()))
            # print(output.max(-2)[1])
            # print(action.type())
            # print(output.type())


            loss = criterion(action.double(), output.double())
            loss.backward()
            optimizer.step()
            running_loss = loss.item()

            all_actions.append(action[0:1].detach().numpy())
            all_outputs.append(output[0:1].detach().numpy())
            losses.append(running_loss)
            if i % 10 == 9:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss:.3f}")
                print(f"Action: {torch.argmax(action[0, 0])}, output: {torch.argmax(output[0, 0])}")
                running_loss = 0.0


    l = np.array(all_actions)
    ll = np.array(all_outputs)

    plt.show()
    plt.plot(losses)

    for i in range(4):
        _, ax = plt.subplots(2, 1)

        print(l.shape)
        ax[0].plot(l[:, 0, i])
        ax[1].plot(ll[:, 0, i])
        plt.show()

    torch.save(net.state_dict(), "resp.pth")


