import torchvision as TV
import torch
import matplotlib.pyplot as plt
import numpy as np


def nn(x, w1, w2):
    l1 = x @ w1  # A
    l1 = torch.relu(l1)  # B
    l2 = l1 @ w2
    return l2


w1 = torch.randn(784, 200, requires_grad=True)  # C
w2 = torch.randn(200, 10, requires_grad=True)

# A Matrix multiplication
# B Non-linear activation function
# C Weight (parameter) matrix, initialized randomly
# D Random input vector


mnist_data = TV.datasets.MNIST("MNIST", train=True, download=False)  # A
plt.figure(figsize=(10, 7))
plt.imshow(mnist_data.train_data[0])
plt.axis('off')

lr = 0.0001
epochs = 2500
batch_size = 1000
losses = []
lossfn = torch.nn.CrossEntropyLoss()  # B
for i in range(epochs):
    rid = np.random.randint(0, mnist_data.train_data.shape[0], size=batch_size)  # C
    x = mnist_data.train_data[rid].float().flatten(start_dim=1)  # D
    x /= x.max()  # E
    pred = nn(x, w1, w2)  # F
    target = mnist_data.train_labels[rid]  # G
    loss = lossfn(pred, target)  # H
    losses.append(loss)
    loss.backward()  # I
    with torch.no_grad():  # J
        w1 -= lr * w1.grad  # K
        w2 -= lr * w2.grad

# A Download and load the MNIST dataset
# B Set up a loss function
# C Get a set of random index values
# D Subset the data and flatten the 28x28 images into 784 vectors
# E Normalize the vector to be between 0 and 1
# F Make a prediction using the neural network
# G Get the ground-truth image labels
# H Compute the loss
# I Backpropagation
# J Do not compute gradients in this block
# K Gradient descent over the parameter matrices


plt.figure(figsize=(10, 7))
plt.xlabel("Training Time", fontsize=22)
plt.ylabel("Loss", fontsize=22)
print(type(losses[0]))
losses_for_plotting = []
for i in range(len(losses)):
    losses_for_plotting.append(losses[i].detach().numpy())
plt.plot(losses_for_plotting)
# plt.savefig("/Users/brandonbrown/Dropbox/DeepReinforcementLearning/media/Appendix/loss1.png")
plt.show()

if __name__ == '__main__':
    pass
