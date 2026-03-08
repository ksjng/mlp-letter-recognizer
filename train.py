import numpy as np

data = np.load("dataset.npz")

X = data["X"]
Y = data["Y"]

X = X.reshape(len(X),784)

input_size = 784
hidden = 128
output = 4

np.random.seed(0)

W1 = np.random.randn(input_size, hidden) * 0.01
b1 = np.zeros((1, hidden))

W2 = np.random.randn(hidden, output) * 0.01
b2 = np.zeros((1, output))

def relu(x):
    return np.maximum(0, x)

def relu_d(x):
    return (x > 0).astype(float)

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)

lr = 0.01
epochs = 1000

for epoch in range(epochs):

    z1 = X@W1 + b1
    a1 = relu(z1)

    z2 = a1@W2 + b2
    a2 = softmax(z2)

    y_one = np.eye(4)[Y]

    d2 = a2 - y_one

    dW2 = a1.T@d2 / len(X)
    db2 = np.mean(d2, axis=0, keepdims=True)

    d1 = (d2@W2.T) * relu_d(z1)

    dW1 = X.T@d1 / len(X)
    db1 = np.mean(d1, axis=0, keepdims=True)

    W2 -= lr*dW2
    b2 -= lr*db2

    W1 -= lr*dW1
    b1 -= lr*db1

    if epoch % 100 == 0:
        
        pred = np.argmax(a2, axis=1)
        acc = np.mean(pred == Y)

        print("epoch", epoch, "accuracy", round(acc*100,2), "%")

np.savez("model.npz", W1=W1, b1=b1, W2=W2, b2=b2)

print("Model zapisany")