import threading
from threading import Thread
import numpy as np
from queue import Queue
from matplotlib import pyplot as plt

# Define the loss function `ell`
def ell(w, x, y):
    return (1 - 1 / (1 + np.exp(-w @ x * y))) ** 2


# Define the gradient of the loss function `ell`
def ell_grad(w, x, y):
    # print(w.shape, x.shape)
    b = x @ w * y
    b = np.clip(b, -700, 700)
    t = (-2 * y * np.exp(b) / ((1 + np.exp(b)) ** 3)) @ x
    return t


# Define the objective function `f`
def f(x, A, Y):
    return np.mean([ell(a, x, y) for a, y in zip(A, Y)])


# Define the RandK
def randk(x, k):
    d = len(x)
    s = np.random.choice(d, k)
    x_compressed = np.zeros(d)
    x_compressed[s] = (d / k) * x[s]
    return x_compressed


def task(X_train_node, y_train_node):
    # Each thread has its own data batch
    result_prev = np.zeros(n)
    grad_prev = np.zeros(n)
    while True:
        with shared_g.condition:
            # shared_g.condition.wait()
            w = shared_g.data
            # Compute the gradient
            # print(X_train_node.shape, y_train_node.shape)
            grad = ell_grad(w, X_train_node, y_train_node)
            if np.random.rand() < p:
                result = grad
            else:
                result = result_prev + randk(grad - grad_prev, k)
            # Send the gradient to the server
            shared_quantized_g.put(result)
            result_prev = result.copy()
            grad_prev = grad.copy()


# Load the dataset
from sklearn.datasets import load_svmlight_file

dataset = "../data/mushrooms.txt"
data = load_svmlight_file(dataset)
X, y = data[0].toarray(), data[1]
y = 2 * y - 3
X = X[:8120]
y = y[:8120]
# Split the dataset into five equal parts
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
X_train_split = np.array_split(X_train, 5)
y_train_split = np.array_split(y_train, 5)
# Init parameters
n = X_train.shape[1]
k = 1
w = np.random.randn(n)
omega = n / k - 1
p = 1 / (omega + 1)


class SharedData:
    def __init__(self):
        self.data = None
        self.condition = threading.Condition()


shared_g = SharedData()
shared_quantized_g = Queue()
threads = []
convergence_criteria = []
with shared_g.condition:
    shared_g.data = w
    shared_g.condition.notify_all()
for i in range(5):
    t = Thread(target=task, args=(X_train_split[i], y_train_split[i],), name=f"Node-{i}")
    t.start()
    threads.append(t)
for iter in range(200):
    # Send the parameters to the nodes
    with shared_g.condition:
        shared_g.data = w
        shared_g.condition.notify_all()
        print(f"Server: send a new w to the nodes, Iteration # {iter}")
    # Get the gradients from the nodes
    quantized_grads = []
    for i in range(5):
        quantized_grads.append(shared_quantized_g.get())
    # Compute the average gradient
    grad = np.mean(quantized_grads, axis=0)
    # Update the parameters
    w = w - 0.01 * grad
    convergence_criteria.append(np.linalg.norm(ell_grad(w, X_train, y_train))**2)

pred = 1 / (1+ np.exp(-(X_test @ w)))
pred[pred > 0.5] = 1
pred[pred <= 0.5] = -1
print(np.mean(pred == y_test))
plt1 = plt.figure()
plt.plot(range(len(convergence_criteria)), convergence_criteria)
plt.xlabel("Iteration")
plt.ylabel("Convergence Criteria")
plt.title("Convergence Criteria vs Iteration")
plt1.savefig("convergence_criteria.png")
plt2 = plt.figure()
plt.plot(range(len(convergence_criteria))[-50:], convergence_criteria[-50:])
plt.xlabel("Iteration")
plt.ylabel("Convergence Criteria")
plt.title("Convergence Criteria vs Iteration")
plt2.savefig("convergence_criteria_last50.png")
for t in threads:
    t.join()