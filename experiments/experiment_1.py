import threading
import time
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

dataset = "./data/mushrooms.txt"
data = load_svmlight_file(dataset)
X, y = data[0].toarray(), data[1]
y = 2 * y - 3
X = X[:8120]
y = y[:8120]
# Split the dataset into five equal parts
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
X_train_split = np.array_split(X_train, 12)
y_train_split = np.array_split(y_train, 12)
# Init parameters
n = X_train.shape[1]
k = int(0.1*n)
w_0 = np.zeros(n)
omega = n / k - 1
p = 1 / (omega + 1)
gamma = 4 * (1+ np.sqrt(((1-p)*omega)/(p*12)))**(-1)
convergences = []
class SharedData:
    def __init__(self):
        self.data = None
        self.condition = threading.Condition()


shared_g = SharedData()
shared_quantized_g = Queue()
threads = []
convergence_criteria_marina = []
w = w_0.copy()
with shared_g.condition:
    shared_g.data = w.copy()
    shared_g.condition.notify_all()
times_marina = []
for iter in range(500):
    iteration_start= time.time()
    for i in range(12):
        t = Thread(target=task, args=(X_train_split[i], y_train_split[i],), name=f"Node-{i}")
        t.start()
        threads.append(t)
    # Send the parameters to the nodes
    with shared_g.condition:
        shared_g.data = w
        shared_g.condition.notify_all()
        print(f"Server: send a new w to the nodes, Iteration # {iter}")
    # Get the gradients from the nodes
    quantized_grads = []
    for i in range(12):
        quantized_grads.append(shared_quantized_g.get())
    # Compute the average gradient
    grad = np.mean(quantized_grads, axis=0)
    # Update the parameters
    w = w - gamma * grad
    convergence_criteria_marina.append(np.linalg.norm(grad))
    times_marina.append(time.time() - iteration_start)
    for t in threads:
        t.join()
convergences.append(convergence_criteria_marina)
k = int(0.2*n)
omega = n / k - 1
p = 1 / (omega + 1)
gamma = 4 * (1+ np.sqrt(((1-p)*omega)/(p*12)))**(-1)
shared_g = SharedData()
shared_quantized_g = Queue()
threads = []
convergence_criteria_marina = []
w = w_0.copy()
with shared_g.condition:
    shared_g.data = w.copy()
    shared_g.condition.notify_all()
times_marina = []
for iter in range(500):
    iteration_start= time.time()
    for i in range(12):
        t = Thread(target=task, args=(X_train_split[i], y_train_split[i],), name=f"Node-{i}")
        t.start()
        threads.append(t)
    # Send the parameters to the nodes
    with shared_g.condition:
        shared_g.data = w
        shared_g.condition.notify_all()
        print(f"Server: send a new w to the nodes, Iteration # {iter}")
    # Get the gradients from the nodes
    quantized_grads = []
    for i in range(12):
        quantized_grads.append(shared_quantized_g.get())
    # Compute the average gradient
    grad = np.mean(quantized_grads, axis=0)
    # Update the parameters
    w = w - gamma * grad
    convergence_criteria_marina.append(np.linalg.norm(grad))
    times_marina.append(time.time() - iteration_start)
    for t in threads:
        t.join()
convergences.append(convergence_criteria_marina)
k = int(0.3*n)
omega = n / k - 1
p = 1 / (omega + 1)
gamma = 4 * (1+ np.sqrt(((1-p)*omega)/(p*12)))**(-1)
shared_g = SharedData()
shared_quantized_g = Queue()
threads = []
convergence_criteria_marina = []
w = w_0.copy()
with shared_g.condition:
    shared_g.data = w.copy()
    shared_g.condition.notify_all()
times_marina = []
for iter in range(500):
    iteration_start= time.time()
    for i in range(12):
        t = Thread(target=task, args=(X_train_split[i], y_train_split[i],), name=f"Node-{i}")
        t.start()
        threads.append(t)
    # Send the parameters to the nodes
    with shared_g.condition:
        shared_g.data = w
        shared_g.condition.notify_all()
        print(f"Server: send a new w to the nodes, Iteration # {iter}")
    # Get the gradients from the nodes
    quantized_grads = []
    for i in range(12):
        quantized_grads.append(shared_quantized_g.get())
    # Compute the average gradient
    grad = np.mean(quantized_grads, axis=0)
    # Update the parameters
    w = w - gamma * grad
    convergence_criteria_marina.append(np.linalg.norm(grad))
    times_marina.append(time.time() - iteration_start)
    for t in threads:
        t.join()
convergences.append(convergence_criteria_marina)
k = int(0.4*n)
omega = n / k - 1
p = 1 / (omega + 1)
gamma = 4 * (1+ np.sqrt(((1-p)*omega)/(p*12)))**(-1)
shared_g = SharedData()
shared_quantized_g = Queue()
threads = []
convergence_criteria_marina = []
w = w_0.copy()
with shared_g.condition:
    shared_g.data = w.copy()
    shared_g.condition.notify_all()
times_marina = []
for iter in range(500):
    iteration_start= time.time()
    for i in range(12):
        t = Thread(target=task, args=(X_train_split[i], y_train_split[i],), name=f"Node-{i}")
        t.start()
        threads.append(t)
    # Send the parameters to the nodes
    with shared_g.condition:
        shared_g.data = w
        shared_g.condition.notify_all()
        print(f"Server: send a new w to the nodes, Iteration # {iter}")
    # Get the gradients from the nodes
    quantized_grads = []
    for i in range(12):
        quantized_grads.append(shared_quantized_g.get())
    # Compute the average gradient
    grad = np.mean(quantized_grads, axis=0)
    # Update the parameters
    w = w - gamma * grad
    convergence_criteria_marina.append(np.linalg.norm(grad))
    times_marina.append(time.time() - iteration_start)
    for t in threads:
        t.join()
convergences.append(convergence_criteria_marina)

k = int(0.5*n)
omega = n / k - 1
p = 1 / (omega + 1)
gamma = 4 * (1+ np.sqrt(((1-p)*omega)/(p*12)))**(-1)
shared_g = SharedData()
shared_quantized_g = Queue()
threads = []
convergence_criteria_marina = []
w = w_0.copy()
with shared_g.condition:
    shared_g.data = w.copy()
    shared_g.condition.notify_all()
times_marina = []
for iter in range(500):
    iteration_start= time.time()
    for i in range(12):
        t = Thread(target=task, args=(X_train_split[i], y_train_split[i],), name=f"Node-{i}")
        t.start()
        threads.append(t)
    # Send the parameters to the nodes
    with shared_g.condition:
        shared_g.data = w
        shared_g.condition.notify_all()
        print(f"Server: send a new w to the nodes, Iteration # {iter}")
    # Get the gradients from the nodes
    quantized_grads = []
    for i in range(12):
        quantized_grads.append(shared_quantized_g.get())
    # Compute the average gradient
    grad = np.mean(quantized_grads, axis=0)
    # Update the parameters
    w = w - gamma * grad
    convergence_criteria_marina.append(np.linalg.norm(grad))
    times_marina.append(time.time() - iteration_start)
    for t in threads:
        t.join()
convergences.append(convergence_criteria_marina)
# pred = 1 / (1 + np.exp(-(X_test @ w)))
# pred[pred > 0.5] = 1
# pred[pred <= 0.5] = -1
# print("Accuracy for MARINA: ", np.mean(pred == y_test))
#
#
# def task(X_train_node, y_train_node, shared_g):
#     # Diana
#     global h
#
#     g_local = ell_grad(shared_g, X_train_node, y_train_node)
#     delta = g_local - h[i]
#     quantized_delta = randk(delta, k)
#     result = h[i] + quantized_delta
#     shared_quantized_g.put(result)
#     h[i] = h[i] + alpha * quantized_delta
#
#
# w = w_0.copy()
# shared_quantized_g = Queue()
# threads = []
# convergence_criteria_diana = []
# alpha = 0.1
# gamma = 0.5
# h = [w_0.copy() for _ in range(12)]
# times_diana = []
# for iter in range(200):
#     iteration_start = time.time()
#     print(f"{iter + 1}/200")
#     for i in range(12):
#         t = Thread(target=task, args=(X_train_split[i], y_train_split[i], w,), name=f"Node-{i}")
#         t.start()
#         threads.append(t)
#     # Get the gradients from the nodes
#     quantized_grads = []
#     for i in range(12):
#         quantized_grads.append(shared_quantized_g.get())
#     # Compute the average gradient
#     grad = np.mean(quantized_grads, axis=0)
#     # Update the parameters
#     w = w - gamma * grad
#     convergence_criteria_diana.append(np.linalg.norm(grad))
#     for t in threads:
#         t.join()
#     times_diana.append(time.time()-iteration_start)
# pred = 1 / (1 + np.exp(-(X_test @ w)))
# pred[pred > 0.5] = 1
# pred[pred <= 0.5] = -1
# print("Accuracy for DIANA: ", np.mean(pred == y_test))
#
# plt1 = plt.figure()
# plt.plot(range(len(convergence_criteria_marina)), convergence_criteria_marina, label="MARINA")
# plt.plot(range(len(convergence_criteria_diana)), convergence_criteria_diana, label="DIANA")
# plt.xlabel("Iteration")
# plt.ylabel("Convergence Criteria")
# plt.title("Marina vs Diana")
# plt.legend()
# plt.show()
# plt1.savefig("convergence_criteria.png")
# plt2 = plt.figure()
# plt.plot(range(len(convergence_criteria_marina))[-20:], convergence_criteria_marina[-20:], label="MARINA")
# plt.plot(range(len(convergence_criteria_diana))[-20:], convergence_criteria_diana[-20:], label="DIANA")
# plt.xlabel("Iteration")
# plt.ylabel("Convergence Criteria")
# plt.title("Marina vs Diana(Last 20 iterations)")
# plt.legend()
# plt.show()
# plt2.savefig("convergence_criteria_last20.png")
cumulative_times_marina = [sum(times_marina[:i+1]) for i in range(len(times_marina))]
# cumulative_times_diana = [sum(times_diana[:i+1]) for i in range(len(times_diana))]
plt3 = plt.figure()
plt.yscale('log')
plt.plot(cumulative_times_marina, convergences[0], label="Sparsity 0.1")
plt.plot(cumulative_times_marina, convergences[1], label="Sparsity 0.2")
plt.plot(cumulative_times_marina, convergences[2], label="Sparsity 0.3")
plt.plot(cumulative_times_marina, convergences[3], label="Sparsity 0.4")
plt.plot(cumulative_times_marina, convergences[4], label="Sparsity 0.5")
# plt.plot(cumulative_times_diana, convergence_criteria_diana, label="DIANA")
#
plt.xlabel("Time")
plt.ylabel("Convergence Criteria")
plt.title("Gradient Norms vs Time in MARINA Algorithm for Various Sparsity Levels")
plt.legend()
# plt.show()
plt3.savefig("convergence_criteria_time.png")