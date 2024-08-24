from funcs import cross_entropy_loss, cross_entropy_derivative
from mnist_loader import load_as_matrix_with_labels
import matplotlib.pyplot as plt
from layers import *
import numpy as np

#An example network
network: list[Layer] = [
    Fully_connected(784, 250),
    ReLU(),
    Fully_connected(250, 100),
    ReLU(),
    Fully_connected(100, 50),
    ReLU(),
    Fully_connected(50, 10),
]

n_train = 50000
n_test = 10000
x_train, y_train, x_test, y_test = load_as_matrix_with_labels(n_train, n_test)


def main():
    epochs = 8
    tr_errs, tr_accs, ts_errs, ts_accs = train(x_train, y_train, x_test, y_test, epochs=epochs, learning_rate=0.1, batch_size=100)
    
    ep_range = np.linspace(1,epochs,epochs)

    plt.figure(1)
    plt.title("NN error as a function of epoch")
    plt.plot(ep_range, tr_errs, color="red",label="training error")
    plt.plot(ep_range, ts_errs, color="blue",label="test error")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend()

    plt.figure(2)
    plt.title("NN accuracy as a function of epoch")
    plt.plot(ep_range, tr_accs, color="green",label="training accuracy")
    plt.plot(ep_range, ts_accs, color="orange",label="test accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()


    random_images = np.random.choice(n_test, 20)
    
    #Plot all Principle components
    fig, axs = plt.subplots(nrows=4, ncols=1, constrained_layout=True)
    fig.suptitle('Random classifications')
    for ax in axs:
        ax.remove()

    gridspec = axs[0].get_subplotspec().get_gridspec()
    subfigs = [fig.add_subfigure(gs) for gs in gridspec]

    for row, subfig in enumerate(subfigs):
        axs = subfig.subplots(nrows=1, ncols=5)
        for col, ax in enumerate(axs):
            img = x_test[:, random_images[row*5 + col]]
            ax.imshow(img.reshape((28, 28)), cmap=plt.cm.gray)
            ax.set_title(f'pred: {cp.argmax(forward_prop(img.reshape((784,1))))}')
            ax.axis("off")


    plt.show()


def train(x_train, y_train, x_test, y_test, epochs, learning_rate, batch_size):
    x_train = cp.array(x_train)
    y_train = cp.array(y_train)
    x_test = cp.array(x_test)
    y_test = cp.array(y_test)

    epoch_errs = []
    epoch_accs = []
    epoch_test_errs = []
    epoch_test_accs = []
    for e in range(epochs):
        errs = []
        accs = []
        for i in range(0, x_train.shape[1], batch_size):
            batch = x_train[:, i:i+batch_size]
            y_true = y_train[i:i+batch_size]
            
            y_pred = forward_prop(batch) #Saves values in network layer objects
            back_prop(y_pred, y_true, learning_rate, batch_size) #Updates parameters

            # Append batch loss
            train_loss = cross_entropy_loss(y_pred, y_true, network[len(network)-1].W.shape[0])
            errs.append(train_loss)

            # Append batch accuracy
            preds = cp.argmax(y_pred, axis=0)
            train_acc = calculate_accuracy(preds, y_true, batch_size)
            accs.append(train_acc)

        average_train_cost = cp.mean(cp.array(errs))
        average_train_acc = cp.mean(cp.array(accs))

        # Convert to floats from cupy objects
        epoch_errs.append(float(average_train_cost))
        epoch_accs.append(float(average_train_acc))

        # Evaluate on test set
        test_pred = forward_prop(x_test)
        test_loss = cross_entropy_loss(test_pred, y_test, network[len(network)-1].W.shape[0])
        preds = cp.argmax(test_pred, axis=0)
        test_acc = calculate_accuracy(preds, y_test, len(y_test))
        print(f"Epoch: {e + 1}, Test loss: {test_loss:.20f}, Test accuracy: {test_acc:.20f}")

        # Convert to floats from cupy objects
        epoch_test_errs.append(float(test_loss))
        epoch_test_accs.append(float(test_acc))

    return epoch_errs, epoch_accs, epoch_test_errs, epoch_test_accs


def forward_prop(batch):
    X = cp.array(batch)
    for layer in network:
        X = layer.forward(X)    
    return X

def back_prop(y_hat, y, learning_rate, batch_size):
    upstream = cross_entropy_derivative(y_hat, y, network[len(network)-1].W.shape[0])
    for layer in reversed(network):
        upstream = layer.backward(upstream, learning_rate, batch_size)
    

def calculate_accuracy(y_pred, y_true, batch_size):
    return cp.sum(y_pred == y_true) / batch_size


if __name__ == '__main__':
    main()