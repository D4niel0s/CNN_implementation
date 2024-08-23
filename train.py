from layers import Fully_connected, ReLU
from funcs import cross_entropy_loss, cross_entropy_derivative
from mnist_loader import load_as_matrix_with_labels

#An example network

net = [
    Fully_connected(784, 300),
    ReLU(),
    Fully_connected(300, 100),
    ReLU(),
    Fully_connected(100, 50),
    ReLU(),
    Fully_connected(50, 10),
    ReLU()
]

n_train = 50000
n_test = 10000
x_train, y_train, x_test, y_test = load_as_matrix_with_labels(n_train, n_test)

epochs = 100
learning_rate = 0.01


def main():
    for e in epochs:
        #TODO: implement main (take inspiration from net implemented in HW3)
        batch = None
        y_true = None
        y_pred = forward_prop(batch) #Saves values in network layer objects
        SGD_step(y_pred, y_true) #Updates parameters

    #TODO: plot accuracy and loss af a function of epoch No.

def forward_prop():
    #TODO: implement this
    pass
def SGD_step():
    #TODO: implement this
    pass




if __name__ == '__main__':
    main()