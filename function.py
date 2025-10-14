import numpy as np
import argparse
from train import main
def square_sin(x):
    assert(len(x) == 2)
    return (x[0] - 3.1)**2 + 2 * (np.sin(x[1]))**2


def train(x):
    assert(len(x) == 6)
    
    lr = 10 ** -x[0]
    hidden_dim = int(round(x[1]))
    hidden_layers = int(round(x[2]))
    hidden_dropout = x[3]
    weight_decay = 10 ** -x[4]
    regularization_lambda = 10 ** -x[5]

    return main(lr, hidden_dim, hidden_layers, hidden_dropout, weight_decay, regularization_lambda)

def choose_function(name):
    if name == "square_sin":
        return square_sin
    elif name == "train":
        return train
    else:
        raise(ValueError("Function {} not defined".format(name)))


def get_input_vals(name):
    input_file = open("./data/{}_next.txt".format(name), mode='r')

    line = input_file.readline()

    return [float(part) for part in line.split(",")]

def append(name, x, y):
    file = open("./data/{}_history.txt".format(name), mode='a')

    x.append(y)

    val_list = ",".join([str(n) for n in x])
    file.write(val_list + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple script with arguments.")
    parser.add_argument("--name", type=str, help="name of function", required=True)
    args = parser.parse_args()
    name = args.name



    x = get_input_vals(name)
    black_box_function = choose_function(name)


    y = black_box_function(x)


    print("f({}) = {}".format(x, y))
    append(name, x, y)
