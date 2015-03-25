from mao_bp_ann import *
import numpy as np

# Read Data
f = open('data1', 'r')
a = np.matrix([[float(j) for j in i.split()]
               for i in f.read().strip().split('\n')])
f.close()

# Split the train and test set
train, test = split_train_test(a)

# Setup the web
webshape = [4, 10, 10, 3]
web = neu_net(
    webshape,['a','b','c','d'], [train[:, :-1], toclass(train[:, -1], classier=[1, 2, 3])])

# Train data
web.simulate(1000, step=2.)
web.simulate(1000, step=1.)
web.simulate(1000, step=.5)
web.simulate(1000, step=.1)

# Test data


def collect(x):
    return list(x.flat).index(x.max())

result = web.test(
    test[:, :-1], toclass(test[:, -1], classier=[1, 2, 3]), f=collect)
print sum(result) * 1. / len(result), len(result)
