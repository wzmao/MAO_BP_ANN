from mao_bp_ann import *

# Read Data
f = open('data2', 'r')
a = np.matrix([[float(j) for j in i.split()]
               for i in f.read().strip().split('\n')])
f.close()

# Split the train and test set
train, test = split_train_test(a)

# Setup the web
webshape = [2, 2]
web = neu_web(
    webshape, [train[:, :-1], toclass(train[:, -1], classier=[1, 2])])

# Train data
web.simulate(1000, step=.1)

# Test data


def collect(x):
    return list(x.flat).index(x.max())

result = web.test(
    test[:, :-1], toclass(test[:, -1], classier=[1, 2]), f=collect)
print sum(result) * 1. / len(result), len(result)
