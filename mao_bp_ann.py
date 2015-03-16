import numpy as np


class neu_web(object):

    '''This is a neu web class to setup a web and perform the training.'''

    def __init__(self, webshape=None, data=None):
        self.inputdata = np.zeros((0))
        self.outputdata = np.zeros((0))
        self.trans = []
        self.webshape = []
        if webshape:
            self.set_webshape(webshape)
        if data:
            self.set_input(data[0])
            self.set_output(data[1])

    def set_input(self, inputdata=None):
        '''Set the input data for the web.'''
        if inputdata == None:
            print '### No input data provided. Ignored.'
        if isinstance(inputdata, (np.ndarray, list)):
            try:
                inputdata = np.matrix(inputdata)
            except:
                raise ValueError("Input data format mistake.")
        if isinstance(inputdata, np.matrix):
            if inputdata.shape[1] != self.webshape[0]:
                raise ValueError("The size doesn't fit the webshape[0].")
            self.inputdata = inputdata
        if self.outputdata != np.zeros((0)):
            if self.inputdata.shape[0] != self.outputdata.shape[0]:
                raise ValueError(
                    "The sizes of Input and Output are different.")

    def set_output(self, outputdata=None):
        '''Set the output data for the web.'''
        if outputdata == None:
            print '### No output data provided. Ignored.'
        if isinstance(outputdata, (np.ndarray, list)):
            try:
                outputdata = np.matrix(outputdata)
            except:
                raise ValueError("Output data format mistake.")
        if isinstance(outputdata, np.matrix):
            if outputdata.shape[1] != self.webshape[-1]:
                raise ValueError("The size doesn't fit the webshape[-1].")
            self.outputdata = outputdata
        if self.inputdata != np.zeros((0)):
            if self.inputdata.shape[0] != self.outputdata.shape[0]:
                raise ValueError(
                    "The sizes of Input and Output are different.")

    def set_webshape(self, webshape):
        '''Set the web shape data for the web.'''
        if not webshape:
            print '### No web shape data provided. Ignored.'
        self.webshape = webshape
        if self.inputdata:
            if self.inputdata.shape[0] != self.webshape[0]:
                raise ValueError(
                    "The web shape and the inputdata doesn't fit.")
        if self.outputdata:
            if self.outputdata.sape[0] != self.webshape[-1]:
                raise ValueError(
                    "The web shape and the outputdata doesn't fit.")
        self.trans = []
        if self.webshape:
            for i in range(len(webshape) - 1):
                self.trans += [np.matrix(np.random.random((webshape[i] + 1, webshape[i + 1])) * (0.1) - 0.05)]
                # self.trans += [np.matrix(np.random.random((webshape[i] + 1, webshape[i + 1])))]

    def simulate(self, times, step=0.1):
        '''Train the data.'''
        if self.inputdata.shape[0] != self.outputdata.shape[0]:
            raise ValueError("The sizes of Input and Output are different.")
        temptimes = times
        while temptimes > self.inputdata.shape[0]:
            temptimes -= self.inputdata.shape[0]
            templist = range(self.inputdata.shape[0])
            np.random.shuffle(templist)
            for i in templist:
                self.__simulate_point(
                    self.inputdata[templist[i]], self.outputdata[templist[i]], step=step)
        templist = range(self.inputdata.shape[0])
        np.random.shuffle(templist)
        for i in templist[:temptimes]:
            self.__simulate_point(
                self.inputdata[templist[i]], self.outputdata[templist[i]], step=step)

    def __simulate_point(self, inputd, outputd, step=0.1):
        '''Train one data point.'''
        outputd1 = np.array(outputd)
        layer = len(self.webshape) - 1
        temp = inputd
        save = [np.array(inputd)]
        for i in range(layer):
            temp = np.concatenate((temp, np.ones((1, 1))), axis=1)
            temp = temp.dot(self.trans[i])
            temp= 1./(1.+np.exp(-temp))
            save = save + [np.array(temp)]
        wucha = [save[-1] * (1 - save[-1]) * (outputd1 - save[-1])]
        for i in reversed(range(layer)):
            wucha = [save[i] * (1 - save[i]) * np.array(wucha[0].dot(self.trans[i][:-1].T))] + wucha
        for i in range(layer):
            self.trans[i] = self.trans[i] + (np.concatenate((save[i], np.ones((1, 1))), axis=1).T.dot(wucha[i + 1])) * step

    def test(self, inputtest, outputtest, f):
        if inputtest.shape[0] != outputtest.shape[0]:
            raise ValueError("Size not same.")
        mark = []
        for i in range(inputtest.shape[0]):
            layer = len(self.webshape) - 1
            temp = inputtest[i]
            for j in range(layer):
                temp = np.concatenate((temp, np.ones((1, 1))), axis=1)
                temp = temp.dot(self.trans[j])
            mark += [f(temp) == f(outputtest[i])]
        return mark


def keepsize(n):
    '''return the sample size for a data set n.'''
    return int(n - round((1 - 1. / n)**n * n))


def split_train_test(a, keep=None):
    '''Split the data to train set and test set by random sample.'''
    n = a.shape[0]
    if keep == None:
        keep = keepsize(n)
    keeplist = range(n)
    removelist = []
    while len(keeplist) > keep:
        removelist.append(
            keeplist.pop(np.random.random_integers(0, len(keeplist) - 1)))
    np.random.shuffle(removelist)
    np.random.shuffle(keeplist)
    return a[keeplist], a[removelist]


def toclass(a, classier=None):
    '''convert a matrix to a matrix split every class.'''
    if classier != None:
        setitem = sorted(list(set([i.item() for i in a])))
    else:
        setitem = classier
    result = np.matrix(np.zeros((a.shape[0], len(setitem))))
    for i in range(len(setitem)):
        result[:, i] = (a == setitem[i])
    return result

# Read Data
f = open('data', 'r')
a = np.matrix([[float(j) for j in i.split()]
               for i in f.read().strip().split('\n')])
f.close()

# Split the train and test set
train, test = split_train_test(a)

# Setup the web
webshape = [4, 10, 10, 3]
web = neu_web(
    webshape, [train[:, :-1], toclass(train[:, -1], classier=[1, 2, 3])])

# Train data
web.simulate(1000,step=2.)
web.simulate(1000,step=1.)
web.simulate(1000,step=.5)
web.simulate(1000,step=.1)
# print web.trans

# Test data


def collect(x):
    return list(x.flat).index(x.max())
result = web.test(
    test[:, :-1], toclass(test[:, -1], classier=[1, 2, 3]), f=collect)
print sum(result) * 1. / len(result),len(result)
