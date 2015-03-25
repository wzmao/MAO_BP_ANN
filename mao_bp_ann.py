import numpy as np
from time import time

__all__ = ['neu_net', 'keepsize', 'split_train_test', 'toclass']


class neu_net(object):

    '''This is a neu web class to setup a web and perform the training.
    The content is based on the Artificial Neural Networks.
    The training is performed by the BP algorithm.'''

    def __init__(self, webshape=None, label=None, data=None, **kwargs):
        '''the init function of `neu_net` class. webshape, label, data
        could be assigned by options.'''

        self.inputdata = np.zeros((0), dtype=np.float64)
        self.outputdata = np.zeros((0), dtype=np.float64)
        self.label = []
        self.trans = np.zeros((0), dtype=object)
        self.webshape = np.zeros((0), dtype=np.int32)

        self._clockdict = {}

        if webshape!=None:
            self.set_webshape(webshape, **kwargs)
        if label!=None:
            self.set_label(label, **kwargs)
        if data!=None and len(data) == 2:
            self.set_input(data[0], **kwargs)
            self.set_output(data[1], **kwargs)

    def _tic(self, key, **kwargs):
        '''Add a start time for some key event.'''
        self._clockdict[key] = time()

    def _toc(self, key, **kwargs):
        '''Remove the key and return the time used for the key event.'''
        tic = self._clockdict.pop(key, None)
        if tic:
            return time() - tic
        else:
            return None

    def _info(self, info, flag=1, **kwargs):
        '''Print the infomation. Several type could be choosen.
        flag:
            1 for normal information.
            2 for warning.
            3 for ValueError.(Not suggested)'''

        if flag == 1 or flag == 'normal':
            print '*** ' + info.lstrip()
        elif flag == 2 or flag == 'warning':
            print '###### ' + info.lstrip() + ' ######'
        elif flag == 3 or flag == 'error':
            raise ValueError(info)

    def _clear_data(self, keep=[], **kwargs):
        '''Delete all data in this class.
        You could use keep to keep the variables you want to keep.'''
        if not ('webshape' in keep or 'shape' in keep):
            self.webshape = np.zeros((0), dtype=np.int32)
        if not ('input' in keep or 'inputdata' in keep):
            self.inputdata = np.zeros((0), dtype=np.float64)
        if not ('output' in keep or 'outputdata' in keep):
            self.outputdata = np.zeros((0), dtype=np.float64)
        if not ('label' in keep):
            self.label = []
        if not ('trans' in keep):
            self.trans = np.zeros((0), dtype=object)
        self._clockdict = {}

    def _check(self, **kwargs):
        '''Check all data compatible to each other.
        Check only the data for calculation.'''
        result = 1
        # Type Check
        # webshape
        if not isinstance(self.webshape, np.ndarray):
            self._info("`webshape` type", 2)
            result = 0
        else:
            if self.webshape.dtype != np.int32:
                self._info("`webshape` dtype", 2)
                result = 0
        # inputdata
        if not isinstance(self.inputdata, np.ndarray):
            self._info("`inputdata` type", 2)
            result = 0
        else:
            if self.inputdata.dtype != np.float64:
                self._info("`inputdata` dtype", 2)
                result = 0
        # outputdata
        if not isinstance(self.outputdata, np.ndarray):
            self._info("`outputdata` type", 2)
            result = 0
        else:
            if self.outputdata.dtype != np.float64:
                self._info("`outputdata` dtype", 2)
                result = 0
        # trans
        if not isinstance(self.trans, np.ndarray):
            self._info("`trans` type", 2)
            result = 0
        else:
            if self.trans.dtype != np.object:
                self._info("`trans` dtype", 2)
                result = 0
        # Shape Check
        if len(self.webshape.shape) != 1:
            self._info("`webshape` shape")
            result = 0
        if len(self.inputdata.shape) != 2:
            self._info("`inputdata` shape")
            result = 0
        if len(self.outputdata.shape) != 2:
            self._info("`outputdata` shape")
            result = 0
        if len(self.trans.shape) != 1:
            self._info("`trans` shape")
            result = 0
        # Size Check
        # webshape
        if len(self.webshape) < 2:
            self._info("`webshape` must has more than 2 layers", 2)
            result = 0
        if self.inputdata.shape[1] != self.webshape[0]:
            self._info("`inputdata` doesn't fit webshape[0]", 2)
            result = 0
        if self.outputdata.shape[1] != self.webshape[-1]:
            self._info("`outputdata` doesn't fit webshape[-1]", 2)
            result = 0
        if self.inputdata.shape[0] != self.outputdata.shape[0]:
            self._info("`inputdata` doesn't fit `outputdata`", 2)
            result = 0
        for i in range(len(self.webshape) - 1):
            if self.trans[i].shape != (self.webshape[i] + 1, self.webshape[i + 1]):
                self._info("`trans`[{0}] size wrong".format(i), 2)
                result = 0
        return bool(result)

    def set_webshape(self, webshape=[], **kwargs):
        '''Set the web shape (nodes for each layer) for the network.'''
        if len(webshape)==0:
            self._info('No web shape data provided. Ignored.', 2)
            return None
        if kwargs.get('force'):
            self.webshape = np.array(webshape, dtype=np.int32)
            return None
        webshape = np.array(webshape, dtype=np.int32)
        if self.webshape != webshape:
            if self.inputdata != np.zeros((0), dtype=np.float64):
                if self.inputdata.shape[0] != webshape[0]:
                    raise ValueError(
                        "The web shape and the inputdata doesn't fit.")
            if self.outputdata != np.zeros((0), dtype=np.float64):
                if self.outputdata.shape[0] != webshape[-1]:
                    raise ValueError(
                        "The web shape and the outputdata doesn't fit.")
            self.webshape = webshape
            self.trans = np.zeros((webshape.shape[0]), dtype=object)
            for i in range(len(webshape) - 1):
                self.trans[i] = np.matrix(
                    np.random.random((webshape[i] + 1, webshape[i + 1])) * 0.1 - 0.05)
        else:
            self._info('webshape unchanged, the trans also unchanged.', 1)

    def set_label(self, label=[], **kwargs):
        '''Set the labels for the inputs.'''
        if len(label) == 0:
            self._info('No label provided. Ignored.', 2)
            return None
        if kwargs.get("force"):
            self.label = label
            return None
        if not isinstance(label, list):
            label = list(label)
        if self.webshape != np.zeros((0), dtype=np.int32):
            if self.webshape[0] != len(label):
                raise ValueError(
                    "The length of the label list doesn't fit the webshape[0].")
            else:
                self.label = label
        else:
            self.label = label

    def set_input(self, inputdata=None, **kwargs):
        '''Set the input data for the network.'''
        if inputdata == None:
            self._info('No input data provided. Ignored.', 2)
            return None
        if isinstance(inputdata, (np.ndarray, list)):
            try:
                inputdata = np.matrix(inputdata, dtype=np.float64)
            except:
                raise ValueError("Input data format mistake.")
        if isinstance(inputdata, np.matrix):
            if not inputdata.dtype == np.float64:
                inputdata = np.matrix(inputdata, dtype=np.float64)
            if kwargs.get('force'):
                self.inputdata = inputdata
                return None
        else:
            raise ValueError(
                "Please provide a list np.ndarray or a np.matrix.")
        if self.webshape != np.zeros((0), dtype=np.int32):
            if inputdata.shape[1] != self.webshape[0]:
                raise ValueError("The size doesn't fit the webshape[0].")
        if self.outputdata != np.zeros((0), dtype=np.float64):
            if self.inputdata.shape[0] != outputdata.shape[0]:
                raise ValueError(
                    "The sizes of Input and Output are different.")
        self.inputdata = inputdata

    def set_output(self, outputdata=None, **kwargs):
        '''Set the output data for the network.'''
        if outputdata == None:
            self._info('No output data provided. Ignored.', 2)
            return None
        if isinstance(outputdata, (np.ndarray, list)):
            try:
                outputdata = np.matrix(outputdata, dtype=np.float64)
            except:
                raise ValueError("Output data format mistake.")
        if isinstance(outputdata, np.matrix):
            if not outputdata.dtype == np.float64:
                outputdata = np.matrix(outputdata, dtype=np.float64)
            if kwargs.get('force'):
                self.outputdata = outputdata
                return None
        else:
            raise ValueError(
                "Please provide a list np.ndarray or a np.matrix.")
        if self.webshape != np.zeros((0), dtype=np.int32):
            if outputdata.shape[1] != self.webshape[-1]:
                raise ValueError("The size doesn't fit the webshape[-1].")
        if self.inputdata != np.zeros((0), dtype=np.float64):
            if self.inputdata.shape[0] != outputdata.shape[0]:
                raise ValueError(
                    "The sizes of Input and Output are different.")
        self.outputdata = outputdata

    def simulate(self, times, step=0.1, **kwargs):
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

    def __simulate_point(self, inputd, outputd, step=0.1, **kwargs):
        '''Train one data point.'''
        outputd1 = np.array(outputd)
        layer = len(self.webshape) - 1
        temp = inputd
        save = [np.array(inputd)]
        for i in range(layer):
            temp = np.concatenate((temp, np.ones((1, 1))), axis=1)
            temp = temp.dot(self.trans[i])
            temp = 1. / (1. + np.exp(-temp))
            save = save + [np.array(temp)]
        wucha = [save[-1] * (1 - save[-1]) * (outputd1 - save[-1])]
        for i in reversed(range(layer)):
            wucha = [
                save[i] * (1 - save[i]) * np.array(wucha[0].dot(self.trans[i][:-1].T))] + wucha
        for i in range(layer):
            self.trans[i] = self.trans[
                i] + (np.concatenate((save[i], np.ones((1, 1))), axis=1).T.dot(wucha[i + 1])) * step
        # return ((outputd1 - save[-1])*(outputd1 - save[-1])).sum()

    def test(self, inputtest, outputtest, f, **kwargs):
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


def keepsize(n, **kwargs):
    '''return the sample size for a data set n.'''
    return int(n - round((1 - 1. / n)**n * n))


def split_train_test(a, keep=None, **kwargs):
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


def toclass(a, classier=None, **kwargs):
    '''convert a matrix to a matrix split every class.'''
    if classier != None:
        setitem = sorted(list(set([i.item() for i in a])))
    else:
        setitem = classier
    result = np.matrix(np.zeros((a.shape[0], len(setitem))))
    for i in range(len(setitem)):
        result[:, i] = (a == setitem[i])
    return result
