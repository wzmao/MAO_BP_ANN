import os

from mao_bp_ann import *

from matplotlib import *
from pylab import *


if not os.path.exists("test3-figure"):
	os.mkdir("test3-figure")
# Read Data
f = open('data3', 'r')
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
count=0
temp=(web.trans[0][:,0]-web.trans[0][:,1]).T
plot(list(array(train[:,0][train[:,2]==1]))[0],list(array(train[:,1][train[:,2]==1]))[0],'o',c='red')
plot(list(array(train[:,0][train[:,2]==2]))[0],list(array(train[:,1][train[:,2]==2]))[0],'o',c='blue')
plot([0,5],[-temp[0,2]/temp[0,1],5*(-temp[0,0]/temp[0,1])-temp[0,2]/temp[0,1]],c='green')
xlim(0,5)
ylim(0,5)
savefig("test3-figure/"+str(count),dpi=100)
clf()
# images=[Image.open('test3-figure/'+str(count)+".png") ]
for i in range(100):
	web.simulate(10, step=0.1)
	count+=10
	temp=(web.trans[0][:,0]-web.trans[0][:,1]).T
	plot(list(array(train[:,0][train[:,2]==1]))[0],list(array(train[:,1][train[:,2]==1]))[0],'o',c='red')
	plot(list(array(train[:,0][train[:,2]==2]))[0],list(array(train[:,1][train[:,2]==2]))[0],'o',c='blue')
	plot([0,5],[-temp[0,2]/temp[0,1],5*(-temp[0,0]/temp[0,1])-temp[0,2]/temp[0,1]],c='green')
	xlim(0,5)
	ylim(0,5)
	savefig("test3-figure/"+str(count),dpi=100)
	clf()
	# images+=[Image.open('test3-figure/'+str(count)+".png")]

# Test data


def collect(x):
    return list(x.flat).index(x.max())

result = web.test(
    test[:, :-1], toclass(test[:, -1], classier=[1, 2]), f=collect)
print sum(result) * 1. / len(result), len(result)
print web.trans
# writeGif("test3.gif", images,)
