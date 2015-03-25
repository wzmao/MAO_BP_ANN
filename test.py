import numpy
from mao_bp_ann import *

print 1
temp=neu_net(webshape=[1,2,1])
del(temp)

print 2
temp=neu_net(webshape=numpy.array([1,2,1]))
del(temp)

print 3
temp=neu_net()
temp.set_webshape([])
temp.set_webshape(numpy.array([]))
del(temp)

# temp=new_net(label=())