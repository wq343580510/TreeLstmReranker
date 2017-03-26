import theano.tensor as T
import theano
from theano.compat.python2x import OrderedDict
import numpy as np
loss = T.scalar('x')
params = theano.shared(np.zeros(2, dtype=theano.config.floatX))
updates = OrderedDict()
updates[params] = T.switch(T.lt(loss, 0), [0.111,0.111],
                           [0.222, 0.222])
a = T.switch(T.lt(loss, 0),-loss,loss)
f = theano.function(inputs=[loss],outputs=[a],updates=updates)
print f(0.2)
print f(-0.2)
print ""