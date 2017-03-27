import numpy as np
import theano
import theano.tensor as T
def init_matrix(shape):
    return np.random.normal(scale=0.1, size=shape).astype(theano.config.floatX)

scoreVector = theano.shared(init_matrix([10, 10, 3]))
compVector = theano.shared(init_matrix([10, 10, 3, 3]))
child = T.ivector(name='child')
child_exists = child>-1
tags = T.ivector(name='tag')
child_tags = tags[child]
compMatrix = compVector[tags[2],child,:,:]
scoreVec = scoreVector[tags[2],child,:]*child_exists.dimshuffle(0, 'x')

# child_tags = tags[child]
# child2 = T.ivector(name='tag2')
# fun = theano.function(inputs=[tags,child],outputs=[T.concatenate([tags,child],axis=0)])
fun = theano.function(inputs=[tags,child],outputs=[compMatrix,scoreVec])
c1 = np.array([4,7,3,4,3,6,12,43,51,12,35,76,75])
c2 = np.array([1,2,3,-1,-1,-1,7,8])
# c3 = np.array([4,7,3,4,3,6,12,])
# c4 = np.array([1,2,3,-1,-1,-1,7])
print fun(c1,c2)
print np.float(0.3)