
import numpy as np
import theano.tensor as T
import theano

W_u = theano.shared(np.full([10,10],0.5,dtype = 'float32'))
W_i = theano.shared(np.full([10,10],0.5,dtype = 'float32'))
W_o = theano.shared(np.full([10,10],0.5,dtype = 'float32'))
W_f = theano.shared(np.full([10,10],0.5,dtype = 'float32'))
U_u = theano.shared(np.full([10,10],0.5,dtype = 'float32'))
U_i = theano.shared(np.full([10,10],0.5,dtype = 'float32'))
U_o = theano.shared(np.full([10,10],0.5,dtype = 'float32'))
U_f = theano.shared(np.full([10,10],0.5,dtype = 'float32'))
b_u = theano.shared(np.full([10],0,dtype = 'float32'))
b_i = theano.shared(np.full([10],0,dtype = 'float32'))
b_o  = theano.shared(np.full([10],0,dtype = 'float32'))
b_f = theano.shared(np.full([10],0,dtype = 'float32'))

parent_x = T.fvector('p')
child_h = T.fmatrix('p1')
child_c = T.fmatrix('p2')
child_exists = T.fvector('p3')
h_tilde = T.sum(child_h, axis = 0)

i = T.nnet.sigmoid(T.dot(W_i, parent_x) + T.dot(U_i, h_tilde) + b_i)
o = T.nnet.sigmoid(T.dot(W_o, parent_x) + T.dot(U_o, h_tilde) + b_o)
u = T.tanh(T.dot(W_u, parent_x) + T.dot(U_u, h_tilde) + b_u)
f = (T.nnet.sigmoid(
    T.dot(W_f, parent_x).dimshuffle('x', 0) +
    T.dot(child_h, U_f.T) +
    b_f.dimshuffle('x', 0)) *
     child_exists.dimshuffle(0, 'x'))

c = i * u + T.sum(f * child_c, axis = 0)
h = o * T.tanh(c)
fun = theano.function(inputs = [parent_x,child_h,child_c,child_exists],
                outputs = [i,o,u,f,c,h])
p = np.full([10],0.4,dtype = 'float32')
ch = np.zeros([10,10],dtype = 'float32')
cc = np.zeros([10,10],dtype = 'float32')
ce = np.zeros([10],dtype = 'float32')

i,o,u,f,c,h = fun(p,ch,cc,ce)

print i
print o
print u
print f
print c
print h