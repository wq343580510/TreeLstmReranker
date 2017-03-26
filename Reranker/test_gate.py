import dependency_model
import numpy as np
import tree_rnn
import theano.tensor as T
import theano
x = T.fvector('x')
c = T.nnet.sigmoid(x)
d = T.tanh(x)
func = theano.function(inputs=[x],outputs=[c])
func2 = theano.function(inputs=[x],outputs=[d])
print func2(np.full([10],0.15,dtype='float32'))
#print 0.0996 * 0.5249
print 'build model'
model = dependency_model.get_model(9, 3)
print 'model established'
embeddings = [[0.1]*10,[0.2]*10,[0.3]*10,[0.4]*10,
              [0.5]*10,[0.6]*10,[0.7]*10,[0.8]*10,[0.9]*10]
model.set_emb(np.array(embeddings,dtype='float32'))
w = np.full([10,10],0.5,dtype='float32')
b = np.full([10],0,dtype='float32')
model.set_parmas(w,b)
a1 = tree_rnn.Node(0,0)
a2 = tree_rnn.Node(1,1)
a3 = tree_rnn.Node(2,2)
a1.add_child(a2)
a1.add_child(a3)
a11 = tree_rnn.Node(0,0)
a22 = tree_rnn.Node(1,1)
a33 = tree_rnn.Node(2,2)
a11.add_child(a22)
a11.add_child(a33)
# a4 = tree_rnn.Node(4,4)
# a1.add_child(a4)
# a5 = tree_rnn.Node(5)
# a6 = tree_rnn.Node(6)
# a3.add_child(a5)
# a3.add_child(a6)
state1,state2 = model.train_margin2(a1,a11)
print state1
print state2

# tree_rnn.Node b1 = tree_rnn.Node(1)
# tree_rnn.Node b2 = tree_rnn.Node(1)
# tree_rnn.Node b3 = tree_rnn.Node(1)
# tree_rnn.Node b4 = tree_rnn.Node(1)
# tree_rnn.Node b5 = tree_rnn.Node(1)
# tree_rnn.Node b6 = tree_rnn.Node(1)

# a = T.fvector('a')
# i = T.sum(a)
# func  = theano.function(inputs=[a],outputs=i)
# print func(np.array([0.1,0.2,0.3,0.4],dtype='float32'))
# arr = np.array([0.1,0.2,0.3,0.4],dtype='float32')
# print np.mean(arr)

