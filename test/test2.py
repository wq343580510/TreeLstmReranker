import numpy as np
import theano
import theano.tensor as T


x1 = T.imatrix(name='x1')  # word indices
# x2 = T.ivector(name='x2')
# embeddings = T.dmatrix(name='emb')
# emb_x1 = embeddings[x1]
# emb_x1 = emb_x1 * T.neq(x1, -1).dimshuffle(0, 'x')
s = T.neq(x1, -1).dimshuffle(0, 'x')
# fns = theano.function(inputs=[x1,embeddings], outputs=emb_x1)
fns3 = theano.function(inputs=[x1], outputs=s)
# c = T.concatenate(x1,x2)
# fns2 = theano.function(inputs=[x1,x2], outputs=[c])
s1 = [[1,3,5,-1,-1,10,21],[1,3,5,-1,-1,10,21],[1,3,5,-1,-1,10,21]]

# s2 = [1,3,5,-1,-1,10,21]
# e = np.random.rand(100,4).tolist()
#q = fns(s1,s2)
#print q
print fns3(s1)
# id_x = [1,2,3,5,4,7,8,6]
# id_x2 = [2,4,5,7,6,8,3,1]
# def get_cor_index(id_x, id_x2):
#     id2_1 = [id_x.index(x) for x in id_x2]
#     id1_2 = [id_x2.index(x) for x in id_x]
#     return id1_2, id2_1
# l1,l2 = get_cor_index(id_x,id_x2)
# print l1,l2