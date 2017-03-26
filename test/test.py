import numpy as np
import theano
import theano.tensor as T
W_out = np.ones((1,6), dtype=theano.config.floatX)
b_out = np.ones((1), dtype=theano.config.floatX)


def compute_one_edge(one_child, tree_states, parent):
    score = T.switch(T.lt(-1, one_child),
                     T.dot(W_out, T.concatenate([tree_states[one_child], parent])) + b_out,
                     T.zeros(1))
    return score


def compute_one_tree(one_tree, tree_states):
    children = one_tree[:-1]
    parent = tree_states[one_tree[-1]]
    result, _ = theano.scan(
        fn=compute_one_edge,
        outputs_info=None,
        sequences=[children],
        non_sequences=[tree_states, parent],
    )
    return T.sum(result)


tree_states = T.dmatrix('treestate')
tree = T.imatrix('tree')
scores, _ = theano.scan(
        fn=compute_one_tree,
        outputs_info=None,
        sequences=[tree],
        non_sequences=[tree_states],
)

fns = theano.function(inputs=[tree_states,tree], outputs=T.sum(scores))

# tree = [[0,1,2,3,-1,-1,4],
#         [4,5,-1,-1,-1,-1,6],
#         [6,-1,-1,-1,-1,-1,7]
#         ]
tree = [[0,-1,-1,-1,-1,-1,4],[1,-1,-1,-1,-1,-1,4]]
tree_states = [[0.1,0.1,0.1],[0.2,0.2,0.2],
        [0.3,0.3,0.3],[0.4,0.4,0.4],
        [0.5,0.5,0.5],[0.6,0.6,0.6],[0.7,0.7,0.7],[0.8,0.8,0.8]]
score = fns(tree_states,tree)
print score