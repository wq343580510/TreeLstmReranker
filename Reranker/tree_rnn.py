import numpy as np
import theano
from theano import tensor as T,printing
from theano.compat.python2x import OrderedDict
theano.config.floatX = 'float32'
SEED = 12389
L2_RATIO = 0.0001
class Node(object):
    def __init__(self, val=None,id_s = 0,tag = 0):
        self.children = []
        self.val = val
        self.idx = None
        self.height = 1
        self.size = 1
        self.tag_idx = tag
        self.num_leaves = 1
        self.id_s = id_s
        self.parent = None
        self.label = None

    def _update(self):
        self.height = 1 + max([child.height for child in self.children if child] or [0])
        self.size = 1 + sum(child.size for child in self.children if child)
        self.num_leaves = (all(child is None for child in self.children) +
                           sum(child.num_leaves for child in self.children if child))
        if self.parent is not None:
            self.parent._update()

    def add_child(self, child):
        self.children.append(child)
        child.parent = self
        self._update()

    def add_children(self, other_children):
        self.children.extend(other_children)
        for child in other_children:
            child.parent = self
        self._update()


class BinaryNode(Node):
    def __init__(self, val=None):
        super(BinaryNode, self).__init__(val=val)

    def add_left(self, node):
        if not self.children:
            self.children = [None, None]
        self.children[0] = node
        node.parent = self
        self._update()

    def add_right(self, node):
        if not self.children:
            self.children = [None, None]
        self.children[1] = node
        node.parent = self
        self._update()

    def get_left(self):
        if not self.children:
            return None
        return self.children[0]

    def get_right(self):
        if not self.children:
            return None
        return self.children[1]


def gen_nn_inputs(root_node, max_degree=None, only_leaves_have_vals=False,
                  with_labels=False):
    """Given a root node, returns the appropriate inputs to NN.

    The NN takes in
        x: the values at the leaves (e.g. word indices)
        tree: a (n x degree) matrix that provides the computation order.
            Namely, a row tree[i] = [a, b, c] in tree signifies that a
            and b are children of c, and that the computation
            f(a, b) -> c should happen on step i.

    """
    _clear_indices(root_node)
    x, leaf_labels,id_x,tags = _get_leaf_vals(root_node)
    tree, internal_x, internal_labels ,id_internal_x,tags_internal = \
        _get_tree_traversal(root_node, len(x), max_degree)
    assert all(v is not None for v in x)
    if not only_leaves_have_vals:
        assert all(v is not None for v in internal_x)
        x.extend(internal_x)
        id_x.extend(id_internal_x)
        tags.extend(tags_internal)
    if max_degree is not None:
        assert all(len(t) == max_degree + 1 for t in tree)
    if with_labels:
        labels = leaf_labels + internal_labels
        labels_exist = [l is not None for l in labels]
        labels = [l or 0 for l in labels]
        return (np.array(x, dtype='int32'),
                np.array(tree, dtype='int32'),
                np.array(labels, dtype=theano.config.floatX),
                np.array(labels_exist, dtype=theano.config.floatX))
    return (np.array(x, dtype='int32'),
            np.array(tree, dtype='int32'),
            id_x,
            np.array(tags, dtype='int32')
            )


def _clear_indices(root_node):
    root_node.idx = None
    [_clear_indices(child) for child in root_node.children if child]


def _get_leaf_vals(root_node):
    """Get leaf values in deep-to-shallow, left-to-right order."""
    all_leaves = []
    layer = [root_node]
    while layer:
        next_layer = []
        for node in layer:
            if all(child is None for child in node.children):
                all_leaves.append(node)
            else:
                next_layer.extend([child for child in node.children[::-1] if child])
        layer = next_layer

    vals = []
    labels = []
    id_x = []
    tags = []
    for idx, leaf in enumerate(reversed(all_leaves)):
        leaf.idx = idx
        vals.append(leaf.val)
        id_x.append(leaf.id_s)
        labels.append(leaf.label)
        tags.append(leaf.tag_idx)
    return vals, labels,id_x,tags


def _get_tree_traversal(root_node, start_idx=0, max_degree=None):
    """Get computation order of leaves -> root."""
    if not root_node.children:
        return [], [], []
    layers = []
    layer = [root_node]
    while layer:
        layers.append(layer[:])
        next_layer = []
        [next_layer.extend([child for child in node.children if child])
         for node in layer]
        layer = next_layer

    tree = []
    internal_vals = []
    id_internal_x = []
    tag_internal_x = []
    labels = []
    idx = start_idx
    for layer in reversed(layers):
        for node in layer:
            if node.idx is not None:
                # must be leaf
                assert all(child is None for child in node.children)
                continue

            child_idxs = [(child.idx if child else -1)
                          for child in node.children]
            if max_degree is not None:
                child_idxs.extend([-1] * (max_degree - len(child_idxs)))
            assert not any(idx is None for idx in child_idxs)

            node.idx = idx
            tree.append(child_idxs + [node.idx])
            internal_vals.append(node.val if node.val is not None else -1)
            id_internal_x.append(node.id_s if node.val is not None else -1)
            tag_internal_x.append(node.tag_idx)
            labels.append(node.label)
            idx += 1

    return tree, internal_vals, labels,id_internal_x,tag_internal_x


class TreeRNN(object):
    """Data is represented in a tree structure.

    Every leaf and internal node has a data (provided by the input)
    and a memory or hidden state.  The hidden state is computed based
    on its own data and the hidden states of its children.  The
    hidden state of leaves is given by a custom init function.

    The entire tree's embedding is represented by the final
    state computed at the root.

    """
    def __init__(self, num_emb,tag_size, emb_dim, hidden_dim, output_dim,
                 degree=2, learning_rate=0.01, momentum=0.9,
                 trainable_embeddings=True,
                 labels_on_nonroot_nodes=False,
                 irregular_tree=False,pairwise = True):
        assert emb_dim > 1 and hidden_dim > 1
        self.num_emb = num_emb
        self.tag_size = tag_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.degree = degree
        self.learning_rate = learning_rate
        self.L2_ratio = L2_RATIO
        self.Pairwise = pairwise
        self.params = []
        np.random.seed(SEED)
        #self.embeddings = theano.shared(self.init_matrix([self.num_emb, self.emb_dim]))
        self.embeddings = theano.shared(self.init_matrix([self.num_emb, self.emb_dim]))
        self.params.append(self.embeddings)
        self.recursive_unit = self.create_recursive_unit()
        self.leaf_unit = self.create_leaf_unit()
        #self.output_fn = self.create_output_fn()
        self.score_fn = self.create_score_fn()
        self.x1 = T.ivector(name='x1')  # word indices
        self.x2 = T.ivector(name='x2')  # word indices
        self.tag_1 = T.ivector(name='tag1')  # word indices
        self.tag_2 = T.ivector(name='tag2')  # word indices
        self.x1_2 = T.ivector(name='x1_2')  # word indices
        self.x2_1 = T.ivector(name='x2_1')  # word indices
        self.num_words = self.x1.shape[0]
        self.emb_x1 = self.embeddings[self.x1]
        self.emb_x1 = self.emb_x1 * T.neq(self.x1, -1).dimshuffle(0, 'x')  # zero-out non-existent embeddings
        self.emb_x2 = self.embeddings[self.x2]
        self.emb_x2 = self.emb_x2 * T.neq(self.x2, -1).dimshuffle(0, 'x')  # zero-out non-existent embeddings
        self.tree_1 = T.imatrix(name='tree1')  # shape [None, self.degree]
        self.tree_2 = T.imatrix(name='tree2')  # shape [None, self.degree]
        self.tree_3 = T.imatrix(name='tree3')  # shape [None, self.degree]
        self.tree_4 = T.imatrix(name='tree4')  # shape [None, self.degree]
        self.tree_states_1,self.score1 = self.compute_tree(self.emb_x1, self.tree_1[:, :-1])
        self.tree_states_2,self.score2 = self.compute_tree(self.emb_x2, self.tree_2[:, :-1])
        #self._compute_emb = theano.function([self.x1,self.tree_1],self.tree_states_1)
        if self.Pairwise:
            self.forget_unit = self.create_forget_gate_fun()
            self._train_pairwise,self._predict_pair = self.create_pairwise_rank()
        else:
            self._predict,self._train_pointwise = self.create_pointwise_rank()

    def create_pairwise_rank(self):
        gate_states_better = self.compute_tree_with_gate(self.x1_2, self.emb_x1, self.tree_3[:, :-1],
                                                         self.tree_states_2,self.tree_c_2)
        gate_states_worse = self.compute_tree_with_gate(self.x2_1,self.emb_x2, self.tree_4[:, :-1],
                                                        self.tree_states_1,self.tree_c_2)

        # better_y = self.output_fn(gate_states_better[-1])
        # worse_y = self.output_fn(gate_states_worse[-1])
        leaveNum1 = self.x1.shape[0] - self.tree_1.shape[0]
        leaveNum2 = self.x2.shape[0] - self.tree_2.shape[0]
        # better_y = self.score_fn(gate_states_better[leaveNum1:],self.tag_1[leaveNum1:])
        # worse_y = self.score_fn(gate_states_worse[leaveNum2:],self.tag_2[leaveNum2:])
        better_y = self.score_fn(gate_states_better,self.tree_3,self.tag_1)
        worse_y = self.score_fn(gate_states_worse,self.tree_4,self.tag_2)
        pred = self.loss_fn(worse_y,better_y)
        loss = self.loss_fn_regular(worse_y,better_y)
        train_inputs_margin = [self.x1, self.x2, self.tree_1, self.tree_2,
                               self.tree_3,
                               self.tree_4,
                               self.x1_2, self.x2_1,self.tag_1,self.tag_2]
        predict = theano.function(train_inputs_margin, [pred])
        train_margin = theano.function(train_inputs_margin,[loss],updates=self.adagrad_pair(loss))
        return train_margin,predict

    def create_pointwise_rank(self):
        train_inputs = [self.x1, self.x2, self.tree_1, self.tree_2]
        # leaveNum1 = self.x1.shape[0] - self.tree_1.shape[0]
        # leaveNum2 = self.x2.shape[0] - self.tree_2.shape[0]
        # tree_y1 = self.score_fn(self.tree_states_1[leaveNum1:],self.x1[leaveNum1:])
        # tree_y2 = self.score_fn(self.tree_states_2[leaveNum2:],self.x2[leaveNum1:])
        # tree_y1 = self.score_fn(self.tree_states_1,self.tree_1,self.tag_1)
        # tree_y2 = self.score_fn(self.tree_states_2,self.tree_2,self.tag_2)
        loss = self.loss_fn_regular(self.score1,self.score2)
        predict = theano.function([self.x1, self.tree_1],self.score1)
        train_func = theano.function(train_inputs,loss,updates=self.adagrad(loss))
        return predict,train_func

    def predict(self, root_node):
        x, tree ,_,tags = gen_nn_inputs(root_node, max_degree=self.degree, with_labels= False)
        # x list the val of leaves and internal nodes
        # self._check_input(x, tree)
        return self._predict(x, tree)

    def train_pointwise(self, root1, root2):
        x, tree, _,tag1 = gen_nn_inputs(root1, max_degree=self.degree, only_leaves_have_vals=False)
        x_2, tree_2, _,tag2 = gen_nn_inputs(root2, max_degree=self.degree, only_leaves_have_vals=False)
        #self._check_input(x, tree)
        #self._check_input(x_2, tree_2)
        loss = self._train_pointwise(x, x_2, tree, tree_2)
        return loss

    def _check_input(self, x, tree):
        assert np.array_equal(tree[:, -1], np.arange(len(x) - len(tree), len(x)))

    def train_pairwise(self,better_root,worse_root,pred = False):
        x, tree,id_x,tags1 = gen_nn_inputs(better_root, max_degree=self.degree, only_leaves_have_vals=False)
        x_2, tree_2,id_x2,tags2 = gen_nn_inputs(worse_root, max_degree=self.degree, only_leaves_have_vals=False)
        id1_2 = np.array([id_x2.index(c) for c in id_x],dtype='int32')
        id2_1 = np.array([id_x.index(c) for c in id_x2],dtype='int32')
        self._check_input(x, tree)
        self._check_input(x_2, tree_2)
        if pred:
            loss = self._predict_pair(x, x_2, tree[:, :-1], tree_2[:, :-1],
                                        tree, tree_2, id1_2,id2_1,tags1,tags2)
        else:
            loss = self._train_pairwise(x, x_2, tree[:, :-1], tree_2[:, :-1],
                                        tree, tree_2, id1_2,id2_1,tags1,tags2)
        return loss

    def init_matrix(self, shape):
        #return np.random.normal(scale=0.1, size=shape).astype(theano.config.floatX)
        return np.random.normal(scale=0.1, size=shape).astype(theano.config.floatX)

    def init_vector(self, shape):
        #return np.ones(shape, dtype=theano.config.floatX)
        return np.zeros(shape, dtype=theano.config.floatX)

    def init_emb(self, shape):
        #return np.random.normal(scale=1, size=shape).astype(theano.config.floatX)
        return np.random.normal(scale=0.1, size=shape).astype(theano.config.floatX)

    def create_score_fn(self):
        self.scoreVector = theano.shared(self.init_matrix([self.hidden_dim]))
        self.params.append(self.scoreVector)
        self.compVector = theano.shared(self.init_matrix([self.hidden_dim+self.emb_dim,self.hidden_dim]))
        self.params.append(self.compVector)
        def compute_one_edge(one_child, tree_states, parent, tags):
            parent_tag = tags[parent]
            child_tag = tags[one_child]
            score = T.switch(T.lt(-1, one_child),
                             T.dot(self.scoreVector[parent_tag][child_tag],
                                   T.concatenate([tree_states[parent], tree_states[one_child]])),
                             T.zeros(1))
            return score

        def compute_one_tree(one_tree, tree_states, tags):
            children = one_tree[0:-1]
            parent_id = one_tree[-1]
            result, _ = theano.scan(
                fn=compute_one_edge,
                outputs_info=None,
                sequences=[children],
                non_sequences=[tree_states, parent_id, tags],
            )
            return T.sum(result)

        def fn(tree_states, tree, tags):
            scores, _ = theano.scan(
                fn=compute_one_tree,
                outputs_info=None,
                sequences=[tree],
                non_sequences=[tree_states, tags],
            )
            return T.sum(scores)

        return fn

    def create_score_fn2(self):
        self.scoreVector = theano.shared(self.init_matrix([self.tag_size,self.tag_size, self.hidden_dim]))
        self.compVector = theano.shared(self.init_matrix([self.tag_size, self.tag_size, self.hidden_dim+self.emb_dim,self.hidden_dim]))
        self.params.append(self.scoreVector)
        def score(state,index,vector):
            return T.dot(vector[index],state)

        def fun(states,index):
            scores,_ = theano.scan(fn = score,sequences=[states,index],non_sequences=[self.scoreVector])
            return T.sum(scores)
        return fun

    def create_output_fn(self):
        self.W_out = theano.shared(self.init_matrix([self.output_dim, self.hidden_dim]))
        self.b_out = theano.shared(self.init_vector([self.output_dim]))
        self.params.extend([self.W_out, self.b_out])

        def fn(state):
            #T.nnet.softmax
            return T.dot(self.W_out, state) + self.b_out
        return fn


    def dropout(x, level):
        if level < 0. or level >= 1:
            raise Exception('Dropout level must be in interval [0, 1[.')
        retain_prob = 1. - level
        sample = np.random.binomial(n=1, p=retain_prob, size=x.shape)
        x *= sample
        x /= retain_prob
        return x

    def create_recursive_unit(self):
        self.W_i = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.U_i = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_i = theano.shared(self.init_vector([self.hidden_dim]))
        self.W_f = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.U_f = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_f = theano.shared(self.init_vector([self.hidden_dim]))
        self.W_o = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.U_o = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_o = theano.shared(self.init_vector([self.hidden_dim]))
        self.W_u = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.U_u = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_u = theano.shared(self.init_vector([self.hidden_dim]))
        self.params.extend([
            self.W_i, self.U_i, self.b_i,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
            self.W_u, self.U_u, self.b_u])

        def unit(parent_x, child_h, child_c, child_exists):
            h_tilde = T.sum(child_h, axis=0)
            i = T.nnet.sigmoid(T.dot(self.W_i, parent_x) + T.dot(self.U_i, h_tilde) + self.b_i)
            o = T.nnet.sigmoid(T.dot(self.W_o, parent_x) + T.dot(self.U_o, h_tilde) + self.b_o)
            u = T.tanh(T.dot(self.W_u, parent_x) + T.dot(self.U_u, h_tilde) + self.b_u)
            f = (T.nnet.sigmoid(
                T.dot(self.W_f, parent_x).dimshuffle('x', 0) +
                T.dot(child_h, self.U_f.T) +
                self.b_f.dimshuffle('x', 0)) *
                 child_exists.dimshuffle(0, 'x'))

            c = i * u + T.sum(f * child_c, axis=0)
            h = o * T.tanh(c)
            return h, c

        return unit

    def create_leaf_unit(self):
        dummy = 0 * theano.shared(self.init_vector([self.degree, self.hidden_dim]))
        def unit(leaf_x):
            return self.recursive_unit(
                leaf_x,
                dummy,
                dummy,
                dummy.sum(axis=1))

        return unit

    def create_forget_gate_fun(self):
        self.W_gate = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.U_gate = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_gate = theano.shared(self.init_vector([self.hidden_dim]))
        self.params.extend([
            self.W_gate, self.U_gate, self.b_gate,
        ])

        def unit(parent_h, compare_h):
            f = T.nnet.sigmoid(
                T.dot(self.W_gate, parent_h) +
                T.dot(self.U_gate, compare_h) +
                self.b_gate)
            return (1-f)*parent_h + f * compare_h

        return unit

    def compute_tree_with_gate(self, x, emb_x, tree, tree_states,tree_cs):
        num_nodes = tree.shape[0]  # num internal nodes
        num_leaves = self.num_words - num_nodes

        # compute leaf hidden states
        (leaf_h, leaf_c), _ = theano.map(
            fn=self.leaf_unit,
            sequences=[emb_x[:num_leaves]])
        # leaf_h, _ = theano.map(
        #     fn=self.forget_unit,
        #     sequences=[leaf_h1[:num_leaves], tree_states[:num_leaves]])

        init_node_h = T.concatenate([leaf_h, leaf_h], axis=0)
        init_node_c = T.concatenate([leaf_c, leaf_c], axis=0)

        # use recurrence to compute internal node hidden states
        def _recurrence(cur_emb, node_info, t, x_index, node_h, node_c, last_h, tree_states,tree_cs):
            child_exists = node_info > -1
            offset = num_leaves - child_exists * t
            child_h = node_h[node_info + offset] * child_exists.dimshuffle(0, 'x')
            child_c = node_c[node_info + offset] * child_exists.dimshuffle(0, 'x')
            parent_h, parent_c = self.recursive_unit(cur_emb, child_h, child_c, child_exists)
            parent_gate_h = self.forget_unit(parent_h, tree_states[x_index])
            parent_gate_c = self.forget_unit(parent_h, tree_cs[x_index])
            node_h = T.concatenate([node_h,
                                    parent_gate_h.reshape([1, self.hidden_dim])])
            node_c = T.concatenate([node_c,
                                    parent_gate_c.reshape([1, self.hidden_dim])])
            return node_h[1:], node_c[1:], parent_gate_h

        dummy = theano.shared(self.init_vector([self.hidden_dim]))
        (_, _, parent_h), _ = theano.scan(
            fn=_recurrence,
            outputs_info=[init_node_h, init_node_c, dummy],
            sequences=[emb_x[num_leaves:], tree, T.arange(num_nodes), x[num_leaves:]],
            non_sequences=[tree_states,tree_cs],
            n_steps=num_nodes)

        return T.concatenate([leaf_h, parent_h], axis=0)

    def compute_tree(self, emb_x, tree):
        num_nodes = tree.shape[0]  # num internal nodes
        num_leaves = self.num_words - num_nodes

        # compute leaf hidden states
        (leaf_h, leaf_c), _ = theano.map(
            fn=self.leaf_unit,
            sequences=[emb_x[:num_leaves]])
        init_node_h = T.concatenate([leaf_h, leaf_h], axis=0)
        init_node_c = T.concatenate([leaf_c, leaf_c], axis=0)
        # use recurrence to compute internal node hidden states
        def _recurrence(cur_emb, node_info, t, node_h, node_c, last_h,last_c,inin_score):
            child_exists = node_info > -1
            offset = num_leaves - child_exists * t
            child_h = node_h[node_info + offset] * child_exists.dimshuffle(0, 'x')
            scores, _ = theano.map(
                _score_edge,
                sequences=[child_h],
                non_sequences=[cur_emb,self.compVector,self.scoreVector]
            )
            child_c = node_c[node_info + offset] * child_exists.dimshuffle(0, 'x')
            parent_h, parent_c = self.recursive_unit(cur_emb, child_h, child_c, child_exists)
            node_h = T.concatenate([node_h,
                                    parent_h.reshape([1, self.hidden_dim])])
            node_c = T.concatenate([node_c,
                                    parent_c.reshape([1, self.hidden_dim])])
            return node_h[1:], node_c[1:], parent_h,parent_c,T.sum(scores)

        def _score_edge(child_h,cur_emb,compMatrix,scoreVec):
            z = T.dot(T.concatenate([child_h,cur_emb],axis=0),compMatrix)
            return T.dot(scoreVec,z)
        dummy = theano.shared(self.init_vector([self.hidden_dim]))
        (_, _, parent_h,parent_c,scores), _ = theano.scan(
            fn=_recurrence,
            outputs_info=[init_node_h, init_node_c, dummy,dummy,np.float(0.0)],
            sequences=[emb_x[num_leaves:], tree, T.arange(num_nodes)],
            n_steps=num_nodes)

        return T.concatenate([leaf_h, parent_h], axis=0),T.sum(scores)


    def loss_fn_regular(self, y1, y2):
        loss = T.sum(y1-y2)
        L2 = T.sum(self.W_i ** 2)+T.sum(self.W_o ** 2)+\
             T.sum(self.W_u ** 2)+T.sum(self.W_f ** 2)+\
             T.sum(self.U_i ** 2)+T.sum(self.U_o ** 2)+\
             T.sum(self.U_u ** 2)+T.sum(self.U_f ** 2)+\
             T.sum(self.scoreVector**2)+T.sum(self.compVector**2)
        if self.Pairwise:
            L2 += T.sum(self.W_gate ** 2)
            L2 += T.sum(self.U_gate ** 2)
        #T.sum(self.W_out ** 2)
        return loss + L2 * self.L2_ratio

    def loss_fn(self, y1, y2):
        return T.sum(y1-y2)

    def adagrad(self, loss, epsilon=1e-6):
        grads = T.grad(loss, self.params)
        updates = OrderedDict()

        for param, grad in zip(self.params, grads):
            value = param.get_value(borrow=True)
            accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable)
            accu_new =  accu + grad ** 2
            updates[accu] = accu_new
            updates[param] = param - (self.learning_rate * grad /
                                      T.sqrt(accu_new + epsilon))
        return updates


    def adagrad_pair(self, loss, epsilon=1e-6):
        grads = T.grad(loss, self.params)
        updates = OrderedDict()

        for param, grad in zip(self.params, grads):
            value = param.get_value(borrow=True)
            accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable)
            accu_new = T.switch(T.gt(loss,0),accu + grad ** 2,accu)
            updates[accu] = accu_new
            # updates[param] = param - (self.learning_rate * grad /
            #                           T.sqrt(accu_new + epsilon))
            updates[param] = T.switch(T.gt(loss,0),
                                      param - (self.learning_rate * grad /
                                               T.sqrt(accu_new + epsilon)),
                                      param
                                      )
        return updates