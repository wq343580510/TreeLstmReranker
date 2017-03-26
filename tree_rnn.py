import numpy as np
import theano
from theano import tensor as T
from theano.compat.python2x import OrderedDict
theano.config.floatX = 'float32'

class TreeRNN(object):
    """Data is represented in a tree structure.

    Every leaf and internal node has a data (provided by the input)
    and a memory or hidden state.  The hidden state is computed based
    on its own data and the hidden states of its children.  The
    hidden state of leaves is given by a custom init function.

    The entire tree's embedding is represented by the final
    state computed at the root.

    """

    def __init__(self, num_emb, emb_dim, hidden_dim, output_dim,
                 degree=2, learning_rate=0.01, momentum=0.9,
                 trainable_embeddings=True,
                 labels_on_nonroot_nodes=False,
                 irregular_tree=False):
        assert emb_dim > 1 and hidden_dim > 1
        self.num_emb = num_emb
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.degree = degree
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.irregular_tree = irregular_tree

        self.params = []
        self.embeddings = theano.shared(self.init_matrix([self.num_emb, self.emb_dim]))
        self.params.append(self.embeddings)
        self.recursive_unit = self.create_recursive_unit()
        self.leaf_unit = self.create_leaf_unit()
        self.recursive_unit2 = self.create_recursive_unit2()
        self.leaf_unit2 = self.create_leaf_unit2()
        self.output_fn = self.create_output_fn()

        self.x1 = T.ivector(name='x1')  # word indices
        self.x2 = T.ivector(name='x2')  # word indices
        self.num_words1 = self.x1.shape[0]
        self.num_words2 = self.x1.shape[0]
        self.emb_x1 = self.embeddings[self.x1]
        self.emb_x1 = self.emb_x1 * T.neq(self.x1, -1).dimshuffle(0, 'x')  # zero-out non-existent embeddings
        self.emb_x2 = self.embeddings[self.x2]
        self.emb_x2 = self.emb_x2 * T.neq(self.x2, -1).dimshuffle(0, 'x')  # zero-out non-existent embeddings
        tree_1 = T.imatrix(name='tree1')  # shape [None, self.degree]
        # do not consider the unk
        tree_2 = T.imatrix(name='tree2')  # shape [None, self.degree]
        tree_3 = T.imatrix(name='tree3')  # shape [None, self.degree]
        # do not consider the unk
        tree_4 = T.imatrix(name='tree4')  # shape [None, self.degree]
        self.tree_states_1 = self.compute_tree(self.emb_x1, tree_1)
        self.tree_states_2 = self.compute_tree(self.emb_x2, tree_2)
        self.gate_states = self.compute_tree_with_gate(self.emb_x1, tree_3,self.tree_states_2)
        #theano.printing.debugprint(self.gate_states)
        self.pred_y = self.output_fn(self.gate_states[-1])
        self.gate_states_gold = self.compute_tree_with_gate(self.emb_x2, tree_4,self.tree_states_1)
        #theano.printing.debugprint(self.gate_states_gold)
        self.gold_y = self.output_fn(self.gate_states_gold[-1])
        self.loss_margin = self.loss_fn(self.gold_y, self.pred_y)
        updates_margin = self.adagrad(self.loss_margin)
        train_inputs_margin  = [self.x1, self.x2, tree_1,tree_2,tree_3,tree_4]
        self._train_margin = theano.function(train_inputs_margin,
                                      [self.loss_margin],
                                      updates=updates_margin
                                      )


    def init_matrix(self, shape):
        return np.random.normal(scale=0.1, size=shape).astype(theano.config.floatX)

    def init_vector(self, shape):
        return np.zeros(shape, dtype=theano.config.floatX)

    def create_output_fn(self):
        self.W_out = theano.shared(self.init_matrix([self.output_dim, self.hidden_dim*2]))
        self.b_out = theano.shared(self.init_vector([self.output_dim]))
        self.params.extend([self.W_out, self.b_out])

        def fn(final_state):
            return T.nnet.softmax(
                T.dot(self.W_out, final_state) + self.b_out)
        return fn


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
            return parent_h - f * compare_h

        return unit

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

    def create_recursive_unit2(self):
        self.W2_i = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.U2_i = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b2_i = theano.shared(self.init_vector([self.hidden_dim]))
        self.W2_f = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.U2_f = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b2_f = theano.shared(self.init_vector([self.hidden_dim]))
        self.W2_o = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.U2_o = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b2_o = theano.shared(self.init_vector([self.hidden_dim]))
        self.W2_u = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.U2_u = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b2_u = theano.shared(self.init_vector([self.hidden_dim]))
        self.params.extend([
            self.W2_i, self.U2_i, self.b2_i,
            self.W2_f, self.U2_f, self.b2_f,
            self.W2_o, self.U2_o, self.b2_o,
            self.W2_u, self.U2_u, self.b2_u])

        def unit(parent_x, child_h, child_c, child_exists):
            h_tilde = T.sum(child_h, axis=0)
            i = T.nnet.sigmoid(T.dot(self.W2_i, parent_x) + T.dot(self.U2_i, h_tilde) + self.b2_i)
            o = T.nnet.sigmoid(T.dot(self.W2_o, parent_x) + T.dot(self.U2_o, h_tilde) + self.b2_o)
            u = T.tanh(T.dot(self.W2_u, parent_x) + T.dot(self.U2_u, h_tilde) + self.b2_u)

            f = (T.nnet.sigmoid(
                T.dot(self.W2_f, parent_x).dimshuffle('x', 0) +
                T.dot(child_h, self.U2_f.T) +
                self.b2_f.dimshuffle('x', 0)) *
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

    def create_leaf_unit2(self):
        dummy = 0 * theano.shared(self.init_vector([self.degree, self.hidden_dim]))

        def unit(leaf_x):
            return self.recursive_unit2(
                leaf_x,
                dummy,
                dummy,
                dummy.sum(axis=1))

        return unit

    def compute_tree_with_gate(self,emb_x, tree,tree_states):
        num_nodes = tree.shape[0]  # num internal nodes
        num_leaves = self.x1.shape[0] - num_nodes

        # compute leaf hidden states
        (leaf_h, leaf_c), _ = theano.map(
            fn=self.leaf_unit2,
            sequences=[emb_x[:num_leaves]])
        init_node_h = T.concatenate([leaf_h, leaf_h], axis=0)
        init_node_c = T.concatenate([leaf_c, leaf_c], axis=0)

        # use recurrence to compute internal node hidden states
        def _recurrence(cur_emb, node_info, t, node_h, node_c, last_h,states):
            child_exists = node_info > -1
            offset = num_leaves * int(self.irregular_tree) - child_exists * t
            child_h = node_h[node_info + offset] * child_exists.dimshuffle(0, 'x')
            child_c = node_c[node_info + offset] * child_exists.dimshuffle(0, 'x')
            parent_h, parent_c = self.recursive_unit2(cur_emb, child_h, child_c, child_exists)
            parent_gate_h = parent_h
            node_h = T.concatenate([node_h,
                                    parent_h.reshape([1, self.hidden_dim])])
            node_c = T.concatenate([node_c,
                                    parent_c.reshape([1, self.hidden_dim])])
            return node_h[1:], node_c[1:], states[0]

        dummy = theano.shared(self.init_vector([self.hidden_dim]))
        (_, _, parent1_h), _ = theano.scan(
            fn=_recurrence,
            outputs_info=[init_node_h, init_node_c, dummy],
            sequences=[emb_x[num_leaves:], tree, T.arange(num_nodes)],
            non_sequences=tree_states,
            n_steps=num_nodes)

        return T.concatenate([leaf_h, parent1_h], axis=0)

    def compute_tree(self, emb_x, tree):
        num_nodes = tree.shape[0]  # num internal nodes
        num_leaves = self.x1.shape[0] - num_nodes

        # compute leaf hidden states
        (leaf_h, leaf_c), _ = theano.map(
            fn=self.leaf_unit,
            sequences=[emb_x[:num_leaves]])
        init_node_h = T.concatenate([leaf_h, leaf_h], axis=0)
        init_node_c = T.concatenate([leaf_c, leaf_c], axis=0)

        # use recurrence to compute internal node hidden states
        def _recurrence(cur_emb, node_info, t, node_h, node_c, last_h):
            child_exists = node_info > -1
            offset = num_leaves * int(self.irregular_tree) - child_exists * t
            child_h = node_h[node_info + offset] * child_exists.dimshuffle(0, 'x')
            child_c = node_c[node_info + offset] * child_exists.dimshuffle(0, 'x')
            parent_h, parent_c = self.recursive_unit(cur_emb, child_h, child_c, child_exists)
            node_h = T.concatenate([node_h,
                                    parent_h.reshape([1, self.hidden_dim])])
            node_c = T.concatenate([node_c,
                                    parent_c.reshape([1, self.hidden_dim])])
            return node_h[1:], node_c[1:], parent_h

        dummy = theano.shared(self.init_vector([self.hidden_dim]))
        (_, _, parent_h), _ = theano.scan(
            fn=_recurrence,
            outputs_info=[init_node_h, init_node_c, dummy],
            sequences=[emb_x[num_leaves:], tree, T.arange(num_nodes)],
            n_steps=num_nodes)

        return T.concatenate([leaf_h, parent_h], axis=0)

    def loss_fn(self, y, pred_y):
        return T.sum(T.sqr(y - pred_y))

    def adagrad(self, loss, epsilon=1e-6):
        grads = T.grad(loss, self.params)
        updates = OrderedDict()

        for param, grad in zip(self.params, grads):
            value = param.get_value(borrow=True)
            accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable)
            accu_new = accu + grad ** 2
            updates[accu] = accu_new
            updates[param] = param - (self.learning_rate * grad /
                                      T.sqrt(accu_new + epsilon))

        return updates