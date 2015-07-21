__author__ = 'johannesjurgovsky'

import theano.tensor as T
import lasagne

l_in = lasagne.layers.InputLayer((100,50))
l_hidden = lasagne.layers.DenseLayer(l_in, num_units=200, name="hidden_layer")
l_out = lasagne.layers.DenseLayer(l_hidden, num_units=10, nonlinearity=T.nnet.softmax)