#! /usr/bin/env python3
from keras import backend as K
import masknet

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

if __name__ == "__main__":
    K.set_learning_phase(0)

    model = masknet.create_model()
    model.summary()
    model.load_weights("weights.hdf5")

    output_name = model.output.name.split(':')[0]

    sess = K.get_session()

    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), [output_name])
    graph_io.write_graph(constant_graph, ".", "fast_mask.pb", as_text=False)
