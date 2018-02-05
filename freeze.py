#! /usr/bin/env python3
import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.models import Sequential,Model
import masknet, os

if __name__ == "__main__":
    sess = tf.Session()
    # tell Keras to use the session
    K.set_session(sess)

    # From this document: https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html

    # let's convert the model for inference
    K.set_learning_phase(0)  # all new operations will be in test mode from now on
    # serialize the model and get its weights, for quick re-building
    model = masknet.create_model()
    model.summary()
    model.load_weights("weights.hdf5")

    print("Input name:")
    print(model.input)
    print("Output name:")
    print(model.output)
    output_name=model.output.name.split(':')[0]
    print(output_name)

    #  not sure what this is for
    export_version = 1 # version number (integer)

    export_path = "./yolo_masknet"

    graph_file=export_path+"_graph.pb"
    ckpt_file=export_path+".ckpt"
    # create a saver
    saver = tf.train.Saver(sharded=True)
    tf.train.write_graph(sess.graph_def, '', graph_file)
    save_path = saver.save(sess, ckpt_file)
    freeze_graph_binary = "python3 /usr/local/lib/python3.5/dist-packages/tensorflow/python/tools/freeze_graph.py"
    command = freeze_graph_binary +" --input_graph=./"+graph_file+" --input_checkpoint=./"+ckpt_file+" --output_node_names="+output_name+" --output_graph=./"+export_path+".pb"
    print(command)
    os.system(command)
