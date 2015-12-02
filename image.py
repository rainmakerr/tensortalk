import os
import numpy as np
import tensorflow as tf

import PIL.Image

import config
from utils import logger, ensure_file, ensure_dir

class ImageManager(object):
    def __init__(self):
        models_dir = os.path.expanduser(config.base_path)
        net_file = os.path.join(models_dir, 'googlenet.pb')
        synset_file = os.path.join(models_dir, 'synset.txt')

        ensure_dir(models_dir)
        ensure_file(net_file,
            'https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/android/assets/tensorflow_inception_graph.pb?raw=true')
        ensure_file(synset_file,
            'https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/android/assets/imagenet_comp_graph_label_strings.txt')

        self.synset = []
        with open(synset_file) as f:
            for line in f:
                self.synset.append(line)

        graph_def = tf.GraphDef()
        with open(net_file) as f:
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def)
        self.session = tf.Session()

    def prepare_data(self, raw_data):
        width, height = raw_data.shape[0], raw_data.shape[1]
        if width < height:
            start = height / 2 - width / 2
            end = start + width
            cropped = raw_data[:, start:end]
        else:
            start = width / 2 - height / 2
            end = start + height
            cropped = raw_data[start:end, :]

        image = PIL.Image.fromarray(np.uint8(cropped))
        image = image.resize((224, 224), PIL.Image.ANTIALIAS)
        return np.float32(image)        

    def classify(self, image):
        data = self.prepare_data(image)
        result = self.session.run('import/output2:0', feed_dict={'import/input:0': data})
        return zip(self.synset, result)

    def feature_vector(self, image):
        data = self.prepare_data(image)
        data = data.reshape((1,) + data.shape)
        features = self.session.run('import/avgpool0/reshape:0', feed_dict={'import/input:0': data})
        return features