import os
import zipfile
import numpy as np
import tensorflow as tf

import PIL.Image

import config
from utils import logger, ensure_file, ensure_dir

class ImageManager(object):
    def __init__(self):
        models_dir = config.base_path
        net_file = os.path.join(models_dir, 'tensorflow_inception_graph.pb')
        synset_file = os.path.join(models_dir, 'imagenet_comp_graph_label_strings.txt')

        ensure_dir(models_dir)
        if not (os.path.isfile(net_file) and os.path.isfile(synset_file)):
            archive_path = os.path.join(models_dir, 'inception5h.zip')
            ensure_file(archive_path, 'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip')
            z = zipfile.ZipFile(archive_path)
            z.extractall(models_dir)

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

    def extract_features(self, raw_data):
        if type(raw_data) == list:
            data = [self.prepare_data(image) for image in raw_data]
            data = np.stack(data, axis=0)
        else:
            data = self.prepare_data(raw_data)
            data = data.reshape((1,) + data.shape)
            
        features = self.session.run('import/avgpool0/reshape:0', feed_dict={'import/input:0': data})

        return features