import os
import urllib
import numpy as np
import tensorflow as tf

import skimage.io as io
import PIL.Image

import config
from utils import logger, ensure_file, ensure_dir
from coco import CocoManager

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

def prepare_features(coco_manager, image_source_dir, target):
    image_manager = ImageManager()
    broken_images = set(['COCO_train2014_000000167126.jpg'])

    processed = 0
    storage = tf.python_io.TFRecordWriter(target)
    for img_id in coco_manager.img_ids():
        logger().info('%d items processed', processed)
        processed += 1

        img = coco_manager.load_images(img_id)[0]

        if img['file_name'] in broken_images:
            logger().warn('Image is broken, skipping')
            continue

        raw_data = io.imread(os.path.join(image_source_dir, img['file_name']))
        if raw_data.ndim == 2:
            raw_data = raw_data.reshape(*(raw_data.shape + (1,))) + np.zeros((1, 1, 3), dtype=np.int32)
            logger().warn('Found grayscale image, fixed')

        if raw_data.shape[0] < 224 or raw_data.shape[1] < 224:
            logger().warn('Image is too small, skipping')
            continue

        image_data = image_manager.feature_vector(raw_data).reshape(-1)
        text_data_batch = np.int64(coco_manager.sents(img_id))

        text_buffer = np.zeros((config.sents_per_sample, config.max_len + 1), dtype=np.int64)
        len_buffer = np.zeros(config.sents_per_sample, dtype=np.int64)

        for idx, text_data in enumerate(text_data_batch[:config.sents_per_sample, :config.max_len]):
            text_buffer[idx, 0] = config.words_count + 1
            text_buffer[idx, 1 : 1 + text_data.shape[-1]] = text_data
            len_buffer[idx] = text_data.shape[-1]

        example = tf.train.Example()
        example.features.feature['image'].float_list.value.extend([float(value) for value in image_data])
        example.features.feature['text'].int64_list.value.extend(text_buffer[:, :-1].reshape(-1))
        example.features.feature['result'].int64_list.value.extend(text_buffer[:, 1:].reshape(-1))
        example.features.feature['len'].int64_list.value.extend(len_buffer)

        storage.write(example.SerializeToString())

if __name__ == '__main__':
    train_coco_manager = CocoManager(
        config.train_annotations_file,
        config.words_count)

    validation_coco_manager = CocoManager(
        config.validation_annotations_file,
        config.words_count,
        vocab=train_coco_manager.vocab)

    prepare_features(train_coco_manager, config.train_image_source_dir, config.train_features_file)
    prepare_features(validation_coco_manager, config.validation_image_source_dir, config.validation_features_file)