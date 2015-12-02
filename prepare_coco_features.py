import os
import numpy as np
import tensorflow as tf

import skimage.io as io

import config
from image import ImageManager
from coco import CocoManager
from utils import logger

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