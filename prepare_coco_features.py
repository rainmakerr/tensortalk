import os
import numpy as np
import tensorflow as tf
import PIL.Image

import config
from image import ImageManager
from coco import CocoManager
from utils import logger

IMAGE_BATCH_SIZE = 64

def prepare_features(coco_manager, image_source_dir, target):
    image_manager = ImageManager()

    processed = 0
    storage = tf.python_io.TFRecordWriter(target)

    image_batch = []
    text_batch = []
    len_batch = []

    for img_id in coco_manager.img_ids():
        logger().info('%d items processed', processed)
        processed += 1

        try:
            img = coco_manager.load_images(img_id)[0]

            raw_data = np.float32(PIL.Image.open(os.path.join(image_source_dir, img['file_name'])))
            if raw_data.ndim == 2:
                raw_data = raw_data.reshape(*(raw_data.shape + (1,))) + np.zeros((1, 1, 3), dtype=np.int32)
                logger().warn('Found grayscale image, fixed')

            text_data_batch = np.int64(coco_manager.sents(img_id))

            text_buffer = np.zeros((config.sents_per_sample, config.max_len + 1), dtype=np.int64)
            len_buffer = np.zeros(config.sents_per_sample, dtype=np.int64)

            for idx, text_data in enumerate(text_data_batch[:config.sents_per_sample, :config.max_len]):
                text_buffer[idx, 0] = config.words_count + 1
                text_buffer[idx, 1 : 1 + text_data.shape[-1]] = text_data
                len_buffer[idx] = text_data.shape[-1]

            image_batch.append(raw_data)
            text_batch.append(text_buffer)
            len_batch.append(len_buffer)

        except Exception as e:
            logger().error('Failed processing image %s: %s', img['file_name'], str(e))

        if (len(image_batch) == IMAGE_BATCH_SIZE):
            image_data = image_manager.extract_features(image_batch)

            for image, text, length in zip(image_data, text_batch, len_batch):
                example = tf.train.Example()
                example.features.feature['image'].float_list.value.extend([float(value) for value in image])
                example.features.feature['text'].int64_list.value.extend(text[:, :-1].reshape(-1))
                example.features.feature['result'].int64_list.value.extend(text[:, 1:].reshape(-1))
                example.features.feature['len'].int64_list.value.extend(length)

                storage.write(example.SerializeToString())

            image_batch = []
            text_batch = []
            len_batch = []

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