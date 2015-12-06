import argparse
import numpy as np
import tensorflow as tf
import PIL.Image

from tensortalk import config
from tensortalk.image import ImageManager
from tensortalk.coco import CocoManager
from tensortalk.network import CaptionNetwork, UserInputPipeline
from tensortalk.sampler import BeamSearchSampler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images', metavar='image', type=str, nargs='+',
        help='Images for captioning')
    parser.add_argument('--model', dest='model_file', required=True,
        help='Path to serialized model')
    args = parser.parse_args()

    image_manager = ImageManager()
    coco_manager = CocoManager(config.train_annotations_file, config.words_count)
    weights_file = args.model_file

    session = tf.Session()
    input_pipeline = UserInputPipeline()

    model = CaptionNetwork(session, input_pipeline)
    model.load(weights_file)

    sampler = BeamSearchSampler(beam_size=5)

    for img_name in args.images:
        img = np.float32(PIL.Image.open(img_name))
        img_features = image_manager.extract_features(img)

        sequences = sampler.sample(model, img_features, size=15)
        print img_name
        for sequence in sequences[-3:]:
            words = [coco_manager.vocab.get_word(word_idx - 1, limit=config.output_words_count - 1) for word_idx in sequence]
            print ' '.join(words)