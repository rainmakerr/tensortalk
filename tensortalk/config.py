import os

from utils import ensure_dir

base_path = os.path.join(os.path.expanduser('~'), '.tensortalk')
models_path = os.path.join(base_path, 'models')
data_path = os.path.join(base_path, 'data')
coco_path = os.path.join(base_path, 'coco')
logs_path = os.path.join(base_path, 'logs')

small_config = {
    'image_features_count': 1024,

    'words_count': 10000,
    'output_words_count': 2000,
    'hidden_count': 512,
    'max_len': 30,
    'sents_per_sample': 5,
    'batch_size': 128,

    'weights_file_template': os.path.join(models_path, 'net_%d.ckpt'),

    'train_annotations_file': os.path.join(coco_path, 'captions_train2014.json'),
    'train_image_source_dir': os.path.join(coco_path, 'train2014'),
    'train_features_file': os.path.join(data_path, 'image_features.pb'),

    'validation_annotations_file': os.path.join(coco_path, 'captions_val2014.json'),
    'validation_image_source_dir': os.path.join(coco_path, 'val2014'),
    'validation_features_file': os.path.join(data_path, 'val_image_features.pb')
}

def set_active_config(config):
    for key, value in config.items():
        globals()[key] = value

ensure_dir(models_path)
ensure_dir(data_path)
ensure_dir(coco_path)
ensure_dir(logs_path)

set_active_config(small_config)