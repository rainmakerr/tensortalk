import os
import argparse

from tensortalk import config
from tensortalk.coco import CocoManager
from tensortalk.sampler import BeamSearchSampler
from tensortalk.metrics import evaluate_sampler

def run_evaluation():
    models = [os.path.join(config.models_path, model) for model in os.listdir(config.models_path) if model.endswith('.ckpt')]
    bleu_scores = {}

    train_coco = CocoManager(config.train_annotations_file, config.words_count)
    validation_coco = CocoManager(config.validation_annotations_file, config.words_count, train_coco.vocab)

    for model_name in models:
        print 'Evaluating %s' % model_name
        bleu_scores[model_name] = evaluate_sampler(BeamSearchSampler(beam_size=5), validation_coco, model_name, limit=100)

    print 'Bleu scores'
    for name, score in bleu_scores.items():
        print name, ':', score

if __name__ == '__main__':
    run_evaluation()