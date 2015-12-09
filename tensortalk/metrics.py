import numpy as np
import os
from collections import defaultdict
import tensorflow as tf

import config
from utils import logger
from network import CaptionNetwork, TrainInputPipeline, UserInputPipeline
from image import ImageManager
from coco import CocoManager
from sampler import BeamSearchSampler

def make_ngrams(sent, n):
    return [tuple(sent[i:i+n]) for i in range(len(sent) - n + 1)]

def bleu_score(candidate, references, n=4):
    candidate_ngrams = make_ngrams(candidate, n)
    reference_ngrams = [make_ngrams(reference, n) for reference in references]

    reference_set = defaultdict(int)
    for ngram_list in reference_ngrams:
        current_reference_set = defaultdict(int)
        for ngram in ngram_list:
            current_reference_set[ngram] += 1

        for ngram, count in current_reference_set.items():
            if reference_set[ngram] < count:
                reference_set[ngram] = count

    hits = 0
    total = 0

    for ngram in candidate_ngrams:
        total += 1
        if reference_set[ngram] > 0:
            hits += 1
            reference_set[ngram] -= 1

    return float(hits) / total

def evaluate_sampler(sampler, coco_manager, model_file, limit=None):
    graph = tf.Graph()
    session = tf.Session(graph=graph)
    with graph.as_default():
        feed_pipeline = UserInputPipeline()
        net = CaptionNetwork(session, feed_pipeline)
        net.load(model_file)
        load_pipeline = TrainInputPipeline([config.validation_features_file], num_epochs=1, batch_size=32)
        unitialized = [v for v in tf.all_variables() if v.name.startswith('input')]
        session.run(tf.initialize_variables(unitialized))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)

        processed = 0
        total_score = 0.

        try:
            while not coord.should_stop():
                image, text, result = session.run([load_pipeline.image, load_pipeline.text, load_pipeline.result])

                candidate = [word_id for word_id in sampler.sample(net, image[0 : 1], size=15)[-1] if word_id != 0]
                score = bleu_score(candidate, text, n=4)
                total_score += score
                processed += 1

                logger().info('Current score: %f', score)
                logger().info('Batch %d completed, average score is %f', processed, total_score / processed)

                candidate_words = [coco_manager.vocab.get_word(i - 1, limit=config.output_words_count - 1) for i in candidate]
                logger().info('Candidate sentence: %s', ' '.join(candidate_words))
                for sequence in result:
                    reference_words = [coco_manager.vocab.get_word(i - 1) for i in sequence]
                    logger().info('Reference sentence: %s', ' '.join(reference_words))

                if limit is not None and processed >= limit:
                    raise StopIteration()

        except tf.errors.OutOfRangeError:
            logger().info('Done')
        except StopIteration:
            logger().info('Limit reached')
        finally:
            coord.request_stop()

        coord.join(threads)
        session.close()

    return total_score / processed