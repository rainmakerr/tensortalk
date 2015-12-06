import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np

import config

def weight_init(width, height):
    w = tf.random_uniform(minval = -2.45 / (width + height), maxval = 2.45 / (width + height), shape = [width, height])
    return tf.Variable(w)

def bias_init(shape):
    b = tf.zeros(shape)
    return tf.Variable(b)

class WrappedCell(rnn_cell.RNNCell):
    def __init__(self, num_units, num_outputs):
        self.lstm = rnn_cell.LSTMCell(num_units, num_units)
        self.w_softmax = weight_init(num_units, num_outputs)
        self.b_softmax = bias_init([num_outputs])

        self.num_units = num_units
        self.num_outputs = num_outputs

    @property
    def input_size(self):
        return self.num_units

    @property
    def state_size(self):
        return self.lstm.state_size

    @property
    def output_size(self):
        return self.num_outputs

    def __call__(self, inputs, state, scope=None):
        outputs, state = self.lstm(inputs, state, scope)
        softmax_outputs = tf.log(tf.nn.softmax(tf.matmul(outputs, self.w_softmax) + self.b_softmax))

        return softmax_outputs, state

class UserInputPipeline(object):
    def __init__(self):
        self.image_input = tf.placeholder('float32', shape=[None, config.image_features_count])
        self.text_input = tf.placeholder('int64', shape=[None, config.max_len])
        self.lens_input = tf.placeholder('int64', shape=[None, 1])
        self.result_input = tf.placeholder('float32', shape=[None, config.max_len, config.output_words_count + 1])

class TrainInputPipeline(object):
    def __init__(self, input_files, num_epochs, batch_size):
        filename_queue = tf.train.string_input_producer(input_files, num_epochs=num_epochs)
        reader = tf.TFRecordReader()
        _, records = reader.read(filename_queue)
        decoded = tf.parse_single_example(records,
            dense_keys=['image', 'text', 'result', 'len'],
            dense_types=['float', 'int64', 'int64', 'int64'],
            dense_shapes=[(1, config.image_features_count),
                          (config.sents_per_sample, config.max_len),
                          (config.sents_per_sample, config.max_len),
                          (config.sents_per_sample, 1)])

        self.image, self.text, self.result, self.lens = \
            decoded['image'], decoded['text'], decoded['result'], decoded['len']
        self.image = tf.concat(0, [self.image] * config.sents_per_sample)

        # result requires one-hot encoding
        clamped_result = tf.minimum(self.result, config.output_words_count)
        sliced_result = [tf.squeeze(tensor, [0]) for tensor in tf.split(0, config.sents_per_sample, clamped_result)]
        sliced_categorical_result = [self.to_categorical(tensor) for tensor in sliced_result]
        self.categorical_result = tf.concat(0, [tf.expand_dims(tensor, 0) for tensor in sliced_categorical_result])

        self.image_input, self.text_input, self.result_input, self.lens_input = tf.train.shuffle_batch(
            [self.image, self.text, self.categorical_result, self.lens],
            batch_size=batch_size,
            capacity=256+config.batch_size,
            min_after_dequeue=128,
            enqueue_many=True)

    @staticmethod
    def to_categorical(tensor):
        labels = tf.reshape(tensor, [-1, 1])
        indices = tf.reshape(tf.range(0, tf.shape(tensor)[0], 1), [-1, 1])
        merged = tf.concat(1, [indices, tf.cast(labels, 'int32')])
        return tf.sparse_to_dense(merged, [config.max_len, config.output_words_count + 1], 1.0, 0.0)

class CaptionNetwork(object):
    def __init__(self, session, input_pipeline):
        self.session = session
        self.input_pipeline = input_pipeline

        text_embeddings = weight_init(config.words_count + 2, config.hidden_count)

        embedded = tf.split(1, config.max_len, tf.nn.embedding_lookup(text_embeddings, input_pipeline.text_input))
        inputs = [tf.squeeze(input_, [1]) for input_ in embedded]

        w_image = weight_init(config.image_features_count, config.hidden_count)
        b_image = bias_init([config.hidden_count])

        image_transform = tf.matmul(input_pipeline.image_input, w_image) + b_image
        hidden_start = tf.concat(1, [tf.zeros_like(image_transform), image_transform])

        cell = WrappedCell(config.hidden_count, config.output_words_count + 1)
        probs_list, self.hidden = rnn.rnn(
            cell=cell,
            inputs=inputs,
            initial_state=hidden_start,
            sequence_length=input_pipeline.lens_input)
        self.probs = tf.concat(1, [tf.expand_dims(prob, 1) for prob in probs_list])

        float_lens = tf.cast(input_pipeline.lens_input, 'float')
        sample_losses = tf.reduce_sum(self.probs * input_pipeline.result_input, [1, 2]) / float_lens
        self.loss = -tf.reduce_mean(sample_losses)
        self.train_task = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
        self.loss_summary = tf.scalar_summary('loss', self.loss)

        self.saver = tf.train.Saver()

    def load(self, weights_file):
        self.saver.restore(self.session, weights_file)

    def save(self, weights_file):
        self.saver.save(self.session, weights_file)