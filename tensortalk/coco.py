import os
import logging
import numpy as np
import json
import itertools
import zipfile

from nltk import FreqDist, word_tokenize
from bisect import bisect
from random import randint
from collections import defaultdict

import config
from utils import ensure_file

class Vocab:
    def __init__(self, words, words_count):
        self.index2word = []
        self.word2index = dict()
        self.words_count = words_count
        self.accumulated_frequencies = []
        self.total_words = len(words)
        
        frequencies = FreqDist()
        accumulated = 0
        
        for word in words:
            frequencies[word] += 1
        
        for word, freq in frequencies.most_common(self.words_count - 1): #leave one index for rare words
            self.word2index[word] = len(self.word2index)
            self.index2word.append(word)
            accumulated += freq
            self.accumulated_frequencies.append(accumulated)
            
        self.index2word.append('UNK')
        self.accumulated_frequencies.append(self.total_words)
            
    def get_word(self, index, limit=None):
        if index == -1:
            return ' '
        elif limit is not None and index >= limit:
            return 'UNK'

        return self.index2word[index]
    
    def get_index(self, word):
        if word in self.word2index:
            return self.word2index[word]
        else:
            return self.words_count - 1
    
    def transform(self, sents):
        max_len = 0
        for sent in sents:
            if len(sent) > max_len:
                max_len = len(sent)

        result = np.zeros((len(sents), max_len), dtype=np.int32)
        for row, sent in enumerate(sents):
            result[row, :len(sent)] = np.asarray([1 + self.get_index(word.lower()) for word in sent])

        return result
    
    def sample(self, size):
        indices = np.zeros(size, dtype=np.int32)
        for i in range(size):
            index = randint(0, self.total_words)
            indices[i] = bisect(self.accumulated_frequencies, index)
            if indices[i] >= self.words_count:
                indices[i] = self.words_count - 1

        return indices

class CocoManager(object):
    def __init__(self, annotations_file, words_count, vocab=None):
        if not os.path.isfile(annotations_file):
            archive_path = os.path.join(config.base_path, 'captions_train-val2014.zip')
            ensure_file(archive_path, 'http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip')
            z = zipfile.ZipFile(archive_path)
            infolist = []
            for zipinfo in z.infolist():
                zipinfo.filename = os.path.basename(zipinfo.filename)
                infolist.append(zipinfo)

            z.extractall(config.coco_path, infolist)

        with open(annotations_file) as f:
            self.dataset = json.load(f)

        self.images = {image['id']: image for image in self.dataset['images']}

        self.img_to_anns = defaultdict(list)
        self.annotations = {}
        for ann in self.dataset['annotations']:
            self.annotations[ann['id']] = ann
            self.img_to_anns[ann['image_id']] += [ann]

        if vocab is not None:
            self.vocab = vocab
        else:
            all_words = []
            for ann in self.load_annotations(self.img_ids()):
                all_words += word_tokenize(ann['caption'].lower())
            self.vocab = Vocab(all_words, words_count)

    def img_ids(self):
        return self.images.keys()

    def load_annotations(self, img_ids):
        if type(img_ids) == list:
            ann_list = list(itertools.chain.from_iterable([self.img_to_anns[img_id] for img_id in self.img_ids()]))
        elif type(img_ids) == int:
            ann_list = self.img_to_anns[img_ids]
        else:
            raise TypeError('img_ids must be either int or list of ints')

        return [self.annotations[ann['id']] for ann in ann_list]

    def load_images(self, img_ids):
        if type(img_ids) == list:
            return [self.images[img_id] for img_id in img_ids]
        elif type(img_ids) == int:
            return [self.images[img_ids]]
        else:
            raise TypeError('img_ids must be either int or list of ints')

    def sents(self, img_ids):
        sents = [word_tokenize(ann['caption'].lower()) for ann in self.load_annotations(img_ids)]
        return self.vocab.transform(sents)