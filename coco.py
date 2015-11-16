from pycocotools.coco import COCO
import os
import logging
import numpy as np
import logging

from nltk import FreqDist
from nltk import word_tokenize
from bisect import bisect
from random import randint

import config

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
            
        self.index2word.append("Something_really_rare")
        self.accumulated_frequencies.append(self.total_words)
            
    def get_word(self, index):
        if index == -1:
            return ' '
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
        self.coco = COCO(annotations_file)
        img_ids = self.coco.getImgIds()
        ann_ids = self.coco.getAnnIds(img_ids)
        anns = self.coco.loadAnns(ann_ids)

        if vocab is not None:
            self.vocab = vocab
        else:
            all_words = []
            for ann in anns:
                all_words += word_tokenize(ann['caption'].lower())
            self.vocab = Vocab(all_words, words_count)

    def img_ids(self):
        return self.coco.getImgIds()

    def sents(self, img_ids):
        ann_ids = self.coco.getAnnIds(img_ids)
        anns = self.coco.loadAnns(ann_ids)
        sents = [word_tokenize(ann['caption'].lower()) for ann in anns]
        return self.vocab.transform(sents)