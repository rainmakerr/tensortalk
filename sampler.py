import numpy as np

import config

class CandidateSequence(object):
    def __init__(self, likelihood, sequence, probs):
        self.likelihood = likelihood
        self.sequence = sequence
        self.probs = probs

class BeamSearchSampler(object):
    def __init__(self, beam_size=32):
        self.beam_size = beam_size

    def sample(self, model, hidden_start, size=10):
        previous = [config.words_count + 1]

        candidates = []
        candidates.append(CandidateSequence(0., previous[:], []))

        for i in xrange(size):
            new_candidates = []
            for candidate in candidates:
                text_input = np.zeros((1, 30), dtype=np.int32)
                text_input[0, :len(candidate.sequence)] = np.int32(candidate.sequence)

                lens_input = len(candidate.sequence) * np.ones((1, 1), dtype=np.int32)

                feed_dict = {
                    model.input_pipeline.image_input: hidden_start,
                    model.input_pipeline.text_input: text_input.reshape(1, -1),
                    model.input_pipeline.lens_input: lens_input}

                model_output = model.session.run(model.probs, feed_dict=feed_dict)
                next_word = model_output[0, len(candidate.sequence) - 1, :]
                for index, word in enumerate(next_word):
                    new_candidates.append(CandidateSequence(candidate.likelihood + word,
                                                            candidate.sequence + [index],
                                                            candidate.probs + [next_word]))
            candidates = sorted(new_candidates, key=lambda x: x.likelihood)[-self.beam_size:]

        return [c.sequence[1:] for c in candidates]