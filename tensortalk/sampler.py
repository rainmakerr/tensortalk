import numpy as np

import config

class CandidateSequence(object):
    def __init__(self, likelihood, sequence, probs):
        self.likelihood = likelihood
        self.sequence = sequence
        self.probs = probs

class BeamSearchSampler(object):
    def __init__(self, beam_size=5):
        self.beam_size = beam_size

    def sample(self, model, hidden_start, size=10):
        previous = [config.words_count + 1]

        candidates = []
        candidates.append(CandidateSequence(0., previous[:], []))

        for i in xrange(size):
            new_candidates = []
            text_input = np.zeros((self.beam_size, 30), dtype=np.int32)
            lens_input = np.zeros((self.beam_size, 1), dtype=np.int32)
            image_input = hidden_start + np.zeros((self.beam_size,) + hidden_start.shape[1:], dtype=np.float32)

            for candidate_id, candidate in enumerate(candidates):
                text_input[candidate_id, :len(candidate.sequence)] = np.int32(candidate.sequence)
                lens_input[candidate_id] = len(candidate.sequence)

            feed_dict = {
                model.input_pipeline.image_input: image_input,
                model.input_pipeline.text_input: text_input,
                model.input_pipeline.lens_input: lens_input}

            model_output = model.session.run(model.probs, feed_dict=feed_dict)
            next_word = model_output[:, len(candidate.sequence) - 1, :]

            for candidate_id, candidate in enumerate(candidates):
                for index, word in enumerate(next_word[candidate_id]):
                    new_candidates.append(CandidateSequence(candidate.likelihood + word,
                                                            candidate.sequence + [index],
                                                            candidate.probs + [next_word]))
            candidates = sorted(new_candidates, key=lambda x: x.likelihood)[-self.beam_size:]

        return [c.sequence[1:] for c in candidates]