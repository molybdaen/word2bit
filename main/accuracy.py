__author__ = 'johannesjurgovsky'

from numpy import exp, dot, zeros, outer, random, dtype, float32 as REAL,\
    uint32, seterr, array, uint8, vstack, argsort, fromstring, sqrt, newaxis,\
    ndarray, empty, sum as np_sum, prod, abs, sort, count_nonzero

from gensim import utils, matutils
from six import iteritems, itervalues, string_types


class WordAccuracy(object):

    def __init__(self, model, logger):
        self.model = model
        self.syn0 = model.syn0
        self.vocab = model.vocab
        self.index2word = model.index2word
        self.logger = logger
        self.init_sims()

    def init_sims(self, replace=False):
            """
            Precompute L2-normalized vectors.

            If `replace` is set, forget the original vectors and only keep the normalized
            ones = saves lots of memory!

            Note that you **cannot continue training** after doing a replace. The model becomes
            effectively read-only = you can call `most_similar`, `similarity` etc., but not `train`.

            """
            if getattr(self, 'syn0norm', None) is None or replace:
                self.logger.info("precomputing L2-norms of word weight vectors")
                if replace:
                    for i in xrange(self.syn0.shape[0]):
                        self.syn0[i, :] /= sqrt((self.syn0[i, :] ** 2).sum(-1))
                    self.syn0norm = self.syn0
                    if hasattr(self, 'syn1'):
                        del self.syn1
                else:
                    self.syn0norm = (self.syn0 / sqrt((self.syn0 ** 2).sum(-1))[..., newaxis]).astype(REAL)

    def most_similar(self, positive=[], negative=[], topn=10):
        """
        Find the top-N most similar words. Positive words contribute positively towards the
        similarity, negative words negatively.

        This method computes cosine similarity between a simple mean of the projection
        weight vectors of the given words, and corresponds to the `word-analogy` and
        `distance` scripts in the original word2vec implementation.

        Example::

          >>> trained_model.most_similar(positive=['woman', 'king'], negative=['man'])
          [('queen', 0.50882536), ...]

        """
        self.init_sims()

        if isinstance(positive, string_types) and not negative:
            # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
            positive = [positive]

        # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
        positive = [(word, 1.0) if isinstance(word, string_types + (ndarray,))
                                else word for word in positive]
        negative = [(word, -1.0) if isinstance(word, string_types + (ndarray,))
                                 else word for word in negative]

        # compute the weighted average of all words
        all_words, mean = set(), []
        for word, weight in positive + negative:
            if isinstance(word, ndarray):
                mean.append(weight * word)
            elif word in self.vocab:
                mean.append(weight * self.syn0norm[self.vocab[word].index])
                all_words.add(self.vocab[word].index)
            else:
                raise KeyError("word '%s' not in vocabulary" % word)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)

        dists = dot(self.syn0norm, mean)
        if not topn:
            return dists
        best = argsort(dists)[::-1][:topn + len(all_words)]
        # ignore (don't return) words from the input
        result = [(self.index2word[sim], float(dists[sim])) for sim in best if sim not in all_words]
        return result[:topn]

    def log_accuracy(self, section):
        correct, incorrect = len(section['correct']), len(section['incorrect'])
        if correct + incorrect > 0:
            self.logger.info("%s: %.1f%% (%i/%i)" %
                (section['section'], 100.0 * correct / (correct + incorrect),
                correct, correct + incorrect))

    def accuracy(self, questions, restrict_vocab=30000, most_similar=most_similar):
        """
        Compute accuracy of the model. `questions` is a filename where lines are
        4-tuples of words, split into sections by ": SECTION NAME" lines.
        See https://code.google.com/p/word2vec/source/browse/trunk/questions-words.txt for an example.

        The accuracy is reported (=printed to log and returned as a list) for each
        section separately, plus there's one aggregate summary at the end.

        Use `restrict_vocab` to ignore all questions containing a word whose frequency
        is not in the top-N most frequent words (default top 30,000).

        This method corresponds to the `compute-accuracy` script of the original C word2vec.

        """
        ok_vocab = dict(sorted(iteritems(self.vocab),
                               key=lambda item: -item[1].count)[:restrict_vocab])
        ok_index = set(v.index for v in itervalues(ok_vocab))

        sections, section = [], None
        for line_no, line in enumerate(utils.smart_open(questions)):
            # TODO: use level3 BLAS (=evaluate multiple questions at once), for speed
            line = utils.to_unicode(line)
            if line.startswith(': '):
                # a new section starts => store the old section
                if section:
                    sections.append(section)
                    self.log_accuracy(section)
                section = {'section': line.lstrip(': ').strip(), 'correct': [], 'incorrect': []}
            else:
                if not section:
                    raise ValueError("missing section header before line #%i in %s" % (line_no, questions))
                try:
                    a, b, c, expected = line.split()#[word.lower() for word in line.split()]  # TODO assumes vocabulary preprocessing uses lowercase, too...
                except:
                    self.logger.info("skipping invalid line #%i in %s" % (line_no, questions))
                if a not in ok_vocab or b not in ok_vocab or c not in ok_vocab or expected not in ok_vocab:
                    self.logger.debug("skipping line #%i with OOV words: %s" % (line_no, line.strip()))
                    continue

                ignore = set(self.vocab[v].index for v in [a, b, c])  # indexes of words to ignore
                predicted = None
                # find the most likely prediction, ignoring OOV words and input words
                for index in argsort(self.most_similar(positive=[b, c], negative=[a], topn=False))[::-1]:
                    if index in ok_index and index not in ignore:
                        predicted = self.index2word[index]
                        if predicted != expected:
                            self.logger.debug("%s: expected %s, predicted %s" % (line.strip(), expected, predicted))
                        break
                if predicted == expected:
                    section['correct'].append((a, b, c, expected))
                else:
                    section['incorrect'].append((a, b, c, expected))
        if section:
            # store the last section, too
            sections.append(section)
            self.log_accuracy(section)

        total = {
            'section': 'total',
            'correct': sum((s['correct'] for s in sections), []),
            'incorrect': sum((s['incorrect'] for s in sections), []),
        }
        self.log_accuracy(total)
        sections.append(total)
        return sections

if __name__ == "__main__":
    from gensim.models.word2vec import Word2Vec
    import logging
    import sys
    import numpy as np

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO, stream=sys.stdout)
    model = Word2Vec.load_word2vec_format("../data/model.w2v", fvocab="../data/vocab.w2v", binary=True, norm_only=False)
    wa = WordAccuracy(model, logging.getLogger())

    strt_percentage = 0.
    end_percentage = 1.
    step_percentage = 0.1
    syn0norm_sorted = sort(abs(wa.syn0norm.flatten()))
    l = len(syn0norm_sorted)
    for i in xrange(int(strt_percentage*l), int(end_percentage*l), int(step_percentage*l)):
        threshold_value = syn0norm_sorted[i]
        print "Threshold: %.3f" % threshold_value
        print "Zeroes: %d" % i
        wa.syn0norm[abs(wa.syn0norm)<threshold_value] = 0.
        relations = wa.accuracy("../data/questions-words.txt", restrict_vocab=30000)
