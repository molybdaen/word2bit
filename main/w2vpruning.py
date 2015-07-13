__author__ = 'johannesjurgovsky'

import matplotlib.pyplot as plt
from gensim.models.word2vec import Word2Vec
import numpy as np
from numpy import exp, dot, zeros, outer, random, dtype, float32 as REAL, uint32, seterr, array, uint8, vstack, argsort, fromstring, sqrt, newaxis, ndarray, empty, sum as np__sum, prod, abs, sort, count_nonzero
import cPickle
from os import path
from gensim import matutils, utils
from six import iteritems, itervalues, string_types
import logging
import gzip
from os.path import join
from general.config import DataEnum, DataType, Config
import codecs

def getEmbeddingsAndVocab(w2vModelFilename, rebuild=False):
    if path.exists(w2vModelFilename):
        p, f = path.split(w2vModelFilename)
        fName = f.split('.')[0]
        matFile = path.join(p, fName + "-mat.npy")
        vocFile = path.join(p, fName + "-voc.pkl")
        if not path.exists(matFile) or not path.exists(vocFile):
            model = Word2Vec.load_word2vec_format(w2vModelFilename, binary=False)
            np.save(matFile, model.syn0)
            cPickle.dump(model.vocab, open(vocFile, "w"))
        m = np.load(matFile)
        v = cPickle.load(open(vocFile, "r"))
        return m, v


class LineSentence(object):
    """Simple format: one sentence = one line; words already preprocessed and separated by whitespace."""
    def __init__(self, filename):
        """
        `source` can be either a string or a file object.

        Example::

            sentences = LineSentence('myfile.txt')

        Or for compressed files::

            sentences = LineSentence('compressed_text.txt.bz2')
            sentences = LineSentence('compressed_text.txt.gz')

        """
        self.source = codecs.open(filename, mode='r', encoding="cp1252")

    def any2utf8(self, text, errors='strict', encoding='utf8'):
        """Convert a string (unicode or bytestring in `encoding`), to bytestring in utf8."""
        if isinstance(text, unicode):
            return text
        # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
        return unicode(text, encoding, errors=errors)

    def __iter__(self):
        """Iterate through the lines in the source."""
        for line in self.source:
            yield line.split()#self.any2utf8(line).split()

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

    import logging
    import sys
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    ch.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(ch)

    def computeAccuracy(model, logger):

        wa = WordAccuracy(model, logger)

        strt_percentage = 0.
        end_percentage = 1.
        step_percentage = 0.1
        syn0norm_sorted = sort(abs(wa.syn0norm.flatten()))
        l = len(syn0norm_sorted)
        results = []
        for i in xrange(int(strt_percentage*l), int(end_percentage*l), int(step_percentage*l)):
            percentage = i/float(l)
            threshold_value = syn0norm_sorted[i]
            print "Percentage of zeroed values: %.2f" % (percentage*100.)
            print "Threshold value: %.3f" % threshold_value
            print "Number of zeroes: %d" % i
            wa.syn0norm[abs(wa.syn0norm)<threshold_value] = 0.
            relations = wa.accuracy("../data/eval/questions-words.txt", restrict_vocab=30000)
            result = {'percentage': percentage, 'thresholdValue': threshold_value, 'zeroes': i, 'relations': relations}
            results.append(result)
        return wa, results

    def cleanAccuracy(results):
        pruning_levels = []
        section_names = []
        sections_accs = []
        for pruning_step in range(len(results)):
            pruning_levels.append(results[pruning_step]['percentage'])
            for section in range(len(results[pruning_step]['relations'])):
                section_name = results[pruning_step]['relations'][section]['section']
                if section_name not in section_names:
                    section_names.append(section_name)
                    sections_accs.append([])
                acc = float(len(results[pruning_step]['relations'][section]['correct']))/(len(results[pruning_step]['relations'][section]['correct'])+len(results[pruning_step]['relations'][section]['incorrect']))
                sections_accs[section_names.index(section_name)].append(acc)
        return pruning_levels, section_names, sections_accs

    def plotAccuracies(size, pruning_levels, section_names, sections_accs):
        fig = plt.figure(4, figsize=(16,8))
        ax = fig.add_subplot(111, xlim=[-0.1, 1.1],ylim=[0.,1.], xlabel="Pruning Level [%]", ylabel="Accuracy [%]")
        ax.set_title("Accuracy of Word Embeddings for all relation types (d=%d)" % size)
        ax.grid(True)
        lineStyles, lineColors = ["-","--","-.",":"], ['b', 'g', 'r', 'c', 'm', 'y']
        for i, s in enumerate(section_names):
            lstyle = lineStyles[int(i/len(lineColors))%len(lineStyles)] + ('o' if s == "total" else '*') + lineColors[i%len(lineColors)]                                                                          #lstyle = '-o' if s == "total" else '-*'
            ax.plot(pruning_levels, sections_accs[i], lstyle, label=s)
        ax.legend(numpoints=1)
        plt.savefig("../data/eval/wikipedia/accuracy-%d.pdf" % size, format='pdf')
        plt.show()

    def plotAccuraciesPerRelation(results):
        relations = results[0]['section_names']
        for i, r in enumerate(relations):
            pruning_levels = []
            accuracies = []
            sizes = []
            for result in results:
                pruning_levels = result['pruning_levels']
                accuracies.append(result['sections_accs'][i])
                sizes.append(result['size'])

            fig = plt.figure(i, figsize=(16,8))
            ax = fig.add_subplot(111, xlim=[-0.1, 1.1],ylim=[0.,1.], xlabel="Pruning Level [%]", ylabel="Accuracy [%]")
            ax.set_title("Accuracy of Word Embeddings (Relation: %s)" % r)
            ax.grid(True)
            lineStyles, lineColors = ["-","--","-.",":"], ['b', 'g', 'r', 'c', 'm', 'y']
            for i, s in enumerate(sizes):
                lstyle = lineStyles[int(i/len(lineColors))%len(lineStyles)] + ('o' if r == "total" else '*') + lineColors[i%len(lineColors)]                                                                          #lstyle = '-o' if s == "total" else '-*'
                ax.plot(pruning_levels, accuracies[i], lstyle, label="d = %d" % s)
            ax.legend(numpoints=1)
            fig.savefig("../data/eval/wikipedia/accuracy-%s.pdf" % r, format='pdf')


    #logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    #sentences_path = join("../data/corpora/wikipedia/train/train_all_data_1000000000.txt")

    sizes = [40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 500]

    # for size in sizes:
    #     sentences = LineSentence(sentences_path)
    #     model = Word2Vec(sg=1, hs=0, negative=20, min_count=100, size=size, workers=4, window=9) # an empty model, no training
    #     model.build_vocab(sentences=sentences)  # can be a non-repeatable, 1-pass generator
    #     model.train(sentences=LineSentence(sentences_path))  # can be a non-repeatable, 1-pass generator
    #     model.save_word2vec_format("../data/models/wikipedia/model-%d.w2v" % size, fvocab="../data/models/wikipedia/vocab-%d.w2v" % size, binary=True)
    #
    #     model = Word2Vec.load_word2vec_format("../data/models/wikipedia/model-%d.w2v" % size, fvocab="../data/models/wikipedia/vocab-%d.w2v" % size, binary=True, norm_only=False)
    #     emb = model.syn0
    #     vocab = model.vocab
    #
    #     wordAccuracy, results = computeAccuracy(model, logger)
    #     pruning_levels, section_names, sections_accs = cleanAccuracy(results)
    #     result = {'size': size, 'pruning_levels': pruning_levels, 'section_names': section_names, 'sections_accs': sections_accs}
    #     cPickle.dump(result, open("../data/eval/wikipedia/result-%d.pkl" % size, 'wb'))

    results = []
    for size in sizes:
        result = cPickle.load(open("../data/eval/wikipedia/result-%d.pkl" % size, 'rb'))
        results.append(result)
        plotAccuracies(size, result['pruning_levels'], result['section_names'], result['sections_accs'])

    plotAccuraciesPerRelation(results)
