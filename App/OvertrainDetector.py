import sys, os
from dataset import Dataset
import tensorflow

VALIDATION_DATASET_PATH = os.path.join(os.path.dirname(__file__), 'data/parsers/test/2')
TEST_DATASET_PATH = os.path.join(os.path.dirname(__file__), 'data/parsers/test/3')


class OvertrainDetector:
    def __init__(self, sess, tf_cross_entropy, model_ref, validation_dataset=None, howmany=1000):
        self.actual_lowest = sys.maxsize
        self.model = model_ref
        self.howmany = howmany
        self.sess = sess
        self.cross_entropy = tf_cross_entropy

        if validation_dataset:
            self.DS = validation_dataset
        else:
            self.DS = Dataset(
                            model=model_ref,
                            file_dataset=open(VALIDATION_DATASET_PATH, 'r')
                            )

    # def isOvertrained(validate_fd, tokens, actual_lowest, sess, tf_cross_entropy, model, howmany=1000):
    def overtrainDetected(self):
        error_sum = 0

        for _ in range(self.howmany):
            _wordInput = self.DS.getWordGraphInput()
            _charInput = self.DS.getCharGraphInput()
            _target    = self.DS.getTarget()

            feed_dict_ = {
                'word_inp' : _wordInput,
                'char_inp' : _charInput,
                'target'   : _target
            }

            error_sum += self.sess.run(self.cross_entropy, feed_dict=feed_dict_)

        if error_sum < self.actual_lowest:
            self.actual_lowest = error_sum
            return False, error_sum

        return True, actual_lowest

#
