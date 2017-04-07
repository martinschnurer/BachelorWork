import sys, os
from dataset import Dataset

VALIDATION_DATASET_PATH = os.path.join(os.path.dirname(__file__), 'data/parsers/test/2')
TEST_DATASET_PATH = os.path.join(os.path.dirname(__file__), 'data/parsers/test/3')


class OvertrainDetector:
    def __init__(self, sess, tf_cross_entropy, model_ref, valid_dataset=None, howmany=1000):
        self.valid_dataset = valid_dataset if valid_dataset else open(VALIDATION_DATASET_PATH, 'r')
        self.actual_lowest = sys.maxsize
        self.model = model_ref

    # def isOvertrained(validate_fd, tokens, actual_lowest, sess, tf_cross_entropy, model, howmany=1000):
    def overtrainDetected(self):
        error_sum = 0

        for _ in range(self.howmany):
            _wordInput = DS.getWordGraphInput()
            _charInput = DS.getCharGraphInput()
            _target    = DS.getTarget()

            error_sum += sess.run(tf_cross_entropy, {
                                            word_inp: _wordInput,
                                            char_inp: _charInput,
                                            target: _target
                                        }
                                    )

        if error_sum < self.actual_lowest:
            self.actual_lowest = error_sum
            return False, error_sum

        return True, actual_lowest

#
