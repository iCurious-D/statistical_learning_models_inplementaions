import numpy as np
from functools import partial
from Hidden_Markov_Model_Baum_Welch import baum_welch
from Hidden_Markov_Model_Viterbi import viterbi
from rich.table import Table
from rich.console import Console


class HMM:
    def __init__(self, state_size, observation_size, epsilon=1e-8, max_iteration=2000, verbose=False):
        self.state_size = state_size
        self.observation_size = observation_size
        self.epsilon = epsilon
        self.max_iteration = max_iteration
        self.verbose = verbose

    def fit(self, X):
        """
        When there is no label in the training data,
        HMM uses baum-welch for training.
        Otherwise, just counting the probability will be fine (not implemented here)
        """
        self.state2state, self.state2observation, self.initial_state = (
            baum_welch(X, self.state_size, self.observation_size, self.epsilon, self.max_iteration))

    def predict(self, X):
        """HMM uses viterbi for predicting"""
        Y = np.zeros_like(X)
        Y = np.apply_along_axis(
            partial(viterbi, self.state2state, self.state2observation, self.initial_state), -1, X)
        return Y


if __name__ == '__main__':
    def demonstrate(X, testX, desc):
        console = Console(markup=False)

        vocab = set(X.flatten())
        vocab_size = len(vocab)
        word2num = {word: num for num, word in enumerate(vocab)}

        f_word2num = np.vectorize(lambda word: word2num[word])

        numX, num_testX = map(f_word2num, (X, testX))

        hmm = HMM(4, vocab_size)
        hmm.fit(numX)
        pred = hmm.predict(num_testX)

        # show in table
        print(desc)
        table = Table()
        for x, p in zip(testX, pred):
            table.add_row(*map(str, x))
            table.add_row(*map(str, p))
        console.print(table)


    # ---------------------- Example 1 --------------------------------------------
    X = np.array([s.split() for s in
                  ['i am good .',
                   'i am bad .',
                   'you are good .',
                   'you are bad .',
                   'it is good .',
                   'it is bad .',
                   ]
                  ])
    testX = X
    demonstrate(X, testX, "Example 1")

    # ---------------------- Example 2 --------------------------------------------
    testX = np.array([s.split() for s in
                  ['you is good .',
                   'i are bad .',
                   'it are good .']
                  ])
    testX = np.concatenate([X, testX])
    demonstrate(X, testX, "Example 2")



