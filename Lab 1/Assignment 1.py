import numpy as np
from enum import Enum


class DataMismatchError(Exception):
    pass


class NNData:
    """
    Object that manages training and testing data.

    """
    class Order(Enum):
        RANDOM = 0
        SEQUENTIAL = 1

    class Set(Enum):
        TRAIN = 0
        TEST = 1

    @staticmethod
    def percentage_limiter(percentage: float):
        if percentage < 0:
            return 0
        if percentage > 1:
            return 1
        if 0 <= percentage <= 1:
            return percentage

    def load_data(self, features=None, labels=None):
        if (features is None) or (labels is None):
            self._features = None
            self._labels = None
            return
        if len(features) != len(labels):
            self._features = None
            self._labels = None
            raise DataMismatchError
        try:
            self._features = np.array(features, dtype=float)
            self._labels = np.array(labels, dtype=float)
        except ValueError:
            self._features = None
            self._labels = None
            raise ValueError

    def __init__(self, features=None, labels=None, train_factor=0.9):
        if features is None:
            features = []
        if labels is None:
            labels = []
        self._features = None
        self._labels = None
        self._train_factor = NNData.percentage_limiter(train_factor)
        try:
            self.load_data(features, labels)
        except (ValueError, DataMismatchError):
            self._features = None
            self._labels = None


def load_XOR():
    features = [[0, 0], [1, 0], [0, 1], [1, 1]]
    labels = [[0], [1], [1], [0]]
    data = NNData(features, labels, 1)


def unit_test():
    test_object = NNData([0, 0, 1], [0, 2, 1], 1)
    print(f"Initial features and labels: {test_object._features} and {test_object._labels}")
    try:
        test_object.load_data([0, 0, 1], [0, 1])
    except DataMismatchError:
        print(f"After raising DataMismatchError, test_object._Features is: "
              f"{test_object._features} "
              f"and test_object._labels is: "
              f"{test_object._labels}")
        print("Test Case 1 Passed")
    test_object.load_data([0, 0, 1], [0, 2, 1])
    print(f"Initial features and labels: {test_object._features} and {test_object._labels}")
    try:
        test_object.load_data([0, 0, 1], [0, "dog", "elephant"])
    except ValueError:
        print(f"After raising ValueError, test_object._Features is: "
              f"{test_object._features} "
              f"and test_object._labels is: "
              f"{test_object._labels}")
        print("Test Case 2 Passed")
    test_object = NNData([0, 0, 1], [0, 1], 1)
    if (test_object._features is None) and (test_object._labels is None):
        print("Test Case 3 Passed")
    test_object = NNData([0, 0, 1], [0, 0, 1], -10)
    if test_object._train_factor == 0:
        print("Test Case 4 Passed")
    test_object = NNData([0, 0, 1], [0, 0, 1], 10)
    if test_object._train_factor == 1:
        print("Test Case 5 Passed")


def main():
    unit_test()
    load_XOR()


if __name__ == "__main__":
    main()
"""
"C:/Users/17147/PycharmProjects/CW2/Lab 1/Assignment 1.py"
Initial features and labels: [0. 0. 1.] and [0. 2. 1.]
After raising DataMismatchError, test_object._Features is: None and test_object._labels is: None
Test Case 1 Passed
Initial features and labels: [0. 0. 1.] and [0. 2. 1.]
After raising ValueError, test_object._Features is: None and test_object._labels is: None
Test Case 2 Passed
Test Case 3 Passed
Test Case 4 Passed
Test Case 5 Passed

Process finished with exit code 0
"""
