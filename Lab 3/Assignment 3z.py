import numpy as np
from enum import Enum
from collections import deque
import random
from abc import abstractmethod
import copy
import math


class DataMismatchError(Exception):
    pass


class LayerType(Enum):
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


class MultiLinkNode:
    class Side(Enum):
        UPSTREAM = 0
        DOWNSTREAM = 1

    def __init__(self):
        self._reporting_nodes = {MultiLinkNode.Side.UPSTREAM: 0, MultiLinkNode.Side.DOWNSTREAM: 0}
        self._reference_value = {MultiLinkNode.Side.UPSTREAM: 0, MultiLinkNode.Side.DOWNSTREAM: 0}
        self._neighbors = {MultiLinkNode.Side.UPSTREAM: [], MultiLinkNode.Side.DOWNSTREAM: []}

    def __str__(self):
        return f'{id(self)} is the ID of our node! ' \
               f'The neighboring nodes upstream are ' \
               f'{[id(item) for item in self._neighbors[MultiLinkNode.Side.UPSTREAM]]}. ' \
               f'The neighboring nodes downstream are ' \
               f'{[id(item) for item in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]]}.'

    @abstractmethod
    def _process_new_neighbor(self, node, side):
        pass

    def reset_neighbors(self, nodes: list, side: Side):
        nodes_copy = copy.copy(nodes)
        self._neighbors[side] = nodes_copy
        for node in nodes_copy:
            self._process_new_neighbor(node, side)
        self._reference_value[side] = int((math.pow(2, len(nodes_copy)))-1)


class Neurode(MultiLinkNode):
    def __init__(self, node_type, learning_rate=.05):
        self._value = 0
        self._node_type = node_type
        self._learning_rate = learning_rate
        self._weights = {}
        super().__init__()

    def _process_new_neighbor(self, node, side):
        if side is MultiLinkNode.Side.UPSTREAM:
            self._weights[node] = random.random()

    def _check_in(self, node, side=MultiLinkNode.Side.UPSTREAM):
        node_index = self._neighbors[side].index(node)
        self._reporting_nodes[side] |= 1 << node_index
        if self._reporting_nodes[side] == self._reference_value[side]:
            self._reporting_nodes[side] = 0
            return True
        return False

    def get_weight(self, node):
        return self._weights[node]

    @property
    def value(self):
        return self._value

    @property
    def node_type(self):
        return self._node_type

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, new_rate):
        self._learning_rate = new_rate


class FFNeurode(Neurode, MultiLinkNode):
    def __init__(self, my_type):
        super().__init__(my_type)

    @staticmethod
    def _sigmoid(value):
        return 1 / (1 + np.exp(-value))

    def _calculate_value(self):
        node_weighted_sum = 0
        for node in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            node_weighted_sum += (self.get_weight(node) * node.value)
        self._value = self._sigmoid(node_weighted_sum)

    def _fire_downstream(self):
        for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            node.data_ready_upstream(self)

    def data_ready_upstream(self, node):
        if self._check_in(node, MultiLinkNode.Side.UPSTREAM):
            self._calculate_value()
            self._fire_downstream()

    def set_input(self, input_value):
        self._value = input_value
        for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            node.data_ready_upstream(self)


class NNData:
    """
    Object that manages training and testing data for
    a Neural Network Application.

    """
    class Order(Enum):
        """
        Indicate whether data will be shuffled for each new epoch
        """
        RANDOM = 0
        SEQUENTIAL = 1

    class Set(Enum):
        """
        Indicate which set should be accessed or manipulated.
        """
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
        """
        Loads features and label data with some checks.
        """
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
        self.split_set()

    def __init__(self, features=None, labels=None, train_factor=0.9):
        if features is None:
            features = []
        if labels is None:
            labels = []
        self._features = None
        self._labels = None
        self._train_factor = NNData.percentage_limiter(train_factor)
        self._train_indices = []
        self._test_indices = []
        self._train_pool = deque()
        self._test_pool = deque()
        try:
            self.load_data(features, labels)
        except (ValueError, DataMismatchError):
            pass

    def split_set(self, new_train_factor=None):
        if new_train_factor is not None:
            self._train_factor = NNData.percentage_limiter(new_train_factor)
        size_loaded = len(self._features)
        size_training = int(self._train_factor * size_loaded)
        self._train_indices = random.sample(range(size_loaded),
                                            k=size_training)
        self._test_indices = list(set(range(size_loaded))-set(self._train_indices))
        random.shuffle(self._train_indices)
        random.shuffle(self._test_indices)

    def prime_data(self, target_set=None, order=None):
        if (target_set is None) or (target_set is NNData.Set.TRAIN):
            train_indices_temp = list(self._train_indices)
            self._train_pool = deque(train_indices_temp)
            if order is NNData.Order.RANDOM:
                random.shuffle(self._train_pool)
            if order is (NNData.Order.SEQUENTIAL or None):
                pass
        if (target_set is None) or (target_set is NNData.Set.TEST):
            test_indices_temp = list(self._test_indices)
            self._test_pool = deque(test_indices_temp)
            if order is NNData.Order.RANDOM:
                random.shuffle(self._test_pool)
            if order is (NNData.Order.SEQUENTIAL or None):
                pass

    def get_one_item(self, target_set=None):
        if target_set is None or target_set is NNData.Set.TRAIN:
            our_pool = self._train_pool
        if target_set is NNData.Set.TEST:
            our_pool = self._test_pool
        try:
            item = our_pool.popleft()
            return (self._features[item],
                    self._labels[item])
        except IndexError:
            return None

    def number_of_samples(self, target_set=None):
        if target_set is NNData.Set.TEST:
            return len(self._test_indices)
        if target_set is NNData.Set.TRAIN:
            return len(self._train_indices)
        if target_set is None:
            return (len(self._test_indices)
                    + len(self._train_indices))

    def pool_is_empty(self, target_set=None):
        if target_set is None or target_set is NNData.Set.TRAIN:
            our_pool = self._train_pool
        if target_set is NNData.Set.TEST:
            our_pool = self._test_pool
        if len(our_pool) > 0:
            return False
        if len(our_pool) == 0:
            return True


def load_XOR():
    """
    Loads XOR examples, with 100% in training.
    """
    features = [[0, 0], [1, 0], [0, 1], [1, 1]]
    labels = [[0], [1], [1], [0]]
    data = NNData(features, labels, 1)
    print(data._features)


def check_point_one_test():
    # Mock up a network with three inputs and three outputs

    inputs = [Neurode(LayerType.INPUT) for _ in range(3)]
    outputs = [Neurode(LayerType.OUTPUT, .01) for _ in range(3)]
    if not inputs[0]._reference_value[MultiLinkNode.Side.DOWNSTREAM] == 0:
        print("Fail - Initial reference value is not zero")
    for node in inputs:
        node.reset_neighbors(outputs, MultiLinkNode.Side.DOWNSTREAM)
    for node in outputs:
        node.reset_neighbors(inputs, MultiLinkNode.Side.UPSTREAM)
    if not inputs[0]._reference_value[MultiLinkNode.Side.DOWNSTREAM] == 7:
        print("Fail - Final reference value is not correct")
    if not inputs[0]._reference_value[MultiLinkNode.Side.UPSTREAM] == 0:
        print("Fail - Final reference value is not correct")

    # Report data ready from each input and make sure _check_in
    # only reports True when all nodes have reported

    if not outputs[0]._reporting_nodes[MultiLinkNode.Side.UPSTREAM] == 0:
        print("Fail - Initial reporting value is not zero")
    if outputs[0]._check_in(inputs[0], MultiLinkNode.Side.UPSTREAM):
        print("Fail - _check_in returned True but not all nodes were"
              "checked in")
    if not outputs[0]._reporting_nodes[MultiLinkNode.Side.UPSTREAM] == 1:
        print("Fail - reporting value is not correct")
    if outputs[0]._check_in(inputs[2], MultiLinkNode.Side.UPSTREAM):
        print("Fail - _check_in returned True but not all nodes were"
              "checked in")
    if not outputs[0]._reporting_nodes[MultiLinkNode.Side.UPSTREAM] == 5:
        print("Fail - reporting value is not correct")
    if outputs[0]._check_in(inputs[2], MultiLinkNode.Side.UPSTREAM):
        print("Fail - _check_in returned True but not all nodes were"
              "checked in (double fire)")
    if not outputs[0]._reporting_nodes[MultiLinkNode.Side.UPSTREAM] == 5:
        print("Fail - reporting value is not correct")
    if not outputs[0]._check_in(inputs[1], MultiLinkNode.Side.UPSTREAM):
        print("Fail - _check_in returned False after all nodes were"
              "checked in")

    # Report data ready from each output and make sure _check_in
    # only reports True when all nodes have reported

    if inputs[1]._check_in(outputs[0], MultiLinkNode.Side.DOWNSTREAM):
        print("Fail - _check_in returned True but not all nodes were"
              "checked in")
    if inputs[1]._check_in(outputs[2], MultiLinkNode.Side.DOWNSTREAM):
        print("Fail - _check_in returned True but not all nodes were"
              "checked in")
    if inputs[1]._check_in(outputs[0], MultiLinkNode.Side.DOWNSTREAM):
        print("Fail - _check_in returned True but not all nodes were"
              "checked in (double fire)")
    if not inputs[1]._check_in(outputs[1], MultiLinkNode.Side.DOWNSTREAM):
        print("Fail - _check_in returned False after all nodes were"
              "checked in")

    # Check that learning rates were set correctly

    if not inputs[0].learning_rate == .05:
        print("Fail - default learning rate was not set")
    if not outputs[0].learning_rate == .01:
        print("Fail - specified learning rate was not set")

    # Check that weights appear random

    weight_list = list()
    for node in outputs:
        for t_node in inputs:
            if node.get_weight(t_node) in weight_list:
                print("Fail - weights do not appear to be set up properly")
            weight_list.append(node.get_weight(t_node))


def check_point_two_test():
    inodes = []
    hnodes = []
    onodes = []
    for k in range(2):
        inodes.append(FFNeurode(LayerType.INPUT))
    for k in range(2):
        hnodes.append(FFNeurode(LayerType.HIDDEN))
    onodes.append(FFNeurode(LayerType.OUTPUT))
    for node in inodes:
        node.reset_neighbors(hnodes, MultiLinkNode.Side.DOWNSTREAM)
    for node in hnodes:
        node.reset_neighbors(inodes, MultiLinkNode.Side.UPSTREAM)
        node.reset_neighbors(onodes, MultiLinkNode.Side.DOWNSTREAM)
    for node in onodes:
        node.reset_neighbors(hnodes, MultiLinkNode.Side.UPSTREAM)
    try:
        inodes[1].set_input(1)
        assert onodes[0].value == 0
    except:
        print("Error: Neurodes may be firing before receiving all input")
    inodes[0].set_input(0)

    # Since input node 0 has value of 0 and input node 1 has value of
    # one, the value of the hidden layers should be the sigmoid of the
    # weight out of input node 1.

    value_0 = (1 / (1 + np.exp(-hnodes[0]._weights[inodes[1]])))
    value_1 = (1 / (1 + np.exp(-hnodes[1]._weights[inodes[1]])))
    inter = onodes[0]._weights[hnodes[0]] * value_0 + \
            onodes[0]._weights[hnodes[1]] * value_1
    final = (1 / (1 + np.exp(-inter)))
    try:
        print(final, onodes[0].value)
        assert final == onodes[0].value
        assert 0 < final < 1
    except:
        print("Error: Calculation of neurode value may be incorrect")


def main():
    load_XOR()
    check_point_one_test()
    check_point_two_test()


if __name__ == "__main__":
    main()

"""
"C:/Users/17147/PycharmProjects/CW2/Lab 3/Assignment 3z.py"
[[0. 0.]
 [1. 0.]
 [0. 1.]
 [1. 1.]]
0.6417601225944725 0.6417601225944725

Process finished with exit code 0
"""