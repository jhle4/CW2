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


class Node:
    def __init__(self, data=None):
        self.data = data
        self.next = None
        self.previous = None


class DoublyLinkedList:
    class EmptyListError(Exception):
        pass

    def __init__(self):
        self._head = None
        self._tail = None
        self._curr = self._head

    def reset_to_head(self):
        self._curr = self._head
        if self._curr is None:
            raise DoublyLinkedList.EmptyListError
        else:
            return self._curr.data

    def reset_to_tail(self):
        self._curr = self._tail
        if self._curr is None:
            raise DoublyLinkedList.EmptyListError
        else:
            return self._curr.data

    def move_forward(self):
        if self._curr is None:
            raise DoublyLinkedList.EmptyListError
        elif self._curr == self._tail:
            raise IndexError
        else:
            self._curr = self._curr.next
        return self._curr.data

    def move_back(self):
        if self._curr is None:
            raise DoublyLinkedList.EmptyListError
        elif self._curr == self._head:
            raise IndexError
        else:
            self._curr = self._curr.previous
        return self._curr.data

    def add_to_head(self, data):
        new_node = Node(data)
        new_node.next = self._head
        if self._head is None:
            self._tail = new_node
        else:
            self._head.previous = new_node
        self._head = new_node
        self.reset_to_head()

    def remove_from_head(self):
        if self._head is None:
            raise DoublyLinkedList.EmptyListError
        ret_val = self._head.data
        if self._head == self._tail:
            self._tail = None
        self._head.next.previous = None
        self._head = self._head.next
        self.reset_to_head()
        return ret_val

    def add_after_cur(self, data):
        if self._curr is None:
            self.add_to_head(data)
            return
        new_node = Node(data)
        if self._curr == self._tail:
            self._tail = new_node
        else:
            new_node.next = self._curr.next
            new_node.next.previous = new_node
        self._curr.next = new_node
        new_node.previous = self._curr

    def remove_after_cur(self):  # need to fix to remove node from list
        if self._curr is None:
            raise DoublyLinkedList.EmptyListError
        if self._curr == self._tail:
            raise IndexError
        ret_val = self._curr.next.data
        if self._curr.next == self._tail:
            self._tail.previous = None
            self._tail = self._curr
            self._curr.next = None
            return ret_val
        self._curr.next = self._curr.next.next
        self._curr.next.previous = self._curr
        return ret_val

    def get_current_data(self):
        if self._head is None:
            raise DoublyLinkedList.EmptyListError
        return self._curr.data


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


class FFNeurode(Neurode):
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


class BPNeurode(Neurode):
    def __init__(self, my_type):
        self._delta = 0
        super().__init__(my_type)

    @staticmethod
    def _sigmoid_derivative(value):
        return value * (1-value)

    def _calculate_delta(self, expected_value=None):
        if self.node_type is LayerType.OUTPUT:
            self._delta = (expected_value - self.value) * (self._sigmoid_derivative(self.value))
        if self.node_type is not LayerType.OUTPUT:
            delta_weighted_sum = 0
            for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
                delta_weighted_sum += node.get_weight(self) * node.delta
            self._delta = delta_weighted_sum * self._sigmoid_derivative(self.value)

    def data_ready_downstream(self, node):
        if self._check_in(node, MultiLinkNode.Side.DOWNSTREAM):
            self._calculate_delta()
            self._fire_upstream()
            self._update_weights()

    def set_expected(self, expected_value):
        self._calculate_delta(expected_value)
        for node in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            node.data_ready_downstream(self)

    def adjust_weights(self, node, adjustment):
        self._weights[node] += adjustment

    def _update_weights(self):
        for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            adjustment_calc = self.value * node.learning_rate * node.delta
            node.adjust_weights(self, adjustment_calc)

    def _fire_upstream(self):
        for node in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            node.data_ready_downstream(self)

    @property
    def delta(self):
        return self._delta


class FFBPNeurode(FFNeurode, BPNeurode):
    pass


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


def dll_test():
    my_list = DoublyLinkedList()
    try:
        my_list.get_current_data()
    except DoublyLinkedList.EmptyListError:
        print("Pass")
    else:
        print("Fail")
    for a in range(3):
        my_list.add_to_head(a)
    if my_list.get_current_data() != 2:
        print("Error")
    my_list.move_forward()
    if my_list.get_current_data() != 1:
        print("Fail")
    my_list.move_forward()
    try:
        my_list.move_forward()
    except IndexError:
        print("Pass")
    else:
        print("Fail")
    if my_list.get_current_data() != 0:
        print("Fail")
    my_list.move_back()
    my_list.remove_after_cur()
    if my_list.get_current_data() != 1:
        print("Fail")
    my_list.move_back()
    if my_list.get_current_data() != 2:
        print("Fail")
    try:
        my_list.move_back()
    except IndexError:
        print("Pass")
    else:
        print("Fail")
    my_list.move_forward()
    if my_list.get_current_data() != 1:
        print("Fail")


def main():
    load_XOR()
    dll_test()


if __name__ == "__main__":
    main()

"""
"C:/Users/chuck/PycharmProjects/CW2/Lab 4/Assignment 4 REVISED.py"
[[0. 0.]
 [1. 0.]
 [0. 1.]
 [1. 1.]]
Pass
Pass
Pass

Process finished with exit code 0
"""
