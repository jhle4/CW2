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
            self._tail = self._curr
            self._tail.next = None
        else:
            self._curr.next = self._curr.next.next
            self._curr.next.previous = self._curr
        return ret_val

    def get_current_data(self):
        if self._head is None:
            raise DoublyLinkedList.EmptyListError
        return self._curr.data


class LayerList(DoublyLinkedList):
    def link(self):
        for neurode in self._curr.next.data:
            neurode.reset_neighbors(self._curr.data, FFBPNeurode.Side.UPSTREAM)
        for neurode in self._curr.data:
            neurode.reset_neighbors(self._curr.next.data, FFBPNeurode.Side.DOWNSTREAM)

    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        input_list = [FFBPNeurode(LayerType.INPUT) for _ in range(inputs)]
        output_list = [FFBPNeurode(LayerType.OUTPUT) for _ in range(outputs)]
        self.add_to_head(input_list)
        self.add_after_cur(output_list)
        self.link()

    def add_layer(self, num_nodes: int):
        if self._curr == self._tail:
            raise IndexError
        hidden_layer = [FFBPNeurode(LayerType.HIDDEN) for _ in range(num_nodes)]
        self.add_after_cur(hidden_layer)
        self.link()
        self.move_forward()
        self.link()
        self.move_back()

    def remove_layer(self):
        if self._curr == self._tail or self._curr.next == self._tail:
            raise IndexError
        self.remove_after_cur()
        self.link()

    @property
    def input_nodes(self):
        return self._head.data

    @property
    def output_nodes(self):
        return self._tail.data


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


class FFBPNetwork:
    class EmptySetException(Exception):
        pass

    def __init__(self, num_inputs: int, num_outputs: int):
        self.layers = LayerList(num_inputs, num_outputs)
        self._inputs = num_inputs
        self._outputs = num_outputs

    def add_hidden_layer(self, num_nodes: int, position=0):
        self.layers.reset_to_head()
        for p in range(position):
            self.layers.move_forward()
        self.layers.add_layer(num_nodes)

    def train(self, data_set: NNData, epochs=1000, verbosity=2, order=NNData.Order.RANDOM):
        if data_set.number_of_samples(NNData.Set.TRAIN) == 0:
            raise FFBPNetwork.EmptySetException
        for epoch in range(epochs):
            data_set.prime_data(NNData.Set.TRAIN, order=order)
            sum_error = 0
            while not data_set.pool_is_empty(NNData.Set.TRAIN):
                x, y = data_set.get_one_item(NNData.Set.TRAIN)
                zipped = zip(x, self.layers.input_nodes)
                for val, node in zipped:
                    node.set_input(val)
                produced = []
                zipped = zip(y, self.layers.output_nodes)
                for val, node in zipped:
                    node.set_expected(val)
                    sum_error += (node.value - val) ** 2 / self._outputs
                    produced.append(node.value)
                if epoch % 1000 == 0 and verbosity > 1:
                    print("Sample:", x, "expected:", y, "produced:", produced)
            if epoch % 100 == 0 and verbosity > 0:
                print("Epoch", epoch, "RMSE = ", math.sqrt(sum_error / data_set.number_of_samples(NNData.Set.TRAIN)))
        print("Final Epoch RMSE = ", math.sqrt(sum_error / data_set.number_of_samples(NNData.Set.TRAIN)))

    def test(self, data_set: NNData, order=NNData.Order.SEQUENTIAL):
        if data_set.number_of_samples(NNData.Set.TEST) == 0:
            raise FFBPNetwork.EmptySetException
        data_set.prime_data(NNData.Set.TEST, order=order)
        sum_error = 0
        while not data_set.pool_is_empty(NNData.Set.TEST):
            x, y = data_set.get_one_item(NNData.Set.TEST)
            zipped = zip(x, self.layers.input_nodes)
            for val, node in zipped:
                node.set_input(val)
            produced = []
            zipped = zip(y, self.layers.output_nodes)
            for val, node in zipped:
                node.set_expected(val)
                sum_error += (node.value - val) ** 2 / self._outputs
                produced.append(node.value)
            print("Sample:", x, "expected:", y, "produced:", produced)
        print("Final Epoch RMSE = ", math.sqrt(sum_error / data_set.number_of_samples(NNData.Set.TEST)))


def load_XOR():
    """
    Loads XOR examples, with 100% in training.
    """
    features = [[0, 0], [1, 0], [0, 1], [1, 1]]
    labels = [[0], [1], [1], [0]]
    data = NNData(features, labels, 1)


def run_iris():
    network = FFBPNetwork(4, 3)
    network.add_hidden_layer(3)
    Iris_X = [[5.1, 3.5, 1.4, 0.2], [4.9, 3, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2], [4.6, 3.1, 1.5, 0.2],
              [5, 3.6, 1.4, 0.2], [5.4, 3.9, 1.7, 0.4], [4.6, 3.4, 1.4, 0.3], [5, 3.4, 1.5, 0.2],
              [4.4, 2.9, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1], [5.4, 3.7, 1.5, 0.2], [4.8, 3.4, 1.6, 0.2],
              [4.8, 3, 1.4, 0.1], [4.3, 3, 1.1, 0.1], [5.8, 4, 1.2, 0.2], [5.7, 4.4, 1.5, 0.4],
              [5.4, 3.9, 1.3, 0.4], [5.1, 3.5, 1.4, 0.3], [5.7, 3.8, 1.7, 0.3], [5.1, 3.8, 1.5, 0.3],
              [5.4, 3.4, 1.7, 0.2], [5.1, 3.7, 1.5, 0.4], [4.6, 3.6, 1, 0.2], [5.1, 3.3, 1.7, 0.5],
              [4.8, 3.4, 1.9, 0.2], [5, 3, 1.6, 0.2], [5, 3.4, 1.6, 0.4], [5.2, 3.5, 1.5, 0.2],
              [5.2, 3.4, 1.4, 0.2], [4.7, 3.2, 1.6, 0.2], [4.8, 3.1, 1.6, 0.2], [5.4, 3.4, 1.5, 0.4],
              [5.2, 4.1, 1.5, 0.1], [5.5, 4.2, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1], [5, 3.2, 1.2, 0.2],
              [5.5, 3.5, 1.3, 0.2], [4.9, 3.1, 1.5, 0.1], [4.4, 3, 1.3, 0.2], [5.1, 3.4, 1.5, 0.2],
              [5, 3.5, 1.3, 0.3], [4.5, 2.3, 1.3, 0.3], [4.4, 3.2, 1.3, 0.2], [5, 3.5, 1.6, 0.6],
              [5.1, 3.8, 1.9, 0.4], [4.8, 3, 1.4, 0.3], [5.1, 3.8, 1.6, 0.2], [4.6, 3.2, 1.4, 0.2],
              [5.3, 3.7, 1.5, 0.2], [5, 3.3, 1.4, 0.2], [7, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5],
              [6.9, 3.1, 4.9, 1.5], [5.5, 2.3, 4, 1.3], [6.5, 2.8, 4.6, 1.5], [5.7, 2.8, 4.5, 1.3],
              [6.3, 3.3, 4.7, 1.6], [4.9, 2.4, 3.3, 1], [6.6, 2.9, 4.6, 1.3], [5.2, 2.7, 3.9, 1.4], [5, 2, 3.5, 1],
              [5.9, 3, 4.2, 1.5], [6, 2.2, 4, 1], [6.1, 2.9, 4.7, 1.4], [5.6, 2.9, 3.6, 1.3], [6.7, 3.1, 4.4, 1.4],
              [5.6, 3, 4.5, 1.5], [5.8, 2.7, 4.1, 1], [6.2, 2.2, 4.5, 1.5], [5.6, 2.5, 3.9, 1.1],
              [5.9, 3.2, 4.8, 1.8], [6.1, 2.8, 4, 1.3], [6.3, 2.5, 4.9, 1.5], [6.1, 2.8, 4.7, 1.2],
              [6.4, 2.9, 4.3, 1.3], [6.6, 3, 4.4, 1.4], [6.8, 2.8, 4.8, 1.4], [6.7, 3, 5, 1.7], [6, 2.9, 4.5, 1.5],
              [5.7, 2.6, 3.5, 1], [5.5, 2.4, 3.8, 1.1], [5.5, 2.4, 3.7, 1], [5.8, 2.7, 3.9, 1.2],
              [6, 2.7, 5.1, 1.6], [5.4, 3, 4.5, 1.5], [6, 3.4, 4.5, 1.6], [6.7, 3.1, 4.7, 1.5],
              [6.3, 2.3, 4.4, 1.3], [5.6, 3, 4.1, 1.3], [5.5, 2.5, 4, 1.3], [5.5, 2.6, 4.4, 1.2],
              [6.1, 3, 4.6, 1.4], [5.8, 2.6, 4, 1.2], [5, 2.3, 3.3, 1], [5.6, 2.7, 4.2, 1.3], [5.7, 3, 4.2, 1.2],
              [5.7, 2.9, 4.2, 1.3], [6.2, 2.9, 4.3, 1.3], [5.1, 2.5, 3, 1.1], [5.7, 2.8, 4.1, 1.3],
              [6.3, 3.3, 6, 2.5], [5.8, 2.7, 5.1, 1.9], [7.1, 3, 5.9, 2.1], [6.3, 2.9, 5.6, 1.8],
              [6.5, 3, 5.8, 2.2], [7.6, 3, 6.6, 2.1], [4.9, 2.5, 4.5, 1.7], [7.3, 2.9, 6.3, 1.8],
              [6.7, 2.5, 5.8, 1.8], [7.2, 3.6, 6.1, 2.5], [6.5, 3.2, 5.1, 2], [6.4, 2.7, 5.3, 1.9],
              [6.8, 3, 5.5, 2.1], [5.7, 2.5, 5, 2], [5.8, 2.8, 5.1, 2.4], [6.4, 3.2, 5.3, 2.3], [6.5, 3, 5.5, 1.8],
              [7.7, 3.8, 6.7, 2.2], [7.7, 2.6, 6.9, 2.3], [6, 2.2, 5, 1.5], [6.9, 3.2, 5.7, 2.3],
              [5.6, 2.8, 4.9, 2], [7.7, 2.8, 6.7, 2], [6.3, 2.7, 4.9, 1.8], [6.7, 3.3, 5.7, 2.1],
              [7.2, 3.2, 6, 1.8], [6.2, 2.8, 4.8, 1.8], [6.1, 3, 4.9, 1.8], [6.4, 2.8, 5.6, 2.1],
              [7.2, 3, 5.8, 1.6], [7.4, 2.8, 6.1, 1.9], [7.9, 3.8, 6.4, 2], [6.4, 2.8, 5.6, 2.2],
              [6.3, 2.8, 5.1, 1.5], [6.1, 2.6, 5.6, 1.4], [7.7, 3, 6.1, 2.3], [6.3, 3.4, 5.6, 2.4],
              [6.4, 3.1, 5.5, 1.8], [6, 3, 4.8, 1.8], [6.9, 3.1, 5.4, 2.1], [6.7, 3.1, 5.6, 2.4],
              [6.9, 3.1, 5.1, 2.3], [5.8, 2.7, 5.1, 1.9], [6.8, 3.2, 5.9, 2.3], [6.7, 3.3, 5.7, 2.5],
              [6.7, 3, 5.2, 2.3], [6.3, 2.5, 5, 1.9], [6.5, 3, 5.2, 2], [6.2, 3.4, 5.4, 2.3], [5.9, 3, 5.1, 1.8]]
    Iris_Y = [[1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ]]
    data = NNData(Iris_X, Iris_Y, .7)
    network.train(data, 10001, order=NNData.Order.RANDOM)
    network.test(data)


def run_sin():
    network = FFBPNetwork(1, 1)
    network.add_hidden_layer(3)
    sin_X = [[0], [0.01], [0.02], [0.03], [0.04], [0.05], [0.06], [0.07], [0.08], [0.09], [0.1], [0.11], [0.12],
             [0.13], [0.14], [0.15], [0.16], [0.17], [0.18], [0.19], [0.2], [0.21], [0.22], [0.23], [0.24], [0.25],
             [0.26], [0.27], [0.28], [0.29], [0.3], [0.31], [0.32], [0.33], [0.34], [0.35], [0.36], [0.37], [0.38],
             [0.39], [0.4], [0.41], [0.42], [0.43], [0.44], [0.45], [0.46], [0.47], [0.48], [0.49], [0.5], [0.51],
             [0.52], [0.53], [0.54], [0.55], [0.56], [0.57], [0.58], [0.59], [0.6], [0.61], [0.62], [0.63], [0.64],
             [0.65], [0.66], [0.67], [0.68], [0.69], [0.7], [0.71], [0.72], [0.73], [0.74], [0.75], [0.76], [0.77],
             [0.78], [0.79], [0.8], [0.81], [0.82], [0.83], [0.84], [0.85], [0.86], [0.87], [0.88], [0.89], [0.9],
             [0.91], [0.92], [0.93], [0.94], [0.95], [0.96], [0.97], [0.98], [0.99], [1], [1.01], [1.02], [1.03],
             [1.04], [1.05], [1.06], [1.07], [1.08], [1.09], [1.1], [1.11], [1.12], [1.13], [1.14], [1.15], [1.16],
             [1.17], [1.18], [1.19], [1.2], [1.21], [1.22], [1.23], [1.24], [1.25], [1.26], [1.27], [1.28], [1.29],
             [1.3], [1.31], [1.32], [1.33], [1.34], [1.35], [1.36], [1.37], [1.38], [1.39], [1.4], [1.41], [1.42],
             [1.43], [1.44], [1.45], [1.46], [1.47], [1.48], [1.49], [1.5], [1.51], [1.52], [1.53], [1.54], [1.55],
             [1.56], [1.57]]
    sin_Y = [[0], [0.00999983333416666], [0.0199986666933331], [0.0299955002024957], [0.0399893341866342],
             [0.0499791692706783], [0.0599640064794446], [0.0699428473375328], [0.0799146939691727],
             [0.089878549198011], [0.0998334166468282], [0.109778300837175], [0.119712207288919],
             [0.129634142619695], [0.139543114644236], [0.149438132473599], [0.159318206614246],
             [0.169182349066996], [0.179029573425824], [0.188858894976501], [0.198669330795061], [0.2084598998461],
             [0.218229623080869], [0.227977523535188], [0.237702626427135], [0.247403959254523],
             [0.257080551892155], [0.266731436688831], [0.276355648564114], [0.285952225104836], [0.29552020666134],
             [0.305058636443443], [0.314566560616118], [0.324043028394868], [0.333487092140814],
             [0.342897807455451], [0.35227423327509], [0.361615431964962], [0.370920469412983], [0.380188415123161],
             [0.389418342308651], [0.398609327984423], [0.40776045305957], [0.416870802429211], [0.425939465066],
             [0.43496553411123], [0.44394810696552], [0.452886285379068], [0.461779175541483], [0.470625888171158],
             [0.479425538604203], [0.488177246882907], [0.496880137843737], [0.505533341204847],
             [0.514135991653113], [0.522687228930659], [0.531186197920883], [0.539632048733969],
             [0.548023936791874], [0.556361022912784], [0.564642473395035], [0.572867460100481],
             [0.581035160537305], [0.58914475794227], [0.597195441362392], [0.60518640573604], [0.613116851973434],
             [0.62098598703656], [0.628793024018469], [0.636537182221968], [0.644217687237691], [0.651833771021537],
             [0.659384671971473], [0.666869635003698], [0.674287911628145], [0.681638760023334],
             [0.688921445110551], [0.696135238627357], [0.70327941920041], [0.710353272417608], [0.717356090899523],
             [0.724287174370143], [0.731145829726896], [0.737931371109963], [0.744643119970859],
             [0.751280405140293], [0.757842562895277], [0.764328937025505], [0.770738878898969],
             [0.777071747526824], [0.783326909627483], [0.78950373968995], [0.795601620036366], [0.801619940883777],
             [0.807558100405114], [0.813415504789374], [0.819191568300998], [0.82488571333845], [0.83049737049197],
             [0.836025978600521], [0.841470984807897], [0.846831844618015], [0.852108021949363],
             [0.857298989188603], [0.862404227243338], [0.867423225594017], [0.872355482344986],
             [0.877200504274682], [0.881957806884948], [0.886626914449487], [0.891207360061435],
             [0.895698685680048], [0.900100442176505], [0.904412189378826], [0.908633496115883],
             [0.912763940260521], [0.916803108771767], [0.920750597736136], [0.92460601240802], [0.928368967249167],
             [0.932039085967226], [0.935616001553386], [0.939099356319068], [0.942488801931697],
             [0.945783999449539], [0.948984619355586], [0.952090341590516], [0.955100855584692],
             [0.958015860289225], [0.960835064206073], [0.963558185417193], [0.966184951612734],
             [0.968715100118265], [0.971148377921045], [0.973484541695319], [0.975723357826659],
             [0.977864602435316], [0.979908061398614], [0.98185353037236], [0.983700814811277], [0.98544972998846],
             [0.98710010101385], [0.98865176285172], [0.990104560337178], [0.991458348191686], [0.992712991037588],
             [0.993868363411645], [0.994924349777581], [0.99588084453764], [0.996737752043143], [0.997494986604054],
             [0.998152472497548], [0.998710143975583], [0.999167945271476], [0.999525830605479],
             [0.999783764189357], [0.999941720229966], [0.999999682931835]]
    data = NNData(sin_X, sin_Y, .1)
    network.train(data, 10001, order=NNData.Order.RANDOM)
    network.test(data)


def run_XOR():
    network = FFBPNetwork(2, 1)
    network.add_hidden_layer(3)
    features = [[0, 0], [1, 0], [0, 1], [1, 1]]
    labels = [[0], [1], [1], [0]]
    data = NNData(features, labels, 1.0)
    network.train(data, 10001, order=NNData.Order.RANDOM)


def main():
    run_iris()
    run_sin()
    run_XOR()


if __name__ == "__main__":
    main()

"""
"C:/Users/17147/PycharmProjects/CW2/Lab 5/Assignment 5 Final.py"
Sample: [6.7 3.1 5.6 2.4] expected: [0. 0. 1.] produced: [0.7547775458719471, 0.6826575181979915, 0.6796007368675216]
Sample: [6.  2.2 4.  1. ] expected: [0. 1. 0.] produced: [0.7503767658691043, 0.6773801943177686, 0.681635623345441]
Sample: [5.  3.2 1.2 0.2] expected: [1. 0. 0.] produced: [0.7409913191988369, 0.6743659864277234, 0.6742884989085752]
Sample: [7.7 3.  6.1 2.3] expected: [0. 0. 1.] produced: [0.748340775696701, 0.6754689567541903, 0.6723595700927768]
Sample: [6.7 3.  5.2 2.3] expected: [0. 0. 1.] produced: [0.7442557085768053, 0.6705279044733625, 0.6746987612733897]
Sample: [4.6 3.4 1.4 0.3] expected: [1. 0. 0.] produced: [0.7347200716170027, 0.6604930130465528, 0.6743830907997433]
Sample: [5.8 2.7 5.1 1.9] expected: [0. 0. 1.] produced: [0.741554860425244, 0.6607246700455478, 0.6722759405887895]
Sample: [6.2 2.9 4.3 1.3] expected: [0. 1. 0.] produced: [0.7373017608676666, 0.6556052985810279, 0.6745830219640215]
Sample: [6.5 3.  5.5 1.8] expected: [0. 0. 1.] produced: [0.7333916899630912, 0.6584208191795791, 0.6698152424285887]
Sample: [4.4 3.2 1.3 0.2] expected: [1. 0. 0.] produced: [0.722722033311621, 0.6473854826421996, 0.6691128281505146]
Sample: [5.1 3.4 1.5 0.2] expected: [1. 0. 0.] produced: [0.7264394619166484, 0.6446563491956857, 0.6655998906923294]
Sample: [5.5 2.4 3.8 1.1] expected: [0. 1. 0.] produced: [0.7316927118114799, 0.6431375678264803, 0.6624111325306752]
Sample: [7.6 3.  6.6 2.1] expected: [0. 0. 1.] produced: [0.728176408641668, 0.646486169143469, 0.6577442335096947]
Sample: [5.1 3.3 1.7 0.5] expected: [1. 0. 0.] produced: [0.7203226950815984, 0.6384524829947916, 0.6588394573643891]
Sample: [4.4 3.  1.3 0.2] expected: [1. 0. 0.] produced: [0.7188441464584651, 0.6305860484697526, 0.6527246841079611]
Sample: [4.4 2.9 1.4 0.2] expected: [1. 0. 0.] produced: [0.7207746841843902, 0.6260607645410038, 0.6481838068310523]
Sample: [6.9 3.2 5.7 2.3] expected: [0. 0. 1.] produced: [0.7287225563120837, 0.6265201032343801, 0.645710173960531]
Sample: [6.4 2.9 4.3 1.3] expected: [0. 1. 0.] produced: [0.7241752948706226, 0.6211894803730813, 0.6483940123221237]
Sample: [6.9 3.1 4.9 1.5] expected: [0. 1. 0.] produced: [0.719999023185316, 0.6244318068295678, 0.6433955667845459]
Sample: [6.3 2.8 5.1 1.5] expected: [0. 0. 1.] produced: [0.7155806411968737, 0.6275027510104676, 0.6383037204363745]
Sample: [5.6 2.9 3.6 1.3] expected: [0. 1. 0.] produced: [0.7106148343593727, 0.6220170379850486, 0.6410072529836781]
Sample: [5.  3.6 1.4 0.2] expected: [1. 0. 0.] produced: [0.7024046464753286, 0.6217527479774004, 0.6347869716270912]
Sample: [5.1 3.8 1.5 0.3] expected: [1. 0. 0.] produced: [0.7048820067496221, 0.6174603276699425, 0.6300959664071305]
Sample: [5.9 3.2 4.8 1.8] expected: [0. 1. 0.] produced: [0.7103078266026421, 0.6153990959422283, 0.6260827204862707]
Sample: [6.1 3.  4.6 1.4] expected: [0. 1. 0.] produced: [0.7057394718517243, 0.6185702301367906, 0.620927585061604]
Sample: [6.5 2.8 4.6 1.5] expected: [0. 1. 0.] produced: [0.7011968228858029, 0.6217692421567728, 0.6157758539874627]
Sample: [5.4 3.7 1.5 0.2] expected: [1. 0. 0.] produced: [0.693396899396652, 0.6219735284108476, 0.6100926236483113]
Sample: [5.4 3.4 1.7 0.2] expected: [1. 0. 0.] produced: [0.6956854919814965, 0.6172472944113123, 0.6051701417022091]
Sample: [4.8 3.  1.4 0.3] expected: [1. 0. 0.] produced: [0.6958986605893507, 0.6108933446464474, 0.5998956462745008]
Sample: [4.3 3.  1.1 0.1] expected: [1. 0. 0.] produced: [0.6952750199616422, 0.6036329115266575, 0.5947157805602812]
Sample: [7.9 3.8 6.4 2. ] expected: [0. 0. 1.] produced: [0.704670506687836, 0.6049868565710578, 0.5904937743949621]
Sample: [6.4 3.2 5.3 2.3] expected: [0. 0. 1.] produced: [0.6999855526925658, 0.5997577040982909, 0.594068274164598]
Sample: [4.8 3.4 1.6 0.2] expected: [1. 0. 0.] produced: [0.6914263772876903, 0.5914966372504702, 0.5972326926961221]
Sample: [5.7 3.8 1.7 0.3] expected: [1. 0. 0.] produced: [0.6948855981579984, 0.5877743004967827, 0.5924247117928991]
Sample: [6.  2.2 5.  1.5] expected: [0. 0. 1.] produced: [0.6992729818376487, 0.5843615423891834, 0.5874398980372367]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.] produced: [0.6903937124587307, 0.5761487340154877, 0.5908675031349147]
Sample: [6.3 2.9 5.6 1.8] expected: [0. 0. 1.] produced: [0.6967811142290758, 0.5742551956056756, 0.5860297241164818]
Sample: [7.7 2.8 6.7 2. ] expected: [0. 0. 1.] produced: [0.6921820679170851, 0.5691385106982502, 0.5896798562479256]
Sample: [7.3 2.9 6.3 1.8] expected: [0. 0. 1.] produced: [0.6874333393247523, 0.5639922382088852, 0.5932756346737906]
Sample: [6.  2.9 4.5 1.5] expected: [0. 1. 0.] produced: [0.6824279253148929, 0.5587862442033431, 0.5967946549531464]
Sample: [7.2 3.2 6.  1.8] expected: [0. 0. 1.] produced: [0.677830279274663, 0.5628801676340731, 0.5916366257577415]
Sample: [5.  2.  3.5 1. ] expected: [0. 1. 0.] produced: [0.6720843370460654, 0.5573217188030902, 0.5951107764250861]
Sample: [6.1 2.6 5.6 1.4] expected: [0. 0. 1.] produced: [0.6680248275475553, 0.5617355592514721, 0.5900447932954105]
Sample: [5.3 3.7 1.5 0.2] expected: [1. 0. 0.] produced: [0.6602453591757149, 0.554813127199836, 0.59342924053109]
Sample: [6.5 3.2 5.1 2. ] expected: [0. 0. 1.] produced: [0.6655738203184661, 0.5516900522264916, 0.5885551962342417]
Sample: [5.2 4.1 1.5 0.1] expected: [1. 0. 0.] produced: [0.6578499596237205, 0.5448940123717344, 0.592016695918617]
Sample: [6.5 3.  5.2 2. ] expected: [0. 0. 1.] produced: [0.6631483636554578, 0.5417197419865659, 0.5870937390140348]
Sample: [6.3 2.7 4.9 1.8] expected: [0. 0. 1.] produced: [0.658125767823031, 0.5367013392601835, 0.5907187605436898]
Sample: [6.4 3.1 5.5 1.8] expected: [0. 0. 1.] produced: [0.6531871918233166, 0.5317405182311992, 0.5943030564058683]
Sample: [5.9 3.  4.2 1.5] expected: [0. 1. 0.] produced: [0.6479520869378828, 0.5267572161542298, 0.5978009392696451]
Sample: [5.7 3.  4.2 1.2] expected: [0. 1. 0.] produced: [0.642866804750599, 0.5311086571447274, 0.592631458542402]
Sample: [4.8 3.1 1.6 0.2] expected: [1. 0. 0.] produced: [0.6351044461971638, 0.5337663926780267, 0.5873506140675758]
Sample: [6.8 2.8 4.8 1.4] expected: [0. 1. 0.] produced: [0.6407991997353621, 0.530699114609336, 0.5824365361863182]
Sample: [5.4 3.9 1.3 0.4] expected: [1. 0. 0.] produced: [0.6332980177325673, 0.5338454492519195, 0.5773668467603238]
Sample: [7.1 3.  5.9 2.1] expected: [0. 0. 1.] produced: [0.6386761059933416, 0.5302398634304359, 0.5722208429210767]
Sample: [5.5 2.3 4.  1.3] expected: [0. 1. 0.] produced: [0.6332186090098654, 0.5252023515666119, 0.5760679085897253]
Sample: [5.2 3.4 1.4 0.2] expected: [1. 0. 0.] produced: [0.6258339844658947, 0.5282139784943772, 0.5712363945851924]
Sample: [5.4 3.9 1.7 0.4] expected: [1. 0. 0.] produced: [0.6296260453137534, 0.5240914515407272, 0.5661874863998623]
Sample: [5.6 3.  4.5 1.5] expected: [0. 1. 0.] produced: [0.634205280127944, 0.5200714964752478, 0.5608892539485463]
Sample: [6.3 3.4 5.6 2.4] expected: [0. 0. 1.] produced: [0.6292463889496124, 0.5245902501088979, 0.5557643550522591]
Sample: [6.2 3.4 5.4 2.3] expected: [0. 0. 1.] produced: [0.6240871680424095, 0.5196977447286093, 0.5598197056011143]
Sample: [5.5 4.2 1.4 0.2] expected: [1. 0. 0.] produced: [0.6170466359787696, 0.5139452552033119, 0.5642397330301862]
Sample: [7.  3.2 4.7 1.4] expected: [0. 1. 0.] produced: [0.6220160516171049, 0.5101191389632154, 0.5588183441533117]
Sample: [5.1 3.5 1.4 0.2] expected: [1. 0. 0.] produced: [0.6145473249837594, 0.513534963457839, 0.5544724787994147]
Sample: [6.1 2.8 4.7 1.2] expected: [0. 1. 0.] produced: [0.619950182935341, 0.5099941074727238, 0.5488042414535231]
Sample: [6.8 3.  5.5 2.1] expected: [0. 0. 1.] produced: [0.6148874729015941, 0.5146128877212892, 0.5437231776101558]
Sample: [6.3 3.3 6.  2.5] expected: [0. 0. 1.] produced: [0.6097243238144441, 0.5098024772125288, 0.5479219295203211]
Sample: [5.2 2.7 3.9 1.4] expected: [0. 1. 0.] produced: [0.6042480651265987, 0.5049857765552889, 0.5521594447825359]
Sample: [5.2 3.5 1.5 0.2] expected: [1. 0. 0.] produced: [0.5976043151951164, 0.5086882823192396, 0.5478891240267741]
Sample: [5.6 2.8 4.9 2. ] expected: [0. 0. 1.] produced: [0.6026765992309681, 0.5049970270706816, 0.5421299403166768]
Sample: [6.6 2.9 4.6 1.3] expected: [0. 1. 0.] produced: [0.597497293867773, 0.5002554831141554, 0.5463592936845925]
Sample: [6.7 3.  5.  1.7] expected: [0. 1. 0.] produced: [0.5923527603531127, 0.504942631697662, 0.5413116142774316]
Sample: [5.1 3.5 1.4 0.3] expected: [1. 0. 0.] produced: [0.5855705519961278, 0.508641702985221, 0.5374393691647819]
Sample: [6.1 2.8 4.  1.3] expected: [0. 1. 0.] produced: [0.590604580393037, 0.5049064913653962, 0.5315363589053297]
Sample: [5.5 2.6 4.4 1.2] expected: [0. 1. 0.] produced: [0.585454200302429, 0.509500792401705, 0.5266149278434974]
Sample: [7.2 3.  5.8 1.6] expected: [0. 0. 1.] produced: [0.5804271784138997, 0.5141365566569421, 0.5216050757412439]
Sample: [6.4 3.2 4.5 1.5] expected: [0. 1. 0.] produced: [0.5751956142473229, 0.5093176639633334, 0.5261205742791201]
Sample: [6.3 3.3 4.7 1.6] expected: [0. 1. 0.] produced: [0.5700627969362237, 0.5139048105335674, 0.5212144392428794]
Sample: [6.4 2.8 5.6 2.1] expected: [0. 0. 1.] produced: [0.5649642384733076, 0.5184601854736015, 0.5163139058439642]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.] produced: [0.558833978723588, 0.512208193335501, 0.5226055204245402]
Sample: [6.1 3.  4.9 1.8] expected: [0. 0. 1.] produced: [0.5637163404499845, 0.5089533393008737, 0.5161250002078218]
Sample: [5.7 2.6 3.5 1. ] expected: [0. 1. 0.] produced: [0.5584664685054181, 0.5041097176308261, 0.5208574394658183]
Sample: [6.4 2.8 5.6 2.2] expected: [0. 0. 1.] produced: [0.5535603542847072, 0.5088244967815709, 0.515764288696951]
Sample: [5.5 2.4 3.7 1. ] expected: [0. 1. 0.] produced: [0.548373274014303, 0.5039654774176119, 0.5205287488582897]
Sample: [6.7 3.3 5.7 2.5] expected: [0. 0. 1.] produced: [0.5434703858533022, 0.5086912237409517, 0.515436429087389]
Sample: [4.9 3.  1.4 0.2] expected: [1. 0. 0.] produced: [0.5379591252693138, 0.5028499238886739, 0.5218245105967138]
Sample: [5.1 2.5 3.  1.1] expected: [0. 1. 0.] produced: [0.5423932945078066, 0.4992700275310123, 0.5157060932247635]
Sample: [4.7 3.2 1.6 0.2] expected: [1. 0. 0.] produced: [0.537224312250952, 0.5029632324996611, 0.5123842749273931]
Sample: [4.5 2.3 1.3 0.3] expected: [1. 0. 0.] produced: [0.5409630097748938, 0.4982991799144427, 0.5086477940334287]
Sample: [5.5 3.5 1.3 0.2] expected: [1. 0. 0.] produced: [0.5452906936150019, 0.49432684233921953, 0.5032028912401625]
Sample: [5.8 2.7 3.9 1.2] expected: [0. 1. 0.] produced: [0.5498647219548451, 0.49040526065118045, 0.4968360851697122]
Sample: [5.7 2.9 4.2 1.3] expected: [0. 1. 0.] produced: [0.5448578167926766, 0.4951556384532373, 0.4921485815351619]
Sample: [6.9 3.1 5.1 2.3] expected: [0. 0. 1.] produced: [0.5398909952101991, 0.49990396866972103, 0.487387971197722]
Sample: [4.9 2.5 4.5 1.7] expected: [0. 0. 1.] produced: [0.5348582880538194, 0.4952125219288744, 0.4923350460941214]
Sample: [6.5 3.  5.8 2.2] expected: [0. 0. 1.] produced: [0.5299503041325452, 0.4905874243117938, 0.4969108374833295]
Sample: [5.  2.3 3.3 1. ] expected: [0. 1. 0.] produced: [0.5249681568881976, 0.48596310664468845, 0.5021319105949986]
Sample: [7.2 3.6 6.1 2.5] expected: [0. 0. 1.] produced: [0.5201591793245245, 0.49077150442792655, 0.49693242414720235]
Sample: [6.7 3.3 5.7 2.1] expected: [0. 0. 1.] produced: [0.5152987474620865, 0.48617916325147337, 0.5016625522877799]
Sample: [5.8 4.  1.2 0.2] expected: [1. 0. 0.] produced: [0.5107055045992728, 0.4814037888314214, 0.5080149266558641]
Sample: [4.6 3.1 1.5 0.2] expected: [1. 0. 0.] produced: [0.5151918852020407, 0.47684803730067754, 0.5040742523123597]
Sample: [5.  3.3 1.4 0.2] expected: [1. 0. 0.] produced: [0.5194406586595548, 0.47273013690301663, 0.49943248404138774]
Sample: [6.6 3.  4.4 1.4] expected: [0. 1. 0.] produced: [0.5237114731963048, 0.46866280252934367, 0.4926393068010598]
Sample: [5.  3.4 1.5 0.2] expected: [1. 0. 0.] produced: [0.5189428745744327, 0.4733545718214633, 0.49032115247441516]
Sample: [6.7 3.1 4.7 1.5] expected: [0. 1. 0.] produced: [0.5232151464237605, 0.46929188769108515, 0.4835320169776069]
Sample: [6.3 2.5 4.9 1.5] expected: [0. 1. 0.] produced: [0.5183409308518838, 0.4742199107264554, 0.4790311757917495]
Epoch 0 RMSE =  0.5426526375517163
Epoch 100 RMSE =  0.3401154494115147
Epoch 200 RMSE =  0.32314536526184434
Epoch 300 RMSE =  0.3096539560104426
Epoch 400 RMSE =  0.30322655619366623
Epoch 500 RMSE =  0.2987242931963546
Epoch 600 RMSE =  0.2979401742827346
Epoch 700 RMSE =  0.29625646100279884
Epoch 800 RMSE =  0.29366431380208813
Epoch 900 RMSE =  0.2930952731644073
Sample: [4.8 3.1 1.6 0.2] expected: [1. 0. 0.] produced: [0.9166815672590475, 0.27461442110432555, 8.817910671095184e-05]
Sample: [6.4 3.2 4.5 1.5] expected: [0. 1. 0.] produced: [0.02040312423190746, 0.3250003480378854, 0.04552494714897281]
Sample: [7.6 3.  6.6 2.1] expected: [0. 0. 1.] produced: [0.00012364999193514685, 0.37501368466201584, 0.8913402753310538]
Sample: [5.  2.3 3.3 1. ] expected: [0. 1. 0.] produced: [0.054839516612018205, 0.31752508432738413, 0.016791456864174537]
Sample: [5.4 3.9 1.7 0.4] expected: [1. 0. 0.] produced: [0.9185027174091348, 0.2774432400652154, 8.599247974647228e-05]
Sample: [6.9 3.2 5.7 2.3] expected: [0. 0. 1.] produced: [0.00013918361354657842, 0.3737998591240425, 0.8794733463888096]
Sample: [6.5 3.  5.5 1.8] expected: [0. 0. 1.] produced: [0.00022019861183617453, 0.36660197688974033, 0.8217990263375697]
Sample: [6.4 2.8 5.6 2.1] expected: [0. 0. 1.] produced: [0.00012539191785537567, 0.36884331323316205, 0.8907045835168316]
Sample: [6.5 3.  5.2 2. ] expected: [0. 0. 1.] produced: [0.00019126758018453272, 0.36212292113565897, 0.842293470239713]
Sample: [6.1 2.8 4.7 1.2] expected: [0. 1. 0.] produced: [0.004079754186594657, 0.3329006499220989, 0.1979352619676058]
Sample: [6.3 2.9 5.6 1.8] expected: [0. 0. 1.] produced: [0.0001835775089621508, 0.3642213318485861, 0.8475468057330579]
Sample: [6.5 2.8 4.6 1.5] expected: [0. 1. 0.] produced: [0.0022570308144859923, 0.33954923889361477, 0.3091552993335144]
Sample: [4.9 2.5 4.5 1.7] expected: [0. 0. 1.] produced: [0.00026682291394674283, 0.362668605041886, 0.7915054531745456]
Sample: [5.6 3.  4.5 1.5] expected: [0. 1. 0.] produced: [0.0057259638385291295, 0.3332451196935336, 0.14850898887043631]
Sample: [5.  2.  3.5 1. ] expected: [0. 1. 0.] produced: [0.018679634795660146, 0.32698632980239906, 0.04979691610845621]
Sample: [6.7 3.1 5.6 2.4] expected: [0. 0. 1.] produced: [0.0001494681906177641, 0.3739625892387289, 0.8719203492501966]
Sample: [6.1 3.  4.9 1.8] expected: [0. 0. 1.] produced: [0.0007958577917108366, 0.35595860138405466, 0.5596773097390252]
Sample: [5.1 3.4 1.5 0.2] expected: [1. 0. 0.] produced: [0.9181338620630796, 0.2746615417612269, 8.768918430024494e-05]
Sample: [5.7 3.  4.2 1.2] expected: [0. 1. 0.] produced: [0.037889658420662824, 0.31878115451422806, 0.024980263710727903]
Sample: [5.4 3.9 1.3 0.4] expected: [1. 0. 0.] produced: [0.9188788560409662, 0.2764324885026761, 8.679045639019116e-05]
Sample: [5.6 2.8 4.9 2. ] expected: [0. 0. 1.] produced: [0.00014907591526316593, 0.3700292613164335, 0.8742419552902568]
Sample: [5.1 3.3 1.7 0.5] expected: [1. 0. 0.] produced: [0.9148103781565683, 0.2740312321605583, 9.178695304854119e-05]
Sample: [6.1 3.  4.6 1.4] expected: [0. 1. 0.] produced: [0.0034086921562547366, 0.33837144376714606, 0.23035243658438223]
Sample: [5.6 2.9 3.6 1.3] expected: [0. 1. 0.] produced: [0.12886936693725687, 0.3103431886422714, 0.006719709608275726]
Sample: [5.7 2.9 4.2 1.3] expected: [0. 1. 0.] produced: [0.012711270916056712, 0.33462721412919977, 0.07278151866987567]
Sample: [6.  2.9 4.5 1.5] expected: [0. 1. 0.] produced: [0.0019130823824168614, 0.35553164366778506, 0.34709423371645853]
Sample: [6.7 3.3 5.7 2.5] expected: [0. 0. 1.] produced: [0.00014527794843191016, 0.3839380082963936, 0.8753387697701114]
Sample: [6.7 3.1 4.7 1.5] expected: [0. 1. 0.] produced: [0.03041468816438816, 0.3320567698905477, 0.030835673543242754]
Sample: [4.7 3.2 1.6 0.2] expected: [1. 0. 0.] produced: [0.9175403639903532, 0.28576561287576524, 8.762923748175511e-05]
Sample: [6.3 2.7 4.9 1.8] expected: [0. 0. 1.] produced: [0.0003672289545195363, 0.37505884120875244, 0.7349218539018312]
Sample: [6.9 3.1 5.1 2.3] expected: [0. 0. 1.] produced: [0.00020255312949211368, 0.37770235052211915, 0.8354630234784203]
Sample: [6.3 3.3 6.  2.5] expected: [0. 0. 1.] produced: [0.0001175284607761394, 0.3797079350653641, 0.8979917019273826]
Sample: [5.  3.6 1.4 0.2] expected: [1. 0. 0.] produced: [0.9187952277666752, 0.27904142043862684, 8.687278387626556e-05]
Sample: [6.  2.2 4.  1. ] expected: [0. 1. 0.] produced: [0.018065364419416127, 0.3305242824968062, 0.05214593973615505]
Sample: [4.4 3.  1.3 0.2] expected: [1. 0. 0.] produced: [0.9167877584737861, 0.28120329766275776, 8.960101574253858e-05]
Sample: [6.9 3.1 4.9 1.5] expected: [0. 1. 0.] produced: [0.007543754515639131, 0.3407878886403687, 0.11796703287381981]
Sample: [7.2 3.  5.8 1.6] expected: [0. 0. 1.] produced: [0.0004945078889341831, 0.36959864685308047, 0.6751702827961664]
Sample: [5.2 2.7 3.9 1.4] expected: [0. 1. 0.] produced: [0.002508164716509233, 0.352161776781347, 0.2906546613313727]
Sample: [5.1 3.8 1.5 0.3] expected: [1. 0. 0.] produced: [0.9189458998035295, 0.28397598242506067, 8.671716520090583e-05]
Sample: [6.3 3.3 4.7 1.6] expected: [0. 1. 0.] produced: [0.009521931644961746, 0.3430081779901869, 0.09560986129602006]
Sample: [5.8 4.  1.2 0.2] expected: [1. 0. 0.] produced: [0.9194990116681119, 0.2857440498368955, 8.599187762800617e-05]
Sample: [7.2 3.6 6.1 2.5] expected: [0. 0. 1.] produced: [0.00014692807086594405, 0.38449064362055796, 0.875831422509932]
Sample: [5.5 2.4 3.7 1. ] expected: [0. 1. 0.] produced: [0.07141494436550128, 0.3246423710208158, 0.012933509713948138]
Sample: [7.2 3.2 6.  1.8] expected: [0. 0. 1.] produced: [0.00027972713224815737, 0.37947284468296183, 0.7871435329807425]
Sample: [6.1 2.8 4.  1.3] expected: [0. 1. 0.] produced: [0.03258374743862738, 0.3330669740340681, 0.0292667782142339]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.] produced: [0.9177515424143772, 0.2871583793153431, 8.864450840307241e-05]
Sample: [5.1 3.5 1.4 0.2] expected: [1. 0. 0.] produced: [0.918759416224951, 0.2858334459637124, 8.743530192662628e-05]
Sample: [6.  2.2 5.  1.5] expected: [0. 0. 1.] produced: [0.00017886748000001796, 0.3822432522618309, 0.8536364680680728]
Sample: [6.7 3.  5.2 2.3] expected: [0. 0. 1.] produced: [0.0001431484917944812, 0.3812269236357447, 0.8797382141799617]
Sample: [4.6 3.1 1.5 0.2] expected: [1. 0. 0.] produced: [0.9167582354429643, 0.28124803900007195, 9.02967752961211e-05]
Sample: [7.7 3.  6.1 2.3] expected: [0. 0. 1.] produced: [0.00012352641390464612, 0.378135565405056, 0.8947924049035447]
Sample: [5.1 2.5 3.  1.1] expected: [0. 1. 0.] produced: [0.1605575893387472, 0.31091911946215, 0.00527287894785901]
Sample: [6.3 2.5 4.9 1.5] expected: [0. 1. 0.] produced: [0.0002691100134782168, 0.37194958119402327, 0.7952396810321288]
Sample: [7.9 3.8 6.4 2. ] expected: [0. 0. 1.] produced: [0.0004429942165587592, 0.3722010685494968, 0.6979282814414947]
Sample: [5.2 3.5 1.5 0.2] expected: [1. 0. 0.] produced: [0.9184682898806901, 0.2822503986581078, 8.744535631518457e-05]
Sample: [5.1 3.5 1.4 0.3] expected: [1. 0. 0.] produced: [0.9182693124077727, 0.2811278364237943, 8.776191058590184e-05]
Sample: [6.8 2.8 4.8 1.4] expected: [0. 1. 0.] produced: [0.001596227780341951, 0.35512322162647564, 0.3913307187453796]
Sample: [5.2 3.4 1.4 0.2] expected: [1. 0. 0.] produced: [0.9189506128102864, 0.28289926096426865, 8.620302797932612e-05]
Sample: [5.5 4.2 1.4 0.2] expected: [1. 0. 0.] produced: [0.919624880695146, 0.28164448055693847, 8.538083099027892e-05]
Sample: [5.8 2.7 5.1 1.9] expected: [0. 0. 1.] produced: [0.0001897132941169274, 0.3765806272942319, 0.8440211227535196]
Sample: [6.4 3.1 5.5 1.8] expected: [0. 0. 1.] produced: [0.00037506367563032645, 0.3673072269478818, 0.7323292235799019]
Sample: [4.4 3.2 1.3 0.2] expected: [1. 0. 0.] produced: [0.917916343367652, 0.27729466470016984, 8.846302321752565e-05]
Sample: [5.5 3.5 1.3 0.2] expected: [1. 0. 0.] produced: [0.9192968600601573, 0.2759469467804704, 8.654534738649329e-05]
Sample: [6.4 2.8 5.6 2.2] expected: [0. 0. 1.] produced: [0.00012647902876291263, 0.3718709035344929, 0.8916341610659378]
Sample: [5.  3.4 1.5 0.2] expected: [1. 0. 0.] produced: [0.9187417935858792, 0.27319942236112144, 8.740358990987842e-05]
Sample: [6.5 3.  5.8 2.2] expected: [0. 0. 1.] produced: [0.00013111870078007332, 0.3672858869798641, 0.8883000730258422]
Sample: [6.4 3.2 5.3 2.3] expected: [0. 0. 1.] produced: [0.00016270458876410695, 0.36243921073724156, 0.8650882188619645]
Sample: [4.6 3.4 1.4 0.3] expected: [1. 0. 0.] produced: [0.9182251248411987, 0.26887811762372466, 8.842474745361854e-05]
Sample: [5.9 3.2 4.8 1.8] expected: [0. 1. 0.] produced: [0.0007607912956427546, 0.3450246603905967, 0.5769904773026447]
Sample: [5.  3.2 1.2 0.2] expected: [1. 0. 0.] produced: [0.9192578314876488, 0.27064375787948003, 8.600717600038281e-05]
Sample: [5.9 3.  4.2 1.5] expected: [0. 1. 0.] produced: [0.08784385590040486, 0.30724020866283164, 0.010254646598578653]
Sample: [6.4 2.9 4.3 1.3] expected: [0. 1. 0.] produced: [0.22427516790976584, 0.30158329288515867, 0.00341969516328603]
Sample: [4.9 3.  1.4 0.2] expected: [1. 0. 0.] produced: [0.9177556003296454, 0.27555577134407516, 8.74119635923311e-05]
Sample: [6.6 3.  4.4 1.4] expected: [0. 1. 0.] produced: [0.07419971890248052, 0.31434213418451806, 0.012253235235509736]
Sample: [7.3 2.9 6.3 1.8] expected: [0. 0. 1.] produced: [0.00017843138659608926, 0.37195448892147387, 0.8511883628305328]
Sample: [5.  3.3 1.4 0.2] expected: [1. 0. 0.] produced: [0.9185413768064553, 0.27557498602971936, 8.660416479361206e-05]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.] produced: [0.91820175978188, 0.27452791953396866, 8.708534097078921e-05]
Sample: [6.2 2.9 4.3 1.3] expected: [0. 1. 0.] produced: [0.04639688882812731, 0.3174026462811356, 0.020127676228386886]
Sample: [5.5 2.3 4.  1.3] expected: [0. 1. 0.] produced: [0.00199473356309711, 0.34885823340031924, 0.3365120139835315]
Sample: [6.8 3.  5.5 2.1] expected: [0. 0. 1.] produced: [0.0002516606079933372, 0.37216889098607303, 0.8009172302357294]
Sample: [5.7 3.8 1.7 0.3] expected: [1. 0. 0.] produced: [0.9193408100598449, 0.27743779463763596, 8.522508481745324e-05]
Sample: [4.8 3.  1.4 0.3] expected: [1. 0. 0.] produced: [0.9174205924196774, 0.27660129276035017, 8.77585350828681e-05]
Sample: [6.1 2.6 5.6 1.4] expected: [0. 0. 1.] produced: [0.00038873145419222985, 0.36273899279915195, 0.7233145325258438]
Sample: [6.2 3.4 5.4 2.3] expected: [0. 0. 1.] produced: [0.00019399479469604238, 0.3663217978196702, 0.8411890823830976]
Sample: [6.7 3.3 5.7 2.1] expected: [0. 0. 1.] produced: [0.00022235006407292192, 0.3622490368880811, 0.8224561286059822]
Sample: [4.8 3.4 1.6 0.2] expected: [1. 0. 0.] produced: [0.9184724566329844, 0.27035303652223025, 8.738512837300589e-05]
Sample: [6.6 2.9 4.6 1.3] expected: [0. 1. 0.] produced: [0.028787873095731883, 0.31634774569460494, 0.032978449611622565]
Sample: [4.3 3.  1.1 0.1] expected: [1. 0. 0.] produced: [0.9181183147639744, 0.27240265970648175, 8.833875181589478e-05]
Sample: [5.4 3.4 1.7 0.2] expected: [1. 0. 0.] produced: [0.9188129332883437, 0.27109718165956576, 8.696371142056857e-05]
Sample: [5.7 2.6 3.5 1. ] expected: [0. 1. 0.] produced: [0.3319880915709713, 0.2945012272828738, 0.0020052978204732693]
Sample: [6.5 3.2 5.1 2. ] expected: [0. 0. 1.] produced: [0.00021844725517533825, 0.36390475262371175, 0.8246079848669265]
Sample: [5.5 2.4 3.8 1.1] expected: [0. 1. 0.] produced: [0.005527175610469214, 0.33288530917106957, 0.15447520842362356]
Sample: [5.4 3.7 1.5 0.2] expected: [1. 0. 0.] produced: [0.9187024182499102, 0.2741694625775336, 8.655163229446499e-05]
Sample: [6.3 2.8 5.1 1.5] expected: [0. 0. 1.] produced: [0.00035516979743762945, 0.3599547355143391, 0.7428860187857237]
Sample: [5.8 2.7 3.9 1.2] expected: [0. 1. 0.] produced: [0.010510568255285147, 0.32775902982801947, 0.08734862580053605]
Sample: [6.3 3.4 5.6 2.4] expected: [0. 0. 1.] produced: [0.00012117257659589918, 0.3713485090442303, 0.8954689398735517]
Sample: [5.2 4.1 1.5 0.1] expected: [1. 0. 0.] produced: [0.9191402134044256, 0.2726191777724617, 8.6544436124071e-05]
Sample: [5.3 3.7 1.5 0.2] expected: [1. 0. 0.] produced: [0.9186413310274765, 0.2716068825530821, 8.719102241211107e-05]
Sample: [4.4 2.9 1.4 0.2] expected: [1. 0. 0.] produced: [0.9141483349748127, 0.2711075753381701, 9.295457839378722e-05]
Sample: [6.7 3.  5.  1.7] expected: [0. 1. 0.] produced: [0.00034867475601969714, 0.355207876574141, 0.748163619972555]
Sample: [5.5 2.6 4.4 1.2] expected: [0. 1. 0.] produced: [0.0031671902427926877, 0.3402638605685047, 0.24052903835490705]
Sample: [7.  3.2 4.7 1.4] expected: [0. 1. 0.] produced: [0.11823306448807448, 0.3117379423280955, 0.0072908029259879615]
Sample: [4.5 2.3 1.3 0.3] expected: [1. 0. 0.] produced: [0.9074876848673527, 0.27958578660749095, 9.921021850967966e-05]
Sample: [7.7 2.8 6.7 2. ] expected: [0. 0. 1.] produced: [0.00012362472303593892, 0.3768143690731122, 0.8913057246312378]
Sample: [7.1 3.  5.9 2.1] expected: [0. 0. 1.] produced: [0.0001514206921214809, 0.37186175720434955, 0.8701183383533836]
Epoch 1000 RMSE =  0.2927796393185675
Epoch 1100 RMSE =  0.293273991878202
Epoch 1200 RMSE =  0.29182218123145054
Epoch 1300 RMSE =  0.2897628824122109
Epoch 1400 RMSE =  0.2897693997015581
Epoch 1500 RMSE =  0.2905469609836012
Epoch 1600 RMSE =  0.28573669230889015
Epoch 1700 RMSE =  0.2906631983603174
Epoch 1800 RMSE =  0.28806250640086184
Epoch 1900 RMSE =  0.2878517186077597
Sample: [6.3 3.4 5.6 2.4] expected: [0. 0. 1.] produced: [2.0001375714833235e-05, 0.38388805826747513, 0.9011105361205557]
Sample: [4.5 2.3 1.3 0.3] expected: [1. 0. 0.] produced: [0.9310368880743009, 0.261998692869724, 4.546157724228796e-06]
Sample: [6.1 3.  4.6 1.4] expected: [0. 1. 0.] produced: [0.0007994000727611482, 0.34464468889362043, 0.14446906474185853]
Sample: [5.1 3.4 1.5 0.2] expected: [1. 0. 0.] produced: [0.9417047650247976, 0.2623795137537888, 3.721134753833582e-06]
Sample: [7.2 3.6 6.1 2.5] expected: [0. 0. 1.] produced: [2.0539571104362926e-05, 0.3827780350521537, 0.8984956764570045]
Sample: [5.5 2.6 4.4 1.2] expected: [0. 1. 0.] produced: [0.00048327206088239903, 0.3496242566014192, 0.22539116866623]
Sample: [6.  2.2 5.  1.5] expected: [0. 0. 1.] produced: [3.36119221510543e-05, 0.3795841511685287, 0.8380700261203531]
Sample: [5.7 2.6 3.5 1. ] expected: [0. 1. 0.] produced: [0.27262917687102894, 0.2916630388074715, 0.0002175932184369686]
Sample: [5.2 3.5 1.5 0.2] expected: [1. 0. 0.] produced: [0.9414744952442087, 0.26360083082396674, 3.7136233592106707e-06]
Sample: [6.3 3.3 6.  2.5] expected: [0. 0. 1.] produced: [1.6777754591893272e-05, 0.38593769170633935, 0.9162419322228613]
Sample: [7.  3.2 4.7 1.4] expected: [0. 1. 0.] produced: [0.0054050440741402585, 0.32857671448566306, 0.020658926383367547]
Sample: [6.1 3.  4.9 1.8] expected: [0. 0. 1.] produced: [4.222620695798212e-05, 0.37822798059697094, 0.8013814662703653]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.] produced: [0.9404067264519843, 0.2621561646779268, 3.8110782869065506e-06]
Sample: [5.8 2.7 5.1 1.9] expected: [0. 0. 1.] produced: [2.0368235276659956e-05, 0.38106843897212256, 0.8992109812241248]
Sample: [6.4 2.8 5.6 2.2] expected: [0. 0. 1.] produced: [1.721173303758635e-05, 0.37954425660700614, 0.9146646452471792]
Sample: [5.1 2.5 3.  1.1] expected: [0. 1. 0.] produced: [0.04772463745156153, 0.3044532262788691, 0.0019170652655289074]
Sample: [5.1 3.5 1.4 0.3] expected: [1. 0. 0.] produced: [0.9411077217960393, 0.26044684093613824, 3.763171794266551e-06]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.] produced: [0.9403170624053303, 0.2596086322021659, 3.825548219589025e-06]
Sample: [6.8 2.8 4.8 1.4] expected: [0. 1. 0.] produced: [0.00017110275135300392, 0.35661208560312124, 0.47223376206055023]
Sample: [4.6 3.4 1.4 0.3] expected: [1. 0. 0.] produced: [0.9412764339085794, 0.2613710295912336, 3.714314993704375e-06]
Sample: [7.2 3.2 6.  1.8] expected: [0. 0. 1.] produced: [5.743620666615566e-05, 0.3703728458321073, 0.7415286666862321]
Sample: [7.2 3.  5.8 1.6] expected: [0. 0. 1.] produced: [6.592103443983826e-05, 0.3662358954894314, 0.7133858887702496]
Sample: [6.3 3.3 4.7 1.6] expected: [0. 1. 0.] produced: [0.0007184458786299281, 0.34145189259673364, 0.15916090819911022]
Sample: [6.3 2.5 4.9 1.5] expected: [0. 1. 0.] produced: [5.688202513011266e-05, 0.3694243879479843, 0.7458944618514394]
Sample: [6.1 2.6 5.6 1.4] expected: [0. 0. 1.] produced: [5.5663781094237845e-05, 0.374472971850757, 0.746549445190174]
Sample: [5.  3.2 1.2 0.2] expected: [1. 0. 0.] produced: [0.9417026387050591, 0.26102494379699714, 3.68105249577054e-06]
Sample: [5.7 2.9 4.2 1.3] expected: [0. 1. 0.] produced: [0.01360423138170971, 0.3194364669425092, 0.007587748992306827]
Sample: [5.9 3.2 4.8 1.8] expected: [0. 1. 0.] produced: [0.0001936298714794984, 0.3626944041056122, 0.43515914320710203]
Sample: [5.6 2.9 3.6 1.3] expected: [0. 1. 0.] produced: [0.48866546770311026, 0.2893589778514301, 7.724771795133297e-05]
Sample: [7.6 3.  6.6 2.1] expected: [0. 0. 1.] produced: [1.9035565297231605e-05, 0.3936942831389892, 0.9018733281873748]
Sample: [6.7 3.3 5.7 2.1] expected: [0. 0. 1.] produced: [3.1395201363523356e-05, 0.38544517153399616, 0.842721611947907]
Sample: [6.  2.2 4.  1. ] expected: [0. 1. 0.] produced: [0.0052927597551498725, 0.33383360378645305, 0.020490362526407693]
Sample: [6.6 2.9 4.6 1.3] expected: [0. 1. 0.] produced: [0.005410969841946517, 0.3374821336668931, 0.020011313529869373]
Sample: [5.2 3.4 1.4 0.2] expected: [1. 0. 0.] produced: [0.9410223007161906, 0.27080995322934265, 3.6502534506279482e-06]
Sample: [5.2 2.7 3.9 1.4] expected: [0. 1. 0.] produced: [0.0009305929795892896, 0.3568913103460845, 0.1210171047457514]
Sample: [5.4 3.7 1.5 0.2] expected: [1. 0. 0.] produced: [0.9414014061477967, 0.2726091764326569, 3.617824596282503e-06]
Sample: [6.4 3.2 5.3 2.3] expected: [0. 0. 1.] produced: [2.4481567424647838e-05, 0.3959038783263311, 0.8753835094987775]
Sample: [4.9 2.5 4.5 1.7] expected: [0. 0. 1.] produced: [3.054917009622258e-05, 0.3903856806053724, 0.8471199218460364]
Sample: [6.9 3.1 5.1 2.3] expected: [0. 0. 1.] produced: [2.9289652318736893e-05, 0.38759306967275753, 0.8532634792568599]
Sample: [5.  2.  3.5 1. ] expected: [0. 1. 0.] produced: [0.001409832762379434, 0.34743818144415245, 0.0811752381979138]
Sample: [6.6 3.  4.4 1.4] expected: [0. 1. 0.] produced: [0.007972175348336366, 0.3352725018331379, 0.01328111857854399]
Sample: [5.  2.3 3.3 1. ] expected: [0. 1. 0.] produced: [0.017880064453385372, 0.3314692886544843, 0.0055316790686570074]
Sample: [7.9 3.8 6.4 2. ] expected: [0. 0. 1.] produced: [4.7920278314045796e-05, 0.3929233233925053, 0.7738042435740558]
Sample: [5.8 2.7 3.9 1.2] expected: [0. 1. 0.] produced: [0.005231930711405404, 0.3442060423243681, 0.020945582084952886]
Sample: [6.  2.9 4.5 1.5] expected: [0. 1. 0.] produced: [0.00019220259398766416, 0.3803463655299018, 0.433739572232816]
Sample: [5.5 2.4 3.7 1. ] expected: [0. 1. 0.] produced: [0.05055936508216898, 0.33011415568930846, 0.0017257971873533506]
Sample: [5.3 3.7 1.5 0.2] expected: [1. 0. 0.] produced: [0.9414415399149654, 0.28180242902055036, 3.6076375395616806e-06]
Sample: [7.3 2.9 6.3 1.8] expected: [0. 0. 1.] produced: [2.5995188972882943e-05, 0.40818001544053295, 0.8678494693049157]
Sample: [4.8 3.  1.4 0.3] expected: [1. 0. 0.] produced: [0.9395225334944164, 0.27898857844718405, 3.757716025515866e-06]
Sample: [5.  3.6 1.4 0.2] expected: [1. 0. 0.] produced: [0.9413513340260545, 0.2775675972144341, 3.624444892797371e-06]
Sample: [6.9 3.1 4.9 1.5] expected: [0. 1. 0.] produced: [0.0031027159767711213, 0.35436423927772676, 0.03598180048393785]
Sample: [6.4 3.2 4.5 1.5] expected: [0. 1. 0.] produced: [0.010273693974724052, 0.3466535135633608, 0.010046790191055076]
Sample: [5.7 3.  4.2 1.2] expected: [0. 1. 0.] produced: [0.027523503930917723, 0.3407321919585556, 0.003419889028741571]
Sample: [5.  3.3 1.4 0.2] expected: [1. 0. 0.] produced: [0.940824793191505, 0.2855082239396122, 3.663425361405451e-06]
Sample: [6.5 2.8 4.6 1.5] expected: [0. 1. 0.] produced: [0.0005042351969402089, 0.3833529878838745, 0.21066837930663718]
Sample: [7.7 3.  6.1 2.3] expected: [0. 0. 1.] produced: [2.1378717228931534e-05, 0.42098148817326864, 0.8900908974792111]
Sample: [6.3 2.9 5.6 1.8] expected: [0. 0. 1.] produced: [3.272851023697836e-05, 0.4128194885096079, 0.8365675532752157]
Sample: [4.3 3.  1.1 0.1] expected: [1. 0. 0.] produced: [0.9405723148125765, 0.28328618776943226, 3.699493602812412e-06]
Sample: [6.2 2.9 4.3 1.3] expected: [0. 1. 0.] produced: [0.018641942133845187, 0.3439461008058084, 0.005255805631693053]
Sample: [7.1 3.  5.9 2.1] expected: [0. 0. 1.] produced: [2.3200836561169724e-05, 0.41593841199730336, 0.8816770126146091]
Sample: [6.7 3.1 5.6 2.4] expected: [0. 0. 1.] produced: [2.0182251196938085e-05, 0.41378133573119735, 0.896684864606164]
Sample: [5.2 4.1 1.5 0.1] expected: [1. 0. 0.] produced: [0.9417588721608422, 0.2807617762484073, 3.6029503988223846e-06]
Sample: [6.4 3.1 5.5 1.8] expected: [0. 0. 1.] produced: [4.603861669438927e-05, 0.40037415133110693, 0.780922974690442]
Sample: [5.4 3.9 1.7 0.4] expected: [1. 0. 0.] produced: [0.9412607050575018, 0.27776049984564694, 3.6522209789499415e-06]
Sample: [6.5 3.  5.2 2. ] expected: [0. 0. 1.] produced: [3.3604070092259916e-05, 0.3989420541634447, 0.8344057686245007]
Sample: [5.4 3.9 1.3 0.4] expected: [1. 0. 0.] produced: [0.9415537259258171, 0.2746968972788126, 3.642005102091523e-06]
Sample: [5.5 2.4 3.8 1.1] expected: [0. 1. 0.] produced: [0.004374900499059373, 0.34681824017853646, 0.02536181237147809]
Sample: [4.4 2.9 1.4 0.2] expected: [1. 0. 0.] produced: [0.9384965350139893, 0.27708435927022157, 3.877826360401782e-06]
Sample: [6.7 3.  5.  1.7] expected: [0. 1. 0.] produced: [0.00011183937780048722, 0.3854528640661748, 0.5795211463351715]
Sample: [5.1 3.5 1.4 0.2] expected: [1. 0. 0.] produced: [0.9416074875525803, 0.27837375292593813, 3.591819741613001e-06]
Sample: [5.7 3.8 1.7 0.3] expected: [1. 0. 0.] produced: [0.9417479260085528, 0.2772145309394248, 3.5792288365452507e-06]
Sample: [5.6 3.  4.5 1.5] expected: [0. 1. 0.] produced: [0.004713730699471669, 0.35040080294420684, 0.023052586730487646]
Sample: [6.4 2.8 5.6 2.1] expected: [0. 0. 1.] produced: [2.6407825230632123e-05, 0.4067945802878826, 0.8655792680809979]
Sample: [6.5 3.  5.8 2.2] expected: [0. 0. 1.] produced: [2.5045366133749134e-05, 0.4038800123586795, 0.8723578674779332]
Sample: [6.5 3.  5.5 1.8] expected: [0. 0. 1.] produced: [8.81873395381846e-05, 0.3877580317762487, 0.6371972552964607]
Sample: [4.7 3.2 1.6 0.2] expected: [1. 0. 0.] produced: [0.9402728415377745, 0.2735788832672563, 3.7339893247848857e-06]
Sample: [5.1 3.8 1.5 0.3] expected: [1. 0. 0.] produced: [0.9415531198472303, 0.272290231282257, 3.6389954258359208e-06]
Sample: [6.8 3.  5.5 2.1] expected: [0. 0. 1.] produced: [2.8574750458009518e-05, 0.3934107701343264, 0.8574327775404094]
Sample: [5.5 2.3 4.  1.3] expected: [0. 1. 0.] produced: [0.00045766687650259166, 0.3630543847350416, 0.23096160982594507]
Sample: [4.4 3.2 1.3 0.2] expected: [1. 0. 0.] produced: [0.9408407286723938, 0.2724459605740719, 3.6944707969657513e-06]
Sample: [7.7 2.8 6.7 2. ] expected: [0. 0. 1.] produced: [2.0080476515892335e-05, 0.3971896199555472, 0.8977874623830457]
Sample: [6.2 3.4 5.4 2.3] expected: [0. 0. 1.] produced: [3.443856142292381e-05, 0.38846704474691135, 0.8307557451624069]
Sample: [6.7 3.  5.2 2.3] expected: [0. 0. 1.] produced: [2.9732975533762473e-05, 0.3867483275915402, 0.8523690258425102]
Sample: [5.  3.4 1.5 0.2] expected: [1. 0. 0.] produced: [0.9413029834458194, 0.26577971319145977, 3.671679294417959e-06]
Sample: [5.4 3.4 1.7 0.2] expected: [1. 0. 0.] produced: [0.9413200500872666, 0.26475488802113556, 3.6687970546015535e-06]
Sample: [6.4 2.9 4.3 1.3] expected: [0. 1. 0.] produced: [0.043082943574395564, 0.31316958695156044, 0.0021034449288854016]
Sample: [6.1 2.8 4.7 1.2] expected: [0. 1. 0.] produced: [0.00330151168936976, 0.3404024582202267, 0.034212205475354536]
Sample: [4.6 3.1 1.5 0.2] expected: [1. 0. 0.] produced: [0.9401502027181929, 0.2697511934973241, 3.761517458821566e-06]
Sample: [4.9 3.  1.4 0.2] expected: [1. 0. 0.] produced: [0.9404373485901546, 0.2686577504958509, 3.7402122413329695e-06]
Sample: [5.1 3.3 1.7 0.5] expected: [1. 0. 0.] produced: [0.9394096963754902, 0.26774988536931166, 3.810754698267201e-06]
Sample: [6.5 3.2 5.1 2. ] expected: [0. 0. 1.] produced: [7.226834443240613e-05, 0.3774122270200007, 0.6893217654303007]
Sample: [6.1 2.8 4.  1.3] expected: [0. 1. 0.] produced: [0.009738558110863103, 0.328279598997433, 0.010901241650753133]
Sample: [6.7 3.3 5.7 2.5] expected: [0. 0. 1.] produced: [1.8391257137386034e-05, 0.39230506256809783, 0.9077498556832031]
Sample: [5.6 2.8 4.9 2. ] expected: [0. 0. 1.] produced: [2.2948262366191948e-05, 0.3868252093402861, 0.8857661914501282]
Sample: [4.8 3.1 1.6 0.2] expected: [1. 0. 0.] produced: [0.9395290319770857, 0.2642670492955399, 3.84025218108613e-06]
Sample: [6.3 2.8 5.1 1.5] expected: [0. 0. 1.] produced: [6.606492413675746e-05, 0.37223835542810957, 0.7122699033388865]
Sample: [6.9 3.2 5.7 2.3] expected: [0. 0. 1.] produced: [1.869602131518442e-05, 0.38143700965585164, 0.9072419599260526]
Sample: [5.9 3.  4.2 1.5] expected: [0. 1. 0.] produced: [0.0005926027820133313, 0.3460735706722475, 0.18872794466729167]
Sample: [5.8 4.  1.2 0.2] expected: [1. 0. 0.] produced: [0.9420325628051484, 0.2623244606833704, 3.667244724590098e-06]
Sample: [4.4 3.  1.3 0.2] expected: [1. 0. 0.] produced: [0.9394488589309797, 0.2617885418943766, 3.873649381151374e-06]
Sample: [6.7 3.1 4.7 1.5] expected: [0. 1. 0.] produced: [0.0009262616095857232, 0.34392289626899386, 0.12511710131387146]
Sample: [4.8 3.4 1.6 0.2] expected: [1. 0. 0.] produced: [0.9407438738370459, 0.2634446365641187, 3.7634659275805947e-06]
Sample: [6.3 2.7 4.9 1.8] expected: [0. 0. 1.] produced: [3.5105411752189734e-05, 0.378048895675355, 0.8313582868188171]
Sample: [5.5 3.5 1.3 0.2] expected: [1. 0. 0.] produced: [0.9416935826103563, 0.26058821496878076, 3.7012444911806793e-06]
Sample: [5.5 4.2 1.4 0.2] expected: [1. 0. 0.] produced: [0.9420920893713208, 0.2595598456367261, 3.671853232389958e-06]
Epoch 2000 RMSE =  0.287728682107726
Epoch 2100 RMSE =  0.2868717256420876
Epoch 2200 RMSE =  0.28838986924399507
Epoch 2300 RMSE =  0.29011845734252634
Epoch 2400 RMSE =  0.2826106676736104
Epoch 2500 RMSE =  0.2845102443061093
Epoch 2600 RMSE =  0.2870932378298025
Epoch 2700 RMSE =  0.2860187131307386
Epoch 2800 RMSE =  0.2886801761858132
Epoch 2900 RMSE =  0.28647395674708104
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.] produced: [0.9538286385397435, 0.25326590016811146, 3.5546642645305913e-07]
Sample: [6.4 2.9 4.3 1.3] expected: [0. 1. 0.] produced: [0.008588239081229122, 0.32073849211180394, 0.002718981865295116]
Sample: [6.4 2.8 5.6 2.1] expected: [0. 0. 1.] produced: [7.351731330446178e-06, 0.3949781123063117, 0.9029902482258693]
Sample: [6.1 2.8 4.7 1.2] expected: [0. 1. 0.] produced: [0.0005081089677451882, 0.34906459311584687, 0.06651536784751144]
Sample: [6.5 3.  5.8 2.2] expected: [0. 0. 1.] produced: [7.152841407016801e-06, 0.39667524282311184, 0.9057793421686573]
Sample: [7.2 3.6 6.1 2.5] expected: [0. 0. 1.] produced: [7.78913782583504e-06, 0.3924244868692346, 0.8971824239301871]
Sample: [6.  2.2 5.  1.5] expected: [0. 0. 1.] produced: [1.0385824923380794e-05, 0.38619813263343744, 0.8625601764622699]
Sample: [6.6 2.9 4.6 1.3] expected: [0. 1. 0.] produced: [0.001565476288806421, 0.33397104149293744, 0.019201707018862382]
Sample: [6.3 2.8 5.1 1.5] expected: [0. 0. 1.] produced: [2.647146890579654e-05, 0.3781217759750153, 0.6819010013714364]
Sample: [6.1 3.  4.6 1.4] expected: [0. 1. 0.] produced: [0.00013328106094387922, 0.3591851801046854, 0.2519968268286824]
Sample: [4.6 3.4 1.4 0.3] expected: [1. 0. 0.] produced: [0.9537313743861792, 0.25481347733227944, 3.5874087454555803e-07]
Sample: [6.5 3.  5.2 2. ] expected: [0. 0. 1.] produced: [1.1777162680880717e-05, 0.38698238644588023, 0.845264395761334]
Sample: [5.1 2.5 3.  1.1] expected: [0. 1. 0.] produced: [0.08807641829120404, 0.297710568881445, 0.00017182937721070023]
Sample: [5.5 3.5 1.3 0.2] expected: [1. 0. 0.] produced: [0.9545055512743317, 0.25476759206356997, 3.514911741249834e-07]
Sample: [5.9 3.2 4.8 1.8] expected: [0. 1. 0.] produced: [2.9319070214945625e-05, 0.3775021322962218, 0.6571498115652299]
Sample: [6.  2.9 4.5 1.5] expected: [0. 1. 0.] produced: [0.0011290505512421766, 0.3457256484750032, 0.027362579146264953]
Sample: [5.7 3.8 1.7 0.3] expected: [1. 0. 0.] produced: [0.9546763262422375, 0.2594267474820468, 3.4463268719136157e-07]
Sample: [5.5 2.4 3.8 1.1] expected: [0. 1. 0.] produced: [0.01166531124410689, 0.3257951980367645, 0.0018907119147729458]
Sample: [5.8 2.7 5.1 1.9] expected: [0. 0. 1.] produced: [1.127108376198973e-05, 0.3999846168587347, 0.8493721520438526]
Sample: [5.  2.3 3.3 1. ] expected: [0. 1. 0.] produced: [0.0420689739933903, 0.3143758060055462, 0.00041880475277955104]
Sample: [5.  3.6 1.4 0.2] expected: [1. 0. 0.] produced: [0.954562618916502, 0.2623568898295722, 3.466935439691102e-07]
Sample: [5.4 3.9 1.3 0.4] expected: [1. 0. 0.] produced: [0.9547287555836608, 0.2613355974833332, 3.4502916035799285e-07]
Sample: [4.3 3.  1.1 0.1] expected: [1. 0. 0.] produced: [0.9539519375579206, 0.2605919318514543, 3.540710478469259e-07]
Sample: [5.2 2.7 3.9 1.4] expected: [0. 1. 0.] produced: [0.001014889128344928, 0.3511465234324587, 0.03092283610549459]
Sample: [5.2 4.1 1.5 0.1] expected: [1. 0. 0.] produced: [0.954862959159198, 0.2622251250261672, 3.439201709046853e-07]
Sample: [5.5 4.2 1.4 0.2] expected: [1. 0. 0.] produced: [0.9549055201302222, 0.26123482153731054, 3.435298178430875e-07]
Sample: [6.3 3.3 6.  2.5] expected: [0. 0. 1.] produced: [6.825393689767374e-06, 0.4047962986495356, 0.9097218284675881]
Sample: [6.6 3.  4.4 1.4] expected: [0. 1. 0.] produced: [0.011849045954844048, 0.3258297798221151, 0.0018640486106333597]
Sample: [4.5 2.3 1.3 0.3] expected: [1. 0. 0.] produced: [0.9470721675153091, 0.2627467148789697, 4.195032088990214e-07]
Sample: [6.9 3.1 4.9 1.5] expected: [0. 1. 0.] produced: [0.0009865278616566179, 0.3530461413288519, 0.03196100521997391]
Sample: [6.5 3.  5.5 1.8] expected: [0. 0. 1.] produced: [1.4202207436342077e-05, 0.40140889785609735, 0.81284297065596]
Sample: [5.5 2.6 4.4 1.2] expected: [0. 1. 0.] produced: [0.0002124458151332038, 0.36989702243549455, 0.16250642085298375]
Sample: [6.3 3.4 5.6 2.4] expected: [0. 0. 1.] produced: [8.502658343429875e-06, 0.40835287397848885, 0.8869401960678879]
Sample: [6.3 2.5 4.9 1.5] expected: [0. 1. 0.] produced: [2.7821830900547832e-05, 0.3923118069071796, 0.667696443615786]
Sample: [5.  3.4 1.5 0.2] expected: [1. 0. 0.] produced: [0.954584318686207, 0.2650709283414207, 3.4268884251717914e-07]
Sample: [5.1 3.5 1.4 0.3] expected: [1. 0. 0.] produced: [0.9546568877898292, 0.2640515733978347, 3.4205912170304515e-07]
Sample: [6.  2.2 4.  1. ] expected: [0. 1. 0.] produced: [0.03960992987186394, 0.32024882794185483, 0.00044417398578088944]
Sample: [6.2 3.4 5.4 2.3] expected: [0. 0. 1.] produced: [1.9700526144120586e-05, 0.40261074383456225, 0.7453936348893034]
Sample: [4.8 3.1 1.6 0.2] expected: [1. 0. 0.] produced: [0.9537870006238577, 0.26423026650665155, 3.519316810902061e-07]
Sample: [7.3 2.9 6.3 1.8] expected: [0. 0. 1.] produced: [1.2520962671188789e-05, 0.40290548496385853, 0.8323793634964395]
Sample: [4.4 3.2 1.3 0.2] expected: [1. 0. 0.] produced: [0.9540976790364228, 0.2613260992200202, 3.506380559194732e-07]
Sample: [4.4 3.  1.3 0.2] expected: [1. 0. 0.] produced: [0.9536858709649583, 0.26044213347422046, 3.5468319641320014e-07]
Sample: [7.6 3.  6.6 2.1] expected: [0. 0. 1.] produced: [8.196235235195974e-06, 0.401603103499927, 0.8903008013049254]
Sample: [5.1 3.5 1.4 0.2] expected: [1. 0. 0.] produced: [0.9546757956592515, 0.257393582696309, 3.4498656238193334e-07]
Sample: [4.7 3.2 1.6 0.2] expected: [1. 0. 0.] produced: [0.9539038891514563, 0.2566154078215883, 3.524231591341003e-07]
Sample: [5.6 2.9 3.6 1.3] expected: [0. 1. 0.] produced: [0.2023440171093577, 0.29403900698468977, 5.55904288215658e-05]
Sample: [5.  3.2 1.2 0.2] expected: [1. 0. 0.] produced: [0.9542330265042495, 0.2583935132824555, 3.481848021426493e-07]
Sample: [5.8 4.  1.2 0.2] expected: [1. 0. 0.] produced: [0.9548514476869958, 0.25730417047756127, 3.4205060336575395e-07]
Sample: [5.5 2.4 3.7 1. ] expected: [0. 1. 0.] produced: [0.018721519199785507, 0.3193329547288786, 0.0010853350035119342]
Sample: [7.7 2.8 6.7 2. ] expected: [0. 0. 1.] produced: [6.818175253529983e-06, 0.4037569536424785, 0.9091830953044269]
Sample: [5.7 2.9 4.2 1.3] expected: [0. 1. 0.] produced: [0.003224381893237986, 0.3374720085127587, 0.00830257812480397]
Sample: [5.3 3.7 1.5 0.2] expected: [1. 0. 0.] produced: [0.9545606126447498, 0.2602388305303325, 3.451386533821714e-07]
Sample: [6.1 3.  4.9 1.8] expected: [0. 0. 1.] produced: [2.6278586822721368e-05, 0.3894086421509003, 0.6798780819428704]
Sample: [6.7 3.3 5.7 2.1] expected: [0. 0. 1.] produced: [8.760879335480304e-06, 0.3978036784028207, 0.8835714099461173]
Sample: [7.  3.2 4.7 1.4] expected: [0. 1. 0.] produced: [0.002231504137824172, 0.3384740341126988, 0.012770421444713317]
Sample: [6.5 3.2 5.1 2. ] expected: [0. 0. 1.] produced: [1.3119588907629204e-05, 0.39479608539861466, 0.8269082562358384]
Sample: [6.7 3.  5.2 2.3] expected: [0. 0. 1.] produced: [7.521334831472193e-06, 0.3973263001663548, 0.9009278937593949]
Sample: [5.6 2.8 4.9 2. ] expected: [0. 0. 1.] produced: [7.898137597569663e-06, 0.3934470629592991, 0.8959182565259455]
Sample: [7.2 3.2 6.  1.8] expected: [0. 0. 1.] produced: [9.447231403272853e-06, 0.3883056443706303, 0.8752295785939601]
Sample: [6.3 2.9 5.6 1.8] expected: [0. 0. 1.] produced: [7.8840543681528e-06, 0.3869213614354065, 0.896427597397672]
Sample: [5.  2.  3.5 1. ] expected: [0. 1. 0.] produced: [0.00019433878447286976, 0.352157170296393, 0.17831448060257846]
Sample: [6.9 3.1 5.1 2.3] expected: [0. 0. 1.] produced: [9.148471923874475e-06, 0.38701604161154124, 0.8791945022330488]
Sample: [6.3 2.7 4.9 1.8] expected: [0. 0. 1.] produced: [1.1413050301333039e-05, 0.38160007054216366, 0.8496977883478434]
Sample: [6.7 3.1 4.7 1.5] expected: [0. 1. 0.] produced: [0.00023622675280992721, 0.3489608838173646, 0.14780710354313]
Sample: [6.9 3.2 5.7 2.3] expected: [0. 0. 1.] produced: [7.514885607590839e-06, 0.38746402774368566, 0.9014385967755302]
Sample: [6.7 3.1 5.6 2.4] expected: [0. 0. 1.] produced: [6.92257661199246e-06, 0.3850514765352657, 0.9096264109484848]
Sample: [5.1 3.8 1.5 0.3] expected: [1. 0. 0.] produced: [0.9543723991396545, 0.24839426522489252, 3.522881170521494e-07]
Sample: [5.9 3.  4.2 1.5] expected: [0. 1. 0.] produced: [0.00048026564230614914, 0.33981508104706254, 0.07118832545525518]
Sample: [5.7 2.6 3.5 1. ] expected: [0. 1. 0.] produced: [0.06861553018212299, 0.29769229533129005, 0.00023399114153205142]
Sample: [7.2 3.  5.8 1.6] expected: [0. 0. 1.] produced: [1.3186587876753905e-05, 0.38311559464105904, 0.8274489821072298]
Sample: [7.9 3.8 6.4 2. ] expected: [0. 0. 1.] produced: [1.2212836321995422e-05, 0.3807848787304627, 0.8401706957146303]
Sample: [5.4 3.7 1.5 0.2] expected: [1. 0. 0.] produced: [0.9543890379749423, 0.24966161751491386, 3.5341488997866064e-07]
Sample: [6.4 3.1 5.5 1.8] expected: [0. 0. 1.] produced: [9.761127436915455e-06, 0.3788238930056785, 0.8722201680684444]
Sample: [5.6 3.  4.5 1.5] expected: [0. 1. 0.] produced: [5.1561240691006015e-05, 0.35965665251100964, 0.5018323843956969]
Sample: [4.6 3.1 1.5 0.2] expected: [1. 0. 0.] produced: [0.9533738984412645, 0.2500921154942496, 3.5968830138690855e-07]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.] produced: [0.9539462045031374, 0.2491021153724786, 3.541924950087905e-07]
Sample: [4.9 3.  1.4 0.2] expected: [1. 0. 0.] produced: [0.9536178371132156, 0.24829592523033736, 3.5742171025240485e-07]
Sample: [5.4 3.4 1.7 0.2] expected: [1. 0. 0.] produced: [0.9542736713631106, 0.24729419777056136, 3.508968250357744e-07]
Sample: [7.7 3.  6.1 2.3] expected: [0. 0. 1.] produced: [7.96719577695619e-06, 0.3782914899860217, 0.894699134411165]
Sample: [6.7 3.3 5.7 2.5] expected: [0. 0. 1.] produced: [7.959447114064978e-06, 0.37520950500943384, 0.8949428947463121]
Sample: [6.1 2.6 5.6 1.4] expected: [0. 0. 1.] produced: [1.4448481251215693e-05, 0.36640783497093404, 0.8111811623167074]
Sample: [6.7 3.  5.  1.7] expected: [0. 1. 0.] produced: [7.480708497371577e-05, 0.3480462014972972, 0.3940567985966197]
Sample: [4.8 3.4 1.6 0.2] expected: [1. 0. 0.] produced: [0.9544221843922223, 0.24431503710741423, 3.483322062549928e-07]
Sample: [5.5 2.3 4.  1.3] expected: [0. 1. 0.] produced: [0.0016667830037604987, 0.3227061990168751, 0.01772244493633776]
Sample: [5.7 3.  4.2 1.2] expected: [0. 1. 0.] produced: [0.18721890083892706, 0.2829103383770352, 6.231248776617063e-05]
Sample: [5.  3.3 1.4 0.2] expected: [1. 0. 0.] produced: [0.954274545568534, 0.2489367495535538, 3.486267741898799e-07]
Sample: [6.4 3.2 5.3 2.3] expected: [0. 0. 1.] produced: [1.3176008166522272e-05, 0.37542622601883247, 0.8251328430351416]
Sample: [6.5 2.8 4.6 1.5] expected: [0. 1. 0.] produced: [0.0008956325349842371, 0.3322612213972217, 0.03557769546559427]
Sample: [5.4 3.9 1.7 0.4] expected: [1. 0. 0.] produced: [0.9544825438276996, 0.24913718781842648, 3.472320064251269e-07]
Sample: [6.8 3.  5.5 2.1] expected: [0. 0. 1.] produced: [1.262341269254508e-05, 0.37633969097981745, 0.832650024142276]
Sample: [5.2 3.5 1.5 0.2] expected: [1. 0. 0.] produced: [0.9544188004236245, 0.24664263243778503, 3.489007579468687e-07]
Sample: [6.8 2.8 4.8 1.4] expected: [0. 1. 0.] produced: [0.0009668932440731007, 0.33094189229720355, 0.03278003963374785]
Sample: [6.1 2.8 4.  1.3] expected: [0. 1. 0.] produced: [0.023579058736368147, 0.30546507983485777, 0.0008361415343002987]
Sample: [7.1 3.  5.9 2.1] expected: [0. 0. 1.] produced: [8.905597605339734e-06, 0.38452407113631654, 0.88177180714954]
Sample: [5.2 3.4 1.4 0.2] expected: [1. 0. 0.] produced: [0.9543423501758405, 0.24962047303285942, 3.5012163459213173e-07]
Sample: [5.8 2.7 3.9 1.2] expected: [0. 1. 0.] produced: [0.01736548134117298, 0.3083574230594706, 0.0011991389075752931]
Sample: [6.4 3.2 4.5 1.5] expected: [0. 1. 0.] produced: [0.004390365807090711, 0.3245944982555927, 0.005894402845247507]
Sample: [6.3 3.3 4.7 1.6] expected: [0. 1. 0.] produced: [0.0007595185227516971, 0.345073525692057, 0.04290156461374134]
Sample: [4.8 3.  1.4 0.3] expected: [1. 0. 0.] produced: [0.9526718139496042, 0.2574565852465493, 3.6607926925139605e-07]
Sample: [4.4 2.9 1.4 0.2] expected: [1. 0. 0.] produced: [0.952196391421323, 0.25663112403448607, 3.7125250243340645e-07]
Sample: [5.1 3.3 1.7 0.5] expected: [1. 0. 0.] produced: [0.9522441441504981, 0.2556447460360157, 3.697470841090791e-07]
Sample: [5.1 3.4 1.5 0.2] expected: [1. 0. 0.] produced: [0.9541403119119024, 0.2543796324336283, 3.5226685621325716e-07]
Sample: [6.4 2.8 5.6 2.2] expected: [0. 0. 1.] produced: [6.9970257891112785e-06, 0.39150967193938624, 0.9080098433218773]
Sample: [6.2 2.9 4.3 1.3] expected: [0. 1. 0.] produced: [0.003834154056364603, 0.3266041428373694, 0.006898176253485521]
Sample: [4.9 2.5 4.5 1.7] expected: [0. 0. 1.] produced: [1.0879659616742666e-05, 0.38822477080310364, 0.856060223715267]
Epoch 3000 RMSE =  0.2835504484833847
Epoch 3100 RMSE =  0.2852257545094191
Epoch 3200 RMSE =  0.2858680704213706
Epoch 3300 RMSE =  0.2853816413156879
Epoch 3400 RMSE =  0.28577163223567126
Epoch 3500 RMSE =  0.2875737149660106
Epoch 3600 RMSE =  0.28732418426162026
Epoch 3700 RMSE =  0.2872144734349318
Epoch 3800 RMSE =  0.283623576663953
Epoch 3900 RMSE =  0.2851414928710228
Sample: [6.7 3.3 5.7 2.5] expected: [0. 0. 1.] produced: [3.981341123017093e-06, 0.4048997493185264, 0.9027988935537072]
Sample: [4.9 3.  1.4 0.2] expected: [1. 0. 0.] produced: [0.9614218333225608, 0.2415193537431458, 5.032591767808232e-08]
Sample: [6.7 3.3 5.7 2.1] expected: [0. 0. 1.] produced: [5.653338496660406e-06, 0.3963479027817403, 0.858591725252916]
Sample: [6.3 3.3 6.  2.5] expected: [0. 0. 1.] produced: [3.567219073850921e-06, 0.39823781164938266, 0.9142214818416593]
Sample: [6.5 3.  5.8 2.2] expected: [0. 0. 1.] produced: [3.88578065148469e-06, 0.3938668877903695, 0.9057875040925248]
Sample: [7.6 3.  6.6 2.1] expected: [0. 0. 1.] produced: [3.724936477577573e-06, 0.39100161816772716, 0.9101831385567075]
Sample: [7.  3.2 4.7 1.4] expected: [0. 1. 0.] produced: [0.004616773384340463, 0.3126183751124816, 0.0017376942955651467]
Sample: [6.1 3.  4.6 1.4] expected: [0. 1. 0.] produced: [0.0002057677493224112, 0.3485240492106289, 0.07151018958962721]
Sample: [6.3 2.9 5.6 1.8] expected: [0. 0. 1.] produced: [4.975462855692627e-06, 0.39373172785425764, 0.8769833455851119]
Sample: [4.3 3.  1.1 0.1] expected: [1. 0. 0.] produced: [0.9616827357969613, 0.23749569623191066, 5.031414505183353e-08]
Sample: [5.1 3.5 1.4 0.3] expected: [1. 0. 0.] produced: [0.9621596550000787, 0.2365503714800798, 4.9321833826804854e-08]
Sample: [5.5 2.3 4.  1.3] expected: [0. 1. 0.] produced: [4.7766907482402805e-05, 0.36360052792215863, 0.3132055851707889]
Sample: [6.3 3.3 4.7 1.6] expected: [0. 1. 0.] produced: [0.0012506105610094467, 0.3331668666934675, 0.008430395766870716]
Sample: [6.5 3.  5.2 2. ] expected: [0. 0. 1.] produced: [8.400816403279079e-06, 0.3920008052843739, 0.7891523707089011]
Sample: [5.6 3.  4.5 1.5] expected: [0. 1. 0.] produced: [0.00013023627814487954, 0.3585670815473299, 0.1181856196875501]
Sample: [6.3 2.7 4.9 1.8] expected: [0. 0. 1.] produced: [1.0068329118115034e-05, 0.3915251849896974, 0.7508117526244076]
Sample: [6.4 3.2 4.5 1.5] expected: [0. 1. 0.] produced: [0.001522141687998481, 0.3337419337547762, 0.006707446926178502]
Sample: [6.7 3.  5.  1.7] expected: [0. 1. 0.] produced: [2.2614629020642872e-05, 0.3837481089106864, 0.531233368446885]
Sample: [7.2 3.  5.8 1.6] expected: [0. 0. 1.] produced: [7.328673581604502e-05, 0.37523510064135146, 0.2102628698965392]
Sample: [6.4 2.8 5.6 2.1] expected: [0. 0. 1.] produced: [3.5537113808197184e-06, 0.40681871088777155, 0.9149751934562281]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.] produced: [0.9611687033943799, 0.24264639272120764, 5.103856735392491e-08]
Sample: [5.5 3.5 1.3 0.2] expected: [1. 0. 0.] produced: [0.9623461045920378, 0.24153841291585576, 4.905103400125242e-08]
Sample: [5.  3.4 1.5 0.2] expected: [1. 0. 0.] produced: [0.9617032226938165, 0.24088893639639697, 5.014429419166063e-08]
Sample: [4.4 2.9 1.4 0.2] expected: [1. 0. 0.] produced: [0.9592127655856951, 0.24068777262074761, 5.445614841173599e-08]
Sample: [5.8 4.  1.2 0.2] expected: [1. 0. 0.] produced: [0.96276489719426, 0.23902267282664758, 4.837127547487645e-08]
Sample: [4.6 3.4 1.4 0.3] expected: [1. 0. 0.] produced: [0.9612531904943525, 0.2386138185153505, 5.096408154969688e-08]
Sample: [6.6 3.  4.4 1.4] expected: [0. 1. 0.] produced: [0.00021071381089707688, 0.35169650582579237, 0.06984561743062563]
Sample: [5.8 2.7 5.1 1.9] expected: [0. 0. 1.] produced: [3.956361988932631e-06, 0.4005719904573305, 0.9043744044870877]
Sample: [5.1 3.4 1.5 0.2] expected: [1. 0. 0.] produced: [0.9617752324603616, 0.2386298607474122, 5.0071936313153195e-08]
Sample: [4.8 3.1 1.6 0.2] expected: [1. 0. 0.] produced: [0.9601509572015141, 0.23822544595612632, 5.282266546632236e-08]
Sample: [6.6 2.9 4.6 1.3] expected: [0. 1. 0.] produced: [0.00010157263178173645, 0.358943008106641, 0.15443622141802052]
Sample: [6.1 2.6 5.6 1.4] expected: [0. 0. 1.] produced: [4.77381149675691e-06, 0.3978170262348152, 0.882579401286769]
Sample: [5.7 3.8 1.7 0.3] expected: [1. 0. 0.] produced: [0.9623375400993309, 0.23786557316356996, 4.908208552278952e-08]
Sample: [4.8 3.  1.4 0.3] expected: [1. 0. 0.] produced: [0.9600746870754572, 0.23763307072530845, 5.2950474140373655e-08]
Sample: [5.7 3.  4.2 1.2] expected: [0. 1. 0.] produced: [0.0009281582228068181, 0.3345730734357971, 0.012207109811560151]
Sample: [5.  3.3 1.4 0.2] expected: [1. 0. 0.] produced: [0.9617536710365131, 0.23914897757166068, 5.011998005020041e-08]
Sample: [5.7 2.6 3.5 1. ] expected: [0. 1. 0.] produced: [0.016159564417456893, 0.3076115422355856, 0.0003751353751522347]
Sample: [6.  2.9 4.5 1.5] expected: [0. 1. 0.] produced: [2.264245352426028e-05, 0.3825392705625123, 0.5313262505844755]
Sample: [4.9 2.5 4.5 1.7] expected: [0. 0. 1.] produced: [6.810908125239919e-06, 0.4012098569611944, 0.8275454608140381]
Sample: [5.  2.  3.5 1. ] expected: [0. 1. 0.] produced: [0.0005431750855335001, 0.34860908293329096, 0.02287439243167966]
Sample: [5.7 2.9 4.2 1.3] expected: [0. 1. 0.] produced: [0.0015637332596955171, 0.34112992987157126, 0.006416859059128174]
Sample: [5.  3.6 1.4 0.2] expected: [1. 0. 0.] produced: [0.9625060527575796, 0.2472374005441905, 4.834494725018416e-08]
Sample: [6.4 2.8 5.6 2.2] expected: [0. 0. 1.] produced: [3.789565463089745e-06, 0.4129467315594313, 0.9076261902804486]
Sample: [5.5 2.4 3.8 1.1] expected: [0. 1. 0.] produced: [0.0012882962341726398, 0.343231865607532, 0.008121134212301062]
Sample: [5.8 2.7 3.9 1.2] expected: [0. 1. 0.] produced: [0.00362419923974889, 0.3358746544144861, 0.0023147703534852782]
Sample: [4.5 2.3 1.3 0.3] expected: [1. 0. 0.] produced: [0.9538522991527986, 0.252088518828038, 6.323405139075665e-08]
Sample: [7.2 3.6 6.1 2.5] expected: [0. 0. 1.] produced: [3.994562088816311e-06, 0.4168324942886179, 0.9022338120450901]
Sample: [4.4 3.2 1.3 0.2] expected: [1. 0. 0.] produced: [0.9616274264825837, 0.2475114718859563, 4.9994990234986516e-08]
Sample: [6.5 3.  5.5 1.8] expected: [0. 0. 1.] produced: [6.043989765620312e-06, 0.40718801663024407, 0.8481309473691953]
Sample: [5.3 3.7 1.5 0.2] expected: [1. 0. 0.] produced: [0.9625546113058236, 0.24456806523785546, 4.8443589329659305e-08]
Sample: [6.1 2.8 4.7 1.2] expected: [0. 1. 0.] produced: [0.00013839930434718291, 0.36682730874197683, 0.11048142443107412]
Sample: [6.8 2.8 4.8 1.4] expected: [0. 1. 0.] produced: [8.63740152562121e-05, 0.37662482435056976, 0.18039021976352076]
Sample: [7.9 3.8 6.4 2. ] expected: [0. 0. 1.] produced: [1.2655089725369284e-05, 0.4036145442378062, 0.6940227375830588]
Sample: [6.5 3.2 5.1 2. ] expected: [0. 0. 1.] produced: [7.307473452423802e-06, 0.4067519580440343, 0.8170197076809176]
Sample: [6.  2.2 5.  1.5] expected: [0. 0. 1.] produced: [4.679855076517469e-06, 0.40853417977837725, 0.8851803148467022]
Sample: [5.2 4.1 1.5 0.1] expected: [1. 0. 0.] produced: [0.9627891558278425, 0.24366524823241942, 4.838544823371372e-08]
Sample: [7.3 2.9 6.3 1.8] expected: [0. 0. 1.] produced: [3.905830562111998e-06, 0.4060352198578868, 0.9058729875426406]
Sample: [5.2 3.5 1.5 0.2] expected: [1. 0. 0.] produced: [0.9622268333242169, 0.24118381762519212, 4.9372767996592374e-08]
Sample: [5.1 3.5 1.4 0.2] expected: [1. 0. 0.] produced: [0.9622972256273383, 0.24036433529978013, 4.927564378771095e-08]
Sample: [6.7 3.1 5.6 2.4] expected: [0. 0. 1.] produced: [3.6516827374796668e-06, 0.40118674291064704, 0.9127468032740021]
Sample: [6.9 3.1 5.1 2.3] expected: [0. 0. 1.] produced: [4.677458258413146e-06, 0.39491859118042066, 0.8856917390070634]
Sample: [5.2 3.4 1.4 0.2] expected: [1. 0. 0.] produced: [0.9621998889409173, 0.23612774211949863, 4.953334425628547e-08]
Sample: [5.1 3.8 1.5 0.3] expected: [1. 0. 0.] produced: [0.9624017330751489, 0.2353112048901067, 4.9195682632350856e-08]
Sample: [6.7 3.1 4.7 1.5] expected: [0. 1. 0.] produced: [0.00010607453649904823, 0.35509297315450494, 0.1482412825423077]
Sample: [7.1 3.  5.9 2.1] expected: [0. 0. 1.] produced: [3.9726786502836615e-06, 0.3962949914200862, 0.9042725724841121]
Sample: [6.4 3.2 5.3 2.3] expected: [0. 0. 1.] produced: [4.32512400662232e-06, 0.3919674712426029, 0.8950543537071485]
Sample: [6.2 3.4 5.4 2.3] expected: [0. 0. 1.] produced: [4.452010851509689e-06, 0.3883504162902646, 0.8918443759423167]
Sample: [5.9 3.2 4.8 1.8] expected: [0. 1. 0.] produced: [1.3834154788910395e-05, 0.3726301838164818, 0.6751821692879799]
Sample: [5.4 3.4 1.7 0.2] expected: [1. 0. 0.] produced: [0.9624543558777292, 0.2347849409895928, 4.843220482451317e-08]
Sample: [7.2 3.2 6.  1.8] expected: [0. 0. 1.] produced: [1.2956232604364209e-05, 0.37722293259983375, 0.6879813854423494]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.] produced: [0.9619678149482012, 0.23257108093215395, 4.962039829778244e-08]
Sample: [4.7 3.2 1.6 0.2] expected: [1. 0. 0.] produced: [0.9615623298527336, 0.23192622622287992, 5.03042624719033e-08]
Sample: [6.3 2.5 4.9 1.5] expected: [0. 1. 0.] produced: [1.2871854463576863e-05, 0.3724146677525822, 0.6917880556624376]
Sample: [5.5 2.4 3.7 1. ] expected: [0. 1. 0.] produced: [0.08955542581476442, 0.28400892785529824, 4.165856088248518e-05]
Sample: [6.1 3.  4.9 1.8] expected: [0. 0. 1.] produced: [4.1825330928134275e-05, 0.36812735013631936, 0.3437572578225341]
Sample: [6.2 2.9 4.3 1.3] expected: [0. 1. 0.] produced: [0.0002192746375713227, 0.3474603049017066, 0.06646552214482815]
Sample: [6.9 3.1 4.9 1.5] expected: [0. 1. 0.] produced: [3.811731988577406e-05, 0.37096993436382025, 0.3742101017450601]
Sample: [5.1 2.5 3.  1.1] expected: [0. 1. 0.] produced: [0.1173838353393676, 0.2890537338086157, 2.9033055878310406e-05]
Sample: [6.3 3.4 5.6 2.4] expected: [0. 0. 1.] produced: [4.174069365712062e-06, 0.4049296336903618, 0.8968126919312622]
Sample: [6.1 2.8 4.  1.3] expected: [0. 1. 0.] produced: [0.0033376937527184653, 0.3274598448684546, 0.0025478085441560405]
Sample: [5.4 3.9 1.7 0.4] expected: [1. 0. 0.] produced: [0.9623708445464803, 0.24360587550536053, 4.839727925013196e-08]
Sample: [6.4 3.1 5.5 1.8] expected: [0. 0. 1.] produced: [6.576298014185345e-06, 0.39965498227032625, 0.8335489243661539]
Sample: [6.5 2.8 4.6 1.5] expected: [0. 1. 0.] produced: [5.269724736301839e-05, 0.37274154846577107, 0.2855719465894095]
Sample: [6.  2.2 4.  1. ] expected: [0. 1. 0.] produced: [0.0024404239167982542, 0.33493065970567887, 0.003713225881763966]
Sample: [6.3 2.8 5.1 1.5] expected: [0. 0. 1.] produced: [2.372134052634227e-05, 0.3908526567577957, 0.5114920751489175]
Sample: [5.6 2.8 4.9 2. ] expected: [0. 0. 1.] produced: [4.043781448966332e-06, 0.4082484195655126, 0.9016009492947396]
Sample: [6.8 3.  5.5 2.1] expected: [0. 0. 1.] produced: [3.926182950249602e-06, 0.4050540566159109, 0.9048501747278389]
Sample: [5.6 2.9 3.6 1.3] expected: [0. 1. 0.] produced: [0.0023785233284640537, 0.3308485279281555, 0.0038984069269540965]
Sample: [5.5 4.2 1.4 0.2] expected: [1. 0. 0.] produced: [0.9627646518309131, 0.24374728171135798, 4.8264986362849523e-08]
Sample: [5.4 3.7 1.5 0.2] expected: [1. 0. 0.] produced: [0.9623634643203972, 0.24302184937021568, 4.894030021030419e-08]
Sample: [5.4 3.9 1.3 0.4] expected: [1. 0. 0.] produced: [0.9624604747137783, 0.24217825539755336, 4.878972057897336e-08]
Sample: [5.  3.2 1.2 0.2] expected: [1. 0. 0.] produced: [0.9617818965059571, 0.2415446316165228, 4.9976876188654366e-08]
Sample: [5.9 3.  4.2 1.5] expected: [0. 1. 0.] produced: [6.152216337473525e-05, 0.3706185094197302, 0.250958929724946]
Sample: [7.7 3.  6.1 2.3] expected: [0. 0. 1.] produced: [3.6259694333207397e-06, 0.40775104192854206, 0.9124835514395265]
Sample: [5.2 2.7 3.9 1.4] expected: [0. 1. 0.] produced: [8.45401201300775e-05, 0.3683999011459045, 0.18461840879769073]
Sample: [4.6 3.1 1.5 0.2] expected: [1. 0. 0.] produced: [0.9610829019995819, 0.24450202847180952, 5.0825050881045796e-08]
Sample: [4.4 3.  1.3 0.2] expected: [1. 0. 0.] produced: [0.9610595835538532, 0.24369598433054224, 5.094811637811392e-08]
Sample: [6.7 3.  5.2 2.3] expected: [0. 0. 1.] produced: [4.430404022163217e-06, 0.4046392171528234, 0.8907223678475613]
Sample: [6.4 2.9 4.3 1.3] expected: [0. 1. 0.] produced: [0.0016062309021102382, 0.33539628108291153, 0.006237986878860448]
Sample: [5.1 3.3 1.7 0.5] expected: [1. 0. 0.] produced: [0.9600054892474745, 0.24401070313013623, 5.260932818482124e-08]
Sample: [7.7 2.8 6.7 2. ] expected: [0. 0. 1.] produced: [3.4938993701793276e-06, 0.4074809897522045, 0.915966374986764]
Sample: [5.5 2.6 4.4 1.2] expected: [0. 1. 0.] produced: [5.684545408153088e-05, 0.372135048384397, 0.2682850723794263]
Sample: [6.9 3.2 5.7 2.3] expected: [0. 0. 1.] produced: [4.623098891564915e-06, 0.40564402770308156, 0.8851053488078766]
Sample: [4.8 3.4 1.6 0.2] expected: [1. 0. 0.] produced: [0.9620921123175488, 0.2417852687537603, 4.902523885479311e-08]
Sample: [5.  2.3 3.3 1. ] expected: [0. 1. 0.] produced: [0.012892180572317386, 0.31364399749839184, 0.0004895715539113679]
Epoch 4000 RMSE =  0.2896282107411303
Epoch 4100 RMSE =  0.28073533130957945
Epoch 4200 RMSE =  0.2872268093642871
Epoch 4300 RMSE =  0.2882092103023831
Epoch 4400 RMSE =  0.28609631156947335
Epoch 4500 RMSE =  0.28219385159214994
Epoch 4600 RMSE =  0.2865627188260619
Epoch 4700 RMSE =  0.2857371003519097
Epoch 4800 RMSE =  0.2813923982392063
Epoch 4900 RMSE =  0.2861660352564466
Sample: [5.2 3.5 1.5 0.2] expected: [1. 0. 0.] produced: [0.9680773084452977, 0.23500563161636312, 8.613364065103903e-09]
Sample: [6.4 2.9 4.3 1.3] expected: [0. 1. 0.] produced: [0.0019249322222063547, 0.3298089147992316, 0.002060637520893985]
Sample: [5.  2.3 3.3 1. ] expected: [0. 1. 0.] produced: [0.00413584484654875, 0.3252826422447356, 0.0007720757576461817]
Sample: [4.9 3.  1.4 0.2] expected: [1. 0. 0.] produced: [0.9670707576276294, 0.23990677063584678, 8.984914375613316e-09]
Sample: [7.9 3.8 6.4 2. ] expected: [0. 0. 1.] produced: [5.163449006603845e-06, 0.40505905072638393, 0.8038837299235124]
Sample: [5.1 2.5 3.  1.1] expected: [0. 1. 0.] produced: [0.029107934101322078, 0.30428490510243417, 6.152649445888671e-05]
Sample: [6.7 3.3 5.7 2.5] expected: [0. 0. 1.] produced: [2.4520810995886095e-06, 0.41484829614993457, 0.9144986484349323]
Sample: [6.5 3.  5.5 1.8] expected: [0. 0. 1.] produced: [3.3933635724049224e-06, 0.4073302647278872, 0.875914967429004]
Sample: [6.5 3.  5.8 2.2] expected: [0. 0. 1.] produced: [2.4182608536388246e-06, 0.40784258685648894, 0.9161115309623074]
Sample: [6.  2.2 4.  1. ] expected: [0. 1. 0.] produced: [0.00013558822041021215, 0.3580531940680607, 0.05889564710819523]
Sample: [6.6 3.  4.4 1.4] expected: [0. 1. 0.] produced: [0.00024852520526256105, 0.3555641749861891, 0.02795523359920422]
Sample: [6.1 2.6 5.6 1.4] expected: [0. 0. 1.] produced: [3.0395336463115465e-06, 0.41128439502805364, 0.8906981286122746]
Sample: [5.2 4.1 1.5 0.1] expected: [1. 0. 0.] produced: [0.9683950245831795, 0.23776336439657242, 8.553886537537817e-09]
Sample: [6.8 3.  5.5 2.1] expected: [0. 0. 1.] produced: [2.722531381272666e-06, 0.40798694512429573, 0.9038548107790182]
Sample: [5.1 3.5 1.4 0.2] expected: [1. 0. 0.] produced: [0.9679060575125696, 0.23535148180502455, 8.74197079191484e-09]
Sample: [6.1 2.8 4.  1.3] expected: [0. 1. 0.] produced: [0.00037727182908197233, 0.3470990621860307, 0.01660272394078517]
Sample: [6.4 2.8 5.6 2.1] expected: [0. 0. 1.] produced: [2.3966671950967066e-06, 0.40972408025890306, 0.9172538858776367]
Sample: [5.6 3.  4.5 1.5] expected: [0. 1. 0.] produced: [1.4057806348819353e-05, 0.3855044966917033, 0.5344806783796823]
Sample: [7.1 3.  5.9 2.1] expected: [0. 0. 1.] produced: [3.1503087829749964e-06, 0.4079634231672127, 0.8846158837035066]
Sample: [7.3 2.9 6.3 1.8] expected: [0. 0. 1.] produced: [3.194797098002305e-06, 0.40428618460390486, 0.8829482049739502]
Sample: [6.3 2.7 4.9 1.8] expected: [0. 0. 1.] produced: [6.4294792475625085e-06, 0.3926905676220642, 0.7550778996517904]
Sample: [5.4 3.4 1.7 0.2] expected: [1. 0. 0.] produced: [0.9678807742031473, 0.2328870241467865, 8.697694541510442e-09]
Sample: [6.4 3.1 5.5 1.8] expected: [0. 0. 1.] produced: [4.839141909825587e-06, 0.39176773084626965, 0.8171000087819724]
Sample: [6.3 2.8 5.1 1.5] expected: [0. 0. 1.] produced: [9.084321939129423e-06, 0.38137204011750153, 0.6666935512889345]
Sample: [6.6 2.9 4.6 1.3] expected: [0. 1. 0.] produced: [0.00010281410853991102, 0.35148116805281737, 0.08257482486995908]
Sample: [5.  3.3 1.4 0.2] expected: [1. 0. 0.] produced: [0.9676080162029751, 0.2315969241976152, 8.887695723587957e-09]
Sample: [5.8 2.7 3.9 1.2] expected: [0. 1. 0.] produced: [0.0005650638356603333, 0.3363408307083887, 0.010008686596223459]
Sample: [5.5 3.5 1.3 0.2] expected: [1. 0. 0.] produced: [0.9681432294502608, 0.23335047455972568, 8.689244178534133e-09]
Sample: [6.1 2.8 4.7 1.2] expected: [0. 1. 0.] produced: [3.114012126539e-05, 0.37162819715374645, 0.2938720591390437]
Sample: [5.  3.6 1.4 0.2] expected: [1. 0. 0.] produced: [0.9681918149135108, 0.23522708877486279, 8.623481574343916e-09]
Sample: [7.7 3.  6.1 2.3] expected: [0. 0. 1.] produced: [2.6069576539962384e-06, 0.4040817961086995, 0.9085045044926208]
Sample: [5.3 3.7 1.5 0.2] expected: [1. 0. 0.] produced: [0.9682792473083974, 0.23269475320020414, 8.595845394490803e-09]
Sample: [4.3 3.  1.1 0.1] expected: [1. 0. 0.] produced: [0.9676135968458627, 0.23219523421173519, 8.875237837021535e-09]
Sample: [5.1 3.5 1.4 0.3] expected: [1. 0. 0.] produced: [0.9680193890393705, 0.2312964923608074, 8.694631354701132e-09]
Sample: [6.2 2.9 4.3 1.3] expected: [0. 1. 0.] produced: [0.0011441999722044233, 0.3294156496192679, 0.004046604581418308]
Sample: [6.7 3.3 5.7 2.1] expected: [0. 0. 1.] produced: [3.36692882228483e-06, 0.39926863405794877, 0.8775045666641886]
Sample: [5.4 3.9 1.3 0.4] expected: [1. 0. 0.] produced: [0.9683416580337844, 0.23140159736894772, 8.587472964647854e-09]
Sample: [5.4 3.9 1.7 0.4] expected: [1. 0. 0.] produced: [0.9681048886279431, 0.23073524688138092, 8.672046292404046e-09]
Sample: [5.1 3.3 1.7 0.5] expected: [1. 0. 0.] produced: [0.9661837211386991, 0.23053577540103157, 9.376543902861839e-09]
Sample: [5.1 3.4 1.5 0.2] expected: [1. 0. 0.] produced: [0.9679149986728893, 0.22934785144867506, 8.74819434877464e-09]
Sample: [7.2 3.6 6.1 2.5] expected: [0. 0. 1.] produced: [2.6891280694138593e-06, 0.39458869733483043, 0.9054840459237103]
Sample: [5.2 3.4 1.4 0.2] expected: [1. 0. 0.] produced: [0.968055962479361, 0.226935038224219, 8.705250155924577e-09]
Sample: [4.4 3.  1.3 0.2] expected: [1. 0. 0.] produced: [0.9669106216215876, 0.22657973144756005, 9.146463646585337e-09]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.] produced: [0.9675262918726756, 0.22569603809495326, 8.904695193236132e-09]
Sample: [5.  3.2 1.2 0.2] expected: [1. 0. 0.] produced: [0.9679453798401246, 0.2248971455813729, 8.755033202728641e-09]
Sample: [6.8 2.8 4.8 1.4] expected: [0. 1. 0.] produced: [5.331015492293969e-05, 0.35387431859275104, 0.17240058429556324]
Sample: [6.1 3.  4.9 1.8] expected: [0. 0. 1.] produced: [8.986924574569544e-06, 0.37850197902557536, 0.6706034240655333]
Sample: [6.3 2.9 5.6 1.8] expected: [0. 0. 1.] produced: [2.9886208333741617e-06, 0.3881079610086851, 0.8940819642405187]
Sample: [5.5 2.4 3.7 1. ] expected: [0. 1. 0.] produced: [0.0012384219303323049, 0.31889004707288154, 0.0036991808105086062]
Sample: [6.  2.9 4.5 1.5] expected: [0. 1. 0.] produced: [2.3496113018971877e-05, 0.3659973384016679, 0.3752281570042061]
Sample: [4.8 3.4 1.6 0.2] expected: [1. 0. 0.] produced: [0.967909958279939, 0.2288047626321298, 8.739408491822073e-09]
Sample: [5.5 4.2 1.4 0.2] expected: [1. 0. 0.] produced: [0.9685751239602086, 0.2278968618925685, 8.493654936712735e-09]
Sample: [5.7 2.6 3.5 1. ] expected: [0. 1. 0.] produced: [0.203274268241507, 0.2708161932086645, 3.969345391431099e-06]
Sample: [4.4 3.2 1.3 0.2] expected: [1. 0. 0.] produced: [0.9673884052464341, 0.23006930464849398, 8.909407358784142e-09]
Sample: [5.  3.4 1.5 0.2] expected: [1. 0. 0.] produced: [0.9678126529266493, 0.22920627545429648, 8.739339388690193e-09]
Sample: [4.6 3.4 1.4 0.3] expected: [1. 0. 0.] produced: [0.9675114138815849, 0.2285864066603152, 8.856462795914322e-09]
Sample: [5.6 2.8 4.9 2. ] expected: [0. 0. 1.] produced: [3.2330164246339454e-06, 0.3925645243438038, 0.8825891451701012]
Sample: [7.7 2.8 6.7 2. ] expected: [0. 0. 1.] produced: [2.3823074841394967e-06, 0.39280857620865095, 0.9176227190199769]
Sample: [6.3 3.3 6.  2.5] expected: [0. 0. 1.] produced: [2.3649906803321202e-06, 0.389558308823235, 0.9183970179633135]
Sample: [6.3 2.5 4.9 1.5] expected: [0. 1. 0.] produced: [6.559909801737299e-06, 0.3747171312454832, 0.7527835341214048]
Sample: [6.9 3.2 5.7 2.3] expected: [0. 0. 1.] produced: [3.7261977397048817e-06, 0.3861322140246002, 0.8603752825392115]
Sample: [5.1 3.8 1.5 0.3] expected: [1. 0. 0.] produced: [0.9682791941012513, 0.22371759168505093, 8.486691288995622e-09]
Sample: [6.9 3.1 5.1 2.3] expected: [0. 0. 1.] produced: [6.033921216311938e-06, 0.3765674137248758, 0.7690757713695813]
Sample: [5.6 2.9 3.6 1.3] expected: [0. 1. 0.] produced: [0.05677251108642671, 0.27721961310553755, 2.502271550194822e-05]
Sample: [6.7 3.1 5.6 2.4] expected: [0. 0. 1.] produced: [2.782606348617418e-06, 0.3863693380164299, 0.9003260610114748]
Sample: [4.7 3.2 1.6 0.2] expected: [1. 0. 0.] produced: [0.9672775980947435, 0.22270991861949563, 8.893602712106502e-09]
Sample: [6.4 3.2 4.5 1.5] expected: [0. 1. 0.] produced: [0.0016901792370727345, 0.3128762161006676, 0.0024315105078118967]
Sample: [4.4 2.9 1.4 0.2] expected: [1. 0. 0.] produced: [0.9664150654615629, 0.22485680117100856, 9.223720600318316e-09]
Sample: [4.8 3.1 1.6 0.2] expected: [1. 0. 0.] produced: [0.9670211493801508, 0.2239935708288037, 8.987711119507051e-09]
Sample: [5.9 3.  4.2 1.5] expected: [0. 1. 0.] produced: [0.00047965876904373483, 0.32816802417447, 0.012126163786117392]
Sample: [6.9 3.1 4.9 1.5] expected: [0. 1. 0.] produced: [0.0001540927431867592, 0.3444896983382146, 0.050020126965749444]
Sample: [7.2 3.2 6.  1.8] expected: [0. 0. 1.] produced: [4.321512745764135e-06, 0.3892892622173191, 0.8372433438670113]
Sample: [6.3 3.3 4.7 1.6] expected: [0. 1. 0.] produced: [0.0001246470990874785, 0.3482493236809312, 0.06480491188849032]
Sample: [4.5 2.3 1.3 0.3] expected: [1. 0. 0.] produced: [0.9606098194985297, 0.2311149885990321, 1.1440101728693542e-08]
Sample: [5.8 2.7 5.1 1.9] expected: [0. 0. 1.] produced: [3.037358499001775e-06, 0.39398687920043424, 0.8902076291271972]
Sample: [6.4 2.8 5.6 2.2] expected: [0. 0. 1.] produced: [2.4778155287662076e-06, 0.3930223648861353, 0.9133701798875888]
Sample: [6.5 2.8 4.6 1.5] expected: [0. 1. 0.] produced: [3.899822041614865e-05, 0.3585712093997725, 0.23553252893720267]
Sample: [6.3 3.4 5.6 2.4] expected: [0. 0. 1.] produced: [3.0923024494359563e-06, 0.3920754200761219, 0.8876060560306313]
Sample: [5.5 2.3 4.  1.3] expected: [0. 1. 0.] produced: [6.735393985564992e-05, 0.3540409107953805, 0.13216181166805438]
Sample: [4.8 3.  1.4 0.3] expected: [1. 0. 0.] produced: [0.9671027358263238, 0.22898520658784297, 8.953158294911014e-09]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.] produced: [0.9677545542083466, 0.22809081600318334, 8.716374065307486e-09]
Sample: [5.7 3.  4.2 1.2] expected: [0. 1. 0.] produced: [0.012686807093835389, 0.30034902873448716, 0.00018117147445623373]
Sample: [4.6 3.1 1.5 0.2] expected: [1. 0. 0.] produced: [0.9672162468791623, 0.23013571928677962, 8.9163407560095e-09]
Sample: [6.4 3.2 5.3 2.3] expected: [0. 0. 1.] produced: [3.6256038569134205e-06, 0.3932347758178446, 0.8655840173537428]
Sample: [6.7 3.1 4.7 1.5] expected: [0. 1. 0.] produced: [0.0006277556477298337, 0.3322228535001521, 0.008627566927237875]
Sample: [6.1 3.  4.6 1.4] expected: [0. 1. 0.] produced: [0.00023211806184462145, 0.34717930132455005, 0.03023496784179268]
Sample: [5.8 4.  1.2 0.2] expected: [1. 0. 0.] produced: [0.9684954488249433, 0.23262821215604126, 8.456474205633701e-09]
Sample: [7.6 3.  6.6 2.1] expected: [0. 0. 1.] produced: [2.498445356161444e-06, 0.4027824723448201, 0.9123078178860066]
Sample: [5.7 2.9 4.2 1.3] expected: [0. 1. 0.] produced: [0.0007746404543080071, 0.33422024498210584, 0.006607594928047347]
Sample: [4.9 2.5 4.5 1.7] expected: [0. 0. 1.] produced: [3.872231160675841e-06, 0.3988248290152944, 0.8558834828234124]
Sample: [7.  3.2 4.7 1.4] expected: [0. 1. 0.] produced: [0.0019974459353912836, 0.3251412618964038, 0.0019720752741487196]
Sample: [5.9 3.2 4.8 1.8] expected: [0. 1. 0.] produced: [1.0401757865993661e-05, 0.388403593650659, 0.6264584710857933]
Sample: [5.2 2.7 3.9 1.4] expected: [0. 1. 0.] produced: [0.0012049195860226595, 0.33863244562512873, 0.0036981039442665338]
Sample: [6.7 3.  5.  1.7] expected: [0. 1. 0.] produced: [7.250835326457663e-05, 0.374843111350232, 0.11998877794724547]
Sample: [6.5 3.2 5.1 2. ] expected: [0. 0. 1.] produced: [2.108195521746805e-05, 0.39406887363727855, 0.398714697641227]
Sample: [6.5 3.  5.2 2. ] expected: [0. 0. 1.] produced: [3.1101566374134135e-06, 0.41399818714952447, 0.8872150655585086]
Sample: [5.  2.  3.5 1. ] expected: [0. 1. 0.] produced: [5.4828666863631926e-05, 0.3763888365391135, 0.16595126322474216]
Sample: [6.7 3.  5.2 2.3] expected: [0. 0. 1.] produced: [2.7368719287751874e-06, 0.41689869426397985, 0.9024718826094706]
Sample: [5.7 3.8 1.7 0.3] expected: [1. 0. 0.] produced: [0.9680441839351032, 0.23911420978746092, 8.62313087243121e-09]
Sample: [5.4 3.7 1.5 0.2] expected: [1. 0. 0.] produced: [0.968141325565901, 0.23829958458024386, 8.591935027938691e-09]
Sample: [5.5 2.4 3.8 1.1] expected: [0. 1. 0.] produced: [0.0001912771357127768, 0.361110406111437, 0.038470779759349864]
Sample: [6.  2.2 5.  1.5] expected: [0. 0. 1.] produced: [2.9833810988184133e-06, 0.4149508785654773, 0.8924273501012877]
Sample: [7.2 3.  5.8 1.6] expected: [0. 0. 1.] produced: [3.933673768775867e-06, 0.4080047494707182, 0.853553979496649]
Sample: [6.2 3.4 5.4 2.3] expected: [0. 0. 1.] produced: [2.733168035202612e-06, 0.40887154669410625, 0.9031129131519087]
Sample: [5.5 2.6 4.4 1.2] expected: [0. 1. 0.] produced: [1.8551895816593826e-05, 0.3828356295685057, 0.44472649903430955]
Epoch 5000 RMSE =  0.28355302686581324
Epoch 5100 RMSE =  0.2826084072523002
Epoch 5200 RMSE =  0.2837090305044548
Epoch 5300 RMSE =  0.28507568321814075
Epoch 5400 RMSE =  0.2871698209172413
Epoch 5500 RMSE =  0.28388510004735806
Epoch 5600 RMSE =  0.2831149541998062
Epoch 5700 RMSE =  0.2814361508403271
Epoch 5800 RMSE =  0.27959332752723987
Epoch 5900 RMSE =  0.28160274508594657
Sample: [5.  2.3 3.3 1. ] expected: [0. 1. 0.] produced: [0.000938494654650397, 0.33495095165756333, 0.002694909569782242]
Sample: [6.3 2.9 5.6 1.8] expected: [0. 0. 1.] produced: [2.142056209371045e-06, 0.4132485694337212, 0.9075229136300755]
Sample: [6.  2.2 5.  1.5] expected: [0. 0. 1.] produced: [2.249719675040439e-06, 0.4090286485845936, 0.9019287943114158]
Sample: [5.6 3.  4.5 1.5] expected: [0. 1. 0.] produced: [1.2285813298155371e-05, 0.38454613155286865, 0.48310141196491896]
Sample: [6.5 3.2 5.1 2. ] expected: [0. 0. 1.] produced: [7.891458547105298e-06, 0.39483278768484403, 0.6252753459896666]
Sample: [6.  2.9 4.5 1.5] expected: [0. 1. 0.] produced: [2.310448589163742e-05, 0.37848825906159855, 0.2843111692726766]
Sample: [6.7 3.3 5.7 2.1] expected: [0. 0. 1.] produced: [3.2676493335766566e-06, 0.4075312530895227, 0.8461084812165904]
Sample: [6.1 2.8 4.  1.3] expected: [0. 1. 0.] produced: [0.004274285681616885, 0.3184639377353397, 0.00034624310455796015]
Sample: [6.9 3.1 4.9 1.5] expected: [0. 1. 0.] produced: [0.0001427954044615409, 0.36199907079602056, 0.032843974873953484]
Sample: [5.6 2.8 4.9 2. ] expected: [0. 0. 1.] produced: [2.626886770978757e-06, 0.4161544986358366, 0.8809346202507388]
Sample: [5.4 3.7 1.5 0.2] expected: [1. 0. 0.] produced: [0.9725543120474269, 0.22988453583369817, 1.8311547774069544e-09]
Sample: [6.7 3.  5.2 2.3] expected: [0. 0. 1.] produced: [2.4565573682021557e-06, 0.41239931226097815, 0.8902760733616882]
Sample: [6.4 2.8 5.6 2.1] expected: [0. 0. 1.] produced: [2.033858925449964e-06, 0.411192880734821, 0.9129018202190151]
Sample: [5.  3.6 1.4 0.2] expected: [1. 0. 0.] produced: [0.9724530241826953, 0.22569938988821708, 1.8448598715927183e-09]
Sample: [5.1 3.8 1.5 0.3] expected: [1. 0. 0.] produced: [0.9724748775822113, 0.2250006023337699, 1.8424364750052247e-09]
Sample: [4.7 3.2 1.6 0.2] expected: [1. 0. 0.] produced: [0.9714550732534977, 0.22466374989083615, 1.9396109328585008e-09]
Sample: [5.8 2.7 5.1 1.9] expected: [0. 0. 1.] produced: [2.4540161370697008e-06, 0.402439791912959, 0.8906976493113687]
Sample: [6.1 2.6 5.6 1.4] expected: [0. 0. 1.] produced: [2.9384126897143815e-06, 0.3967937525183264, 0.864911636630888]
Sample: [5.2 3.4 1.4 0.2] expected: [1. 0. 0.] produced: [0.9723223078816633, 0.22037770208927196, 1.862279051504695e-09]
Sample: [5.9 3.2 4.8 1.8] expected: [0. 1. 0.] produced: [9.921381219551793e-06, 0.3778717810743094, 0.554611985395437]
Sample: [7.1 3.  5.9 2.1] expected: [0. 0. 1.] produced: [3.712840568362308e-06, 0.3946927796400015, 0.8211747938291983]
Sample: [7.7 3.  6.1 2.3] expected: [0. 0. 1.] produced: [2.765293532717726e-06, 0.3950143534916551, 0.8727095116029577]
Sample: [5.  3.3 1.4 0.2] expected: [1. 0. 0.] produced: [0.9724581971631894, 0.2189611597545892, 1.8326160407814652e-09]
Sample: [6.1 3.  4.9 1.8] expected: [0. 0. 1.] produced: [2.5080369107879532e-05, 0.3644784706383565, 0.2604689069159222]
Sample: [4.8 3.4 1.6 0.2] expected: [1. 0. 0.] produced: [0.9714158396697132, 0.217213128487859, 1.957349277686581e-09]
Sample: [6.  2.2 4.  1. ] expected: [0. 1. 0.] produced: [4.1183383947060054e-05, 0.3552153936669851, 0.15544379760668425]
Sample: [6.5 3.  5.2 2. ] expected: [0. 0. 1.] produced: [2.444470970899351e-06, 0.3934995183064433, 0.8919501699418662]
Sample: [6.6 2.9 4.6 1.3] expected: [0. 1. 0.] produced: [5.516325870302962e-05, 0.3533130276383706, 0.11029530892712079]
Sample: [4.6 3.4 1.4 0.3] expected: [1. 0. 0.] produced: [0.9715763848598754, 0.21998580061358905, 1.939480503580223e-09]
Sample: [5.2 2.7 3.9 1.4] expected: [0. 1. 0.] produced: [3.24473078598257e-05, 0.36313928126426787, 0.2020163396878889]
Sample: [4.3 3.  1.1 0.1] expected: [1. 0. 0.] produced: [0.9717899783253593, 0.22183957391927359, 1.9187456314958804e-09]
Sample: [6.3 2.7 4.9 1.8] expected: [0. 0. 1.] produced: [3.3186501158285724e-06, 0.3944469185147649, 0.8448157839636704]
Sample: [5.7 3.8 1.7 0.3] expected: [1. 0. 0.] produced: [0.9724300134434911, 0.2193007945642385, 1.8544016738704678e-09]
Sample: [5.  3.4 1.5 0.2] expected: [1. 0. 0.] produced: [0.9720584714081402, 0.2187888774406161, 1.890793251041761e-09]
Sample: [7.9 3.8 6.4 2. ] expected: [0. 0. 1.] produced: [3.5849835428904394e-06, 0.388444376341625, 0.8311270288273207]
Sample: [6.5 2.8 4.6 1.5] expected: [0. 1. 0.] produced: [1.6484950862318367e-05, 0.3670456882785784, 0.38719808728002486]
Sample: [7.2 3.  5.8 1.6] expected: [0. 0. 1.] produced: [8.96930319060391e-06, 0.3790653424351666, 0.5862290727872257]
Sample: [6.3 3.3 6.  2.5] expected: [0. 0. 1.] produced: [1.7913485919043719e-06, 0.39558829109663834, 0.9264574804848278]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.] produced: [0.9716254351446785, 0.2160972852457391, 1.939769166214144e-09]
Sample: [5.6 2.9 3.6 1.3] expected: [0. 1. 0.] produced: [0.004193581971000606, 0.30247906325115237, 0.0003605397873768976]
Sample: [6.5 3.  5.8 2.2] expected: [0. 0. 1.] produced: [1.86226833110693e-06, 0.3953147087945316, 0.9228723062723634]
Sample: [5.3 3.7 1.5 0.2] expected: [1. 0. 0.] produced: [0.9724424294567767, 0.21610131939972152, 1.8623550120102208e-09]
Sample: [5.1 3.4 1.5 0.2] expected: [1. 0. 0.] produced: [0.9720174647939828, 0.21562134063047836, 1.903105475869699e-09]
Sample: [5.2 4.1 1.5 0.1] expected: [1. 0. 0.] produced: [0.9727164516153497, 0.21477884297142083, 1.8369965384839966e-09]
Sample: [6.8 2.8 4.8 1.4] expected: [0. 1. 0.] produced: [1.6599495986526537e-05, 0.36332656640795624, 0.38574935586431064]
Sample: [6.3 2.8 5.1 1.5] expected: [0. 0. 1.] produced: [1.4178541203193899e-05, 0.3698926264382563, 0.4340529837184143]
Sample: [6.4 2.9 4.3 1.3] expected: [0. 1. 0.] produced: [6.974385175913325e-05, 0.34851122318975314, 0.08366779025240052]
Sample: [6.3 3.4 5.6 2.4] expected: [0. 0. 1.] produced: [1.8700087553227263e-06, 0.39626053040683545, 0.9229798204680164]
Sample: [7.  3.2 4.7 1.4] expected: [0. 1. 0.] produced: [7.000838552059945e-05, 0.3498608955503399, 0.08327061522386234]
Sample: [6.3 2.5 4.9 1.5] expected: [0. 1. 0.] produced: [2.834906722523217e-06, 0.3926845392361135, 0.8724366822269457]
Sample: [5.4 3.9 1.3 0.4] expected: [1. 0. 0.] produced: [0.9725187575303195, 0.2211638683368872, 1.8451464413663862e-09]
Sample: [5.5 2.3 4.  1.3] expected: [0. 1. 0.] produced: [1.0021221827960066e-05, 0.38135573136267387, 0.5515577473151034]
Sample: [5.5 3.5 1.3 0.2] expected: [1. 0. 0.] produced: [0.9726466588986108, 0.2229867812991017, 1.8087868493794823e-09]
Sample: [6.7 3.3 5.7 2.5] expected: [0. 0. 1.] produced: [2.1836513740591085e-06, 0.40421649556720013, 0.9038445047490888]
Sample: [6.7 3.  5.  1.7] expected: [0. 1. 0.] produced: [1.8983804470235372e-05, 0.3741798798632999, 0.3381234842958625]
Sample: [5.1 3.5 1.4 0.3] expected: [1. 0. 0.] produced: [0.9726144289527522, 0.2231851457145781, 1.800285901498848e-09]
Sample: [6.4 2.8 5.6 2.2] expected: [0. 0. 1.] produced: [2.6275183546154255e-06, 0.40249956566737977, 0.8788953856315811]
Sample: [7.2 3.2 6.  1.8] expected: [0. 0. 1.] produced: [1.3183218603613424e-05, 0.3791941542430423, 0.45292282938983736]
Sample: [5.5 4.2 1.4 0.2] expected: [1. 0. 0.] produced: [0.9727786249719081, 0.21925140986698136, 1.8112041353965783e-09]
Sample: [5.7 3.  4.2 1.2] expected: [0. 1. 0.] produced: [0.0006407024074314605, 0.32968451846906566, 0.004479138533934004]
Sample: [5.1 2.5 3.  1.1] expected: [0. 1. 0.] produced: [0.014465415889664254, 0.2985532118070289, 6.623205934879767e-05]
Sample: [6.1 3.  4.6 1.4] expected: [0. 1. 0.] produced: [2.4834835057202165e-05, 0.3762737832615082, 0.2643694104412969]
Sample: [6.4 3.2 5.3 2.3] expected: [0. 0. 1.] produced: [2.412481512638352e-06, 0.4101767017467322, 0.8919788340590042]
Sample: [5.5 2.6 4.4 1.2] expected: [0. 1. 0.] produced: [3.593301220614852e-05, 0.3731169836095682, 0.17849529254197902]
Sample: [4.4 2.9 1.4 0.2] expected: [1. 0. 0.] produced: [0.9709727252514125, 0.22771886178913006, 1.9724989549803672e-09]
Sample: [5.2 3.5 1.5 0.2] expected: [1. 0. 0.] produced: [0.9724655654504653, 0.2264867229592699, 1.82885936924653e-09]
Sample: [4.6 3.1 1.5 0.2] expected: [1. 0. 0.] produced: [0.9715521026405086, 0.22611675291342787, 1.9161696411210006e-09]
Sample: [5.1 3.5 1.4 0.2] expected: [1. 0. 0.] produced: [0.9725069835051763, 0.22509101783870034, 1.8257676225930784e-09]
Sample: [6.7 3.1 5.6 2.4] expected: [0. 0. 1.] produced: [2.1111615624964404e-06, 0.40945477253552937, 0.9079749589819569]
Sample: [5.7 2.9 4.2 1.3] expected: [0. 1. 0.] produced: [0.0007736988954057079, 0.3341495695930675, 0.0034523967687530703]
Sample: [4.5 2.3 1.3 0.3] expected: [1. 0. 0.] produced: [0.9660557632170564, 0.2273051249613533, 2.4582659602225838e-09]
Sample: [6.9 3.1 5.1 2.3] expected: [0. 0. 1.] produced: [3.0870040285335673e-06, 0.40481470743731485, 0.8555133748129302]
Sample: [6.4 3.1 5.5 1.8] expected: [0. 0. 1.] produced: [3.572837731285113e-06, 0.39954416403054255, 0.8297984829948583]
Sample: [5.5 2.4 3.8 1.1] expected: [0. 1. 0.] produced: [0.0004129979427907777, 0.3388878697399862, 0.008056291207787966]
Sample: [4.8 3.1 1.6 0.2] expected: [1. 0. 0.] produced: [0.9712026727686663, 0.22422071505255284, 1.9581171175684396e-09]
Sample: [7.6 3.  6.6 2.1] expected: [0. 0. 1.] produced: [1.85383835279359e-06, 0.4082150927876822, 0.9221453086197478]
Sample: [6.2 3.4 5.4 2.3] expected: [0. 0. 1.] produced: [2.4108081582965717e-06, 0.4013748608880717, 0.892708855518998]
Sample: [6.6 3.  4.4 1.4] expected: [0. 1. 0.] produced: [0.00046891079988186735, 0.33480261683871443, 0.006811804592001868]
Sample: [5.8 4.  1.2 0.2] expected: [1. 0. 0.] produced: [0.9728127613081069, 0.22218224905695044, 1.8090306046012801e-09]
Sample: [5.4 3.4 1.7 0.2] expected: [1. 0. 0.] produced: [0.9720504869884457, 0.22177723207513225, 1.8802097513625526e-09]
Sample: [4.9 2.5 4.5 1.7] expected: [0. 0. 1.] produced: [2.6970115483000484e-06, 0.3994282628365255, 0.8775415641548233]
Sample: [7.2 3.6 6.1 2.5] expected: [0. 0. 1.] produced: [1.991297178838892e-06, 0.39980675496788415, 0.9152878936904678]
Sample: [5.4 3.9 1.7 0.4] expected: [1. 0. 0.] produced: [0.972364517162224, 0.2177264807520354, 1.8545326442516087e-09]
Sample: [5.  2.  3.5 1. ] expected: [0. 1. 0.] produced: [7.459236487737412e-05, 0.35191374064525394, 0.07577423141609879]
Sample: [6.7 3.1 4.7 1.5] expected: [0. 1. 0.] produced: [6.637231914969362e-05, 0.3576927669288066, 0.08747394021901413]
Sample: [4.9 3.  1.4 0.2] expected: [1. 0. 0.] produced: [0.971375217056635, 0.22250228369744268, 1.9475886709187544e-09]
Sample: [5.8 2.7 3.9 1.2] expected: [0. 1. 0.] produced: [0.0007567745718579263, 0.33250887689772324, 0.0035895557232873652]
Sample: [7.7 2.8 6.7 2. ] expected: [0. 0. 1.] produced: [1.7824631895869376e-06, 0.41049882385268455, 0.9261331132388313]
Sample: [5.  3.2 1.2 0.2] expected: [1. 0. 0.] produced: [0.9721394190398316, 0.22242929056942204, 1.8768266269334075e-09]
Sample: [5.1 3.3 1.7 0.5] expected: [1. 0. 0.] produced: [0.97015369656063, 0.2223941133390482, 2.063479873409874e-09]
Sample: [4.8 3.  1.4 0.3] expected: [1. 0. 0.] produced: [0.9707042451050951, 0.22157083969067257, 2.013251620802395e-09]
Sample: [6.3 3.3 4.7 1.6] expected: [0. 1. 0.] produced: [4.186233916862864e-05, 0.3652225660836366, 0.1514152312669135]
Sample: [5.9 3.  4.2 1.5] expected: [0. 1. 0.] produced: [0.00013483893897617958, 0.35558085013543095, 0.035489654876630805]
Sample: [7.3 2.9 6.3 1.8] expected: [0. 0. 1.] produced: [2.0422309075815983e-06, 0.4122645634779519, 0.9124260765509425]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.] produced: [0.9717610435516932, 0.2239418440259699, 1.910048687319432e-09]
Sample: [6.4 3.2 4.5 1.5] expected: [0. 1. 0.] produced: [0.00020466053568187343, 0.3508675135481396, 0.02055540951731335]
Sample: [6.9 3.2 5.7 2.3] expected: [0. 0. 1.] produced: [2.0296656300310877e-06, 0.41264668309662894, 0.9131748588030553]
Sample: [6.5 3.  5.5 1.8] expected: [0. 0. 1.] produced: [2.649653359174651e-06, 0.40566336248974527, 0.8802568596991535]
Sample: [6.8 3.  5.5 2.1] expected: [0. 0. 1.] produced: [2.1830307377331283e-06, 0.40462892587664445, 0.9053182861730449]
Sample: [4.4 3.2 1.3 0.2] expected: [1. 0. 0.] produced: [0.9716321857835228, 0.22077656838292953, 1.930514516305458e-09]
Sample: [6.1 2.8 4.7 1.2] expected: [0. 1. 0.] produced: [3.231129861169647e-05, 0.3671500499015326, 0.20225070801289394]
Sample: [6.2 2.9 4.3 1.3] expected: [0. 1. 0.] produced: [0.0006002377277991684, 0.33679346003668587, 0.004894657599845511]
Sample: [5.5 2.4 3.7 1. ] expected: [0. 1. 0.] produced: [0.001314103006618076, 0.33161313461057984, 0.0017067577821970501]
Sample: [4.4 3.  1.3 0.2] expected: [1. 0. 0.] produced: [0.971162990941265, 0.22799731712996654, 1.9694971982541297e-09]
Sample: [5.7 2.6 3.5 1. ] expected: [0. 1. 0.] produced: [0.02492863111385661, 0.3006514958066516, 3.13857263567942e-05]
Epoch 6000 RMSE =  0.2878962648277072
Epoch 6100 RMSE =  0.28244681477398076
Epoch 6200 RMSE =  0.28816343886266865
Epoch 6300 RMSE =  0.28504945433452844
Epoch 6400 RMSE =  0.28241959566916947
Epoch 6500 RMSE =  0.280861452212918
Epoch 6600 RMSE =  0.2811707369243721
Epoch 6700 RMSE =  0.2839690735662353
Epoch 6800 RMSE =  0.28297174843601974
Epoch 6900 RMSE =  0.2830187752483963
Sample: [6.3 2.7 4.9 1.8] expected: [0. 0. 1.] produced: [3.069578328033464e-06, 0.4114551263316678, 0.8157052368790982]
Sample: [5.  2.3 3.3 1. ] expected: [0. 1. 0.] produced: [0.0017881803020309273, 0.3260820970062962, 0.0005657836643329155]
Sample: [7.7 2.8 6.7 2. ] expected: [0. 0. 1.] produced: [1.583072075891852e-06, 0.4215057581359526, 0.918636798267766]
Sample: [5.8 2.7 5.1 1.9] expected: [0. 0. 1.] produced: [1.934361780634295e-06, 0.4150607446655003, 0.8949833234439717]
Sample: [6.4 2.9 4.3 1.3] expected: [0. 1. 0.] produced: [0.00031570857525644496, 0.3454412852230817, 0.006490176800357364]
Sample: [6.1 2.8 4.  1.3] expected: [0. 1. 0.] produced: [0.0003202578807274121, 0.34941098425997485, 0.006361575673305786]
Sample: [4.8 3.4 1.6 0.2] expected: [1. 0. 0.] produced: [0.9761223885186939, 0.22117714903732558, 4.139144132521109e-10]
Sample: [4.8 3.1 1.6 0.2] expected: [1. 0. 0.] produced: [0.9752154231003219, 0.22089116316913987, 4.369368961251569e-10]
Sample: [5.9 3.  4.2 1.5] expected: [0. 1. 0.] produced: [4.0350543759332996e-05, 0.3786115112839788, 0.1058829045721204]
Sample: [4.6 3.1 1.5 0.2] expected: [1. 0. 0.] produced: [0.9753727106588785, 0.22270727335729856, 4.32784955294569e-10]
Sample: [4.9 3.  1.4 0.2] expected: [1. 0. 0.] produced: [0.9756743390936059, 0.2219148129079361, 4.2503338746758183e-10]
Sample: [6.5 3.  5.5 1.8] expected: [0. 0. 1.] produced: [2.203306592880382e-06, 0.42052129455633586, 0.8765049954758233]
Sample: [5.4 3.9 1.3 0.4] expected: [1. 0. 0.] produced: [0.9770396392653345, 0.21891367352336333, 3.911890609353861e-10]
Sample: [7.9 3.8 6.4 2. ] expected: [0. 0. 1.] produced: [2.550365884592201e-06, 0.4139464225963667, 0.8526827988503316]
Sample: [5.1 2.5 3.  1.1] expected: [0. 1. 0.] produced: [0.008712432270571384, 0.30689701447568424, 6.059502569657853e-05]
Sample: [6.8 3.  5.5 2.1] expected: [0. 0. 1.] produced: [1.7936423990809622e-06, 0.4195714866814864, 0.9050112433336781]
Sample: [6.4 2.8 5.6 2.2] expected: [0. 0. 1.] produced: [1.5900208828063586e-06, 0.4174955201771047, 0.9187215997434541]
Sample: [6.3 3.3 6.  2.5] expected: [0. 0. 1.] produced: [1.5421819797777448e-06, 0.4142158240125569, 0.9219419860897256]
Sample: [6.3 2.8 5.1 1.5] expected: [0. 0. 1.] produced: [3.2194383647552864e-06, 0.40083409489241845, 0.8074245607026336]
Sample: [5.5 4.2 1.4 0.2] expected: [1. 0. 0.] produced: [0.9772327588271006, 0.21221959497708714, 3.8906800248842596e-10]
Sample: [5.8 2.7 3.9 1.2] expected: [0. 1. 0.] produced: [0.00013621488691525075, 0.34887177720298285, 0.02110850028273731]
Sample: [5.5 3.5 1.3 0.2] expected: [1. 0. 0.] produced: [0.9768578225545255, 0.21426342801194784, 3.9845139722987003e-10]
Sample: [7.2 3.  5.8 1.6] expected: [0. 0. 1.] produced: [2.2183255339473278e-06, 0.4054726712614453, 0.8767876133976079]
Sample: [6.7 3.  5.2 2.3] expected: [0. 0. 1.] produced: [1.7050538677825706e-06, 0.40545910997610496, 0.9117309025239742]
Sample: [6.5 3.  5.8 2.2] expected: [0. 0. 1.] produced: [1.5752668807841618e-06, 0.402984246519931, 0.9203800445436348]
Sample: [7.7 3.  6.1 2.3] expected: [0. 0. 1.] produced: [1.5702324176392003e-06, 0.39954065296641994, 0.9207733660759445]
Sample: [7.6 3.  6.6 2.1] expected: [0. 0. 1.] produced: [1.5419529096849252e-06, 0.3963384090936012, 0.9226824001786793]
Sample: [4.4 3.2 1.3 0.2] expected: [1. 0. 0.] produced: [0.9756330036992056, 0.20613485581461877, 4.3183839182826364e-10]
Sample: [4.5 2.3 1.3 0.3] expected: [1. 0. 0.] produced: [0.9657609821518809, 0.20867916907323777, 7.084488807550123e-10]
Sample: [6.7 3.1 4.7 1.5] expected: [0. 1. 0.] produced: [1.3710949180127284e-05, 0.3638397598957148, 0.3549591304238951]
Sample: [6.4 3.1 5.5 1.8] expected: [0. 0. 1.] produced: [2.912703755291035e-06, 0.38818209569242146, 0.8282210493180794]
Sample: [5.7 2.9 4.2 1.3] expected: [0. 1. 0.] produced: [0.00021651518390877904, 0.33192298312832275, 0.011078839540092528]
Sample: [6.3 2.5 4.9 1.5] expected: [0. 1. 0.] produced: [3.7816175709985984e-06, 0.38635813707160904, 0.7701144129616998]
Sample: [5.6 2.9 3.6 1.3] expected: [0. 1. 0.] produced: [0.03818146021148231, 0.2799596914647749, 7.16064767091729e-06]
Sample: [6.5 3.  5.2 2. ] expected: [0. 0. 1.] produced: [3.3194604349197562e-06, 0.397187571254301, 0.7977866065584175]
Sample: [4.8 3.  1.4 0.3] expected: [1. 0. 0.] produced: [0.9758013430293832, 0.21152893476629253, 4.202787374028155e-10]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.] produced: [0.9765134553071686, 0.21066161057246774, 4.0243766046812346e-10]
Sample: [5.6 2.8 4.9 2. ] expected: [0. 0. 1.] produced: [2.323023105677109e-06, 0.3968283360966944, 0.8676128941969661]
Sample: [6.1 2.6 5.6 1.4] expected: [0. 0. 1.] produced: [2.7591373449221427e-06, 0.39126808692764714, 0.8375324772927171]
Sample: [7.  3.2 4.7 1.4] expected: [0. 1. 0.] produced: [0.0015694873990965371, 0.3107886134213782, 0.0006806524839969887]
Sample: [6.7 3.  5.  1.7] expected: [0. 1. 0.] produced: [7.958876574673691e-06, 0.3789965969648378, 0.5377893287772664]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.] produced: [0.9768974627575259, 0.21168572366392766, 3.8924478945844256e-10]
Sample: [6.4 3.2 4.5 1.5] expected: [0. 1. 0.] produced: [0.021415046257174438, 0.2875766916105658, 1.64361986170037e-05]
Sample: [4.9 2.5 4.5 1.7] expected: [0. 0. 1.] produced: [4.561088181140977e-06, 0.3943859814059483, 0.7143264808050346]
Sample: [6.3 2.9 5.6 1.8] expected: [0. 0. 1.] produced: [3.1842688781231104e-06, 0.3958123341304808, 0.8070134710937398]
Sample: [5.1 3.5 1.4 0.3] expected: [1. 0. 0.] produced: [0.9769953864873443, 0.2103377455342714, 3.901106894620997e-10]
Sample: [5.3 3.7 1.5 0.2] expected: [1. 0. 0.] produced: [0.9771846180667798, 0.20967938666761457, 3.8539928065013473e-10]
Sample: [6.1 3.  4.6 1.4] expected: [0. 1. 0.] produced: [0.00041956244079888346, 0.33017887340287533, 0.004329280792361263]
Sample: [6.6 3.  4.4 1.4] expected: [0. 1. 0.] produced: [0.003430118707822015, 0.30927862797715105, 0.0002246420084818232]
Sample: [5.  3.3 1.4 0.2] expected: [1. 0. 0.] produced: [0.9768248241388345, 0.21419114972017717, 3.944517987939617e-10]
Sample: [6.1 2.8 4.7 1.2] expected: [0. 1. 0.] produced: [0.00017042332105640719, 0.34837263145416864, 0.015231471117133915]
Sample: [5.1 3.5 1.4 0.2] expected: [1. 0. 0.] produced: [0.9770051404958503, 0.21600488904093984, 3.8995474478448287e-10]
Sample: [5.5 2.4 3.8 1.1] expected: [0. 1. 0.] produced: [0.0007593829247152749, 0.33332614697171614, 0.0018817266556436483]
Sample: [5.9 3.2 4.8 1.8] expected: [0. 1. 0.] produced: [8.751611200989598e-06, 0.3942997737824595, 0.5029481080211545]
Sample: [5.1 3.3 1.7 0.5] expected: [1. 0. 0.] produced: [0.976376172579109, 0.22063954298971297, 4.003100505283482e-10]
Sample: [6.4 3.2 5.3 2.3] expected: [0. 0. 1.] produced: [4.198942518459217e-06, 0.40805117921544026, 0.7365434488909999]
Sample: [6.8 2.8 4.8 1.4] expected: [0. 1. 0.] produced: [0.0006652506491699835, 0.33938102275805987, 0.002245074981759065]
Sample: [6.4 2.8 5.6 2.1] expected: [0. 0. 1.] produced: [2.138710695702605e-06, 0.4183657750174175, 0.8792380451208198]
Sample: [6.1 3.  4.9 1.8] expected: [0. 0. 1.] produced: [1.250399994152177e-05, 0.3912512891967008, 0.3777736072873414]
Sample: [5.4 3.7 1.5 0.2] expected: [1. 0. 0.] produced: [0.9768964887026742, 0.2172831656969339, 3.958438259570645e-10]
Sample: [7.1 3.  5.9 2.1] expected: [0. 0. 1.] produced: [1.6493582206528266e-06, 0.41404728966565435, 0.9148220408970854]
Sample: [5.5 2.4 3.7 1. ] expected: [0. 1. 0.] produced: [0.00019750245335528155, 0.34887100630786033, 0.012562855913638746]
Sample: [5.1 3.4 1.5 0.2] expected: [1. 0. 0.] produced: [0.9763002050308097, 0.21771083708989689, 4.1111214306840685e-10]
Sample: [4.4 2.9 1.4 0.2] expected: [1. 0. 0.] produced: [0.9739293099164822, 0.21802059297336424, 4.729041033447665e-10]
Sample: [5.  3.6 1.4 0.2] expected: [1. 0. 0.] produced: [0.9767112046586525, 0.21628414988432465, 4.0097066579497105e-10]
Sample: [4.6 3.4 1.4 0.3] expected: [1. 0. 0.] produced: [0.9757957357979996, 0.21604022397985115, 4.242679643973079e-10]
Sample: [5.2 4.1 1.5 0.1] expected: [1. 0. 0.] produced: [0.9771849794158002, 0.21484610119110936, 3.8914857025265574e-10]
Sample: [6.5 2.8 4.6 1.5] expected: [0. 1. 0.] produced: [6.597789030256239e-06, 0.3924901048948082, 0.6041953487254397]
Sample: [5.5 2.3 4.  1.3] expected: [0. 1. 0.] produced: [3.2889736150896124e-05, 0.376294709108261, 0.13480180815493004]
Sample: [5.1 3.8 1.5 0.3] expected: [1. 0. 0.] produced: [0.9771011275066528, 0.21921805610797662, 3.85080235560618e-10]
Sample: [4.4 3.  1.3 0.2] expected: [1. 0. 0.] produced: [0.9761513595899861, 0.2190011301294137, 4.0939159225659836e-10]
Sample: [5.5 2.6 4.4 1.2] expected: [0. 1. 0.] produced: [6.902250015079101e-05, 0.3694521800452886, 0.05191664032222683]
Sample: [5.8 4.  1.2 0.2] expected: [1. 0. 0.] produced: [0.9773598224709903, 0.220337752057612, 3.786632380245246e-10]
Sample: [6.9 3.1 4.9 1.5] expected: [0. 1. 0.] produced: [0.00017151961453517074, 0.36117282851448285, 0.014967655793894238]
Sample: [5.4 3.4 1.7 0.2] expected: [1. 0. 0.] produced: [0.9767824545500078, 0.22247901685671004, 3.9274077514440223e-10]
Sample: [6.3 3.4 5.6 2.4] expected: [0. 0. 1.] produced: [1.941348279032482e-06, 0.42485355403760144, 0.8930781814681122]
Sample: [6.6 2.9 4.6 1.3] expected: [0. 1. 0.] produced: [0.00044258265514329385, 0.34914964000886667, 0.003988628580279618]
Sample: [5.  3.2 1.2 0.2] expected: [1. 0. 0.] produced: [0.9768170639353911, 0.2225720873323016, 3.9271831800582536e-10]
Sample: [5.4 3.9 1.7 0.4] expected: [1. 0. 0.] produced: [0.9769588319586617, 0.2218337452034654, 3.888414376622704e-10]
Sample: [5.  2.  3.5 1. ] expected: [0. 1. 0.] produced: [0.00013947757161815512, 0.36650479123639135, 0.01996314056443739]
Sample: [5.7 3.  4.2 1.2] expected: [0. 1. 0.] produced: [0.0012814042005684799, 0.3420115377382443, 0.0008947448709012299]
Sample: [5.  3.4 1.5 0.2] expected: [1. 0. 0.] produced: [0.976663105802768, 0.22643903551682443, 3.9632463423042326e-10]
Sample: [6.9 3.2 5.7 2.3] expected: [0. 0. 1.] produced: [1.8357561284308168e-06, 0.4334375234355235, 0.9005181173844666]
Sample: [5.6 3.  4.5 1.5] expected: [0. 1. 0.] produced: [1.786950199368165e-05, 0.39829983055999907, 0.2690231028395656]
Sample: [5.7 3.8 1.7 0.3] expected: [1. 0. 0.] produced: [0.9771425321420902, 0.22618685123082874, 3.8270280694048436e-10]
Sample: [5.2 2.7 3.9 1.4] expected: [0. 1. 0.] produced: [0.00015535007338226265, 0.3728337609166763, 0.017101940577268878]
Sample: [6.  2.9 4.5 1.5] expected: [0. 1. 0.] produced: [5.445589848111881e-05, 0.3913976109864244, 0.07073038112408472]
Sample: [6.5 3.2 5.1 2. ] expected: [0. 0. 1.] produced: [4.949741407876926e-06, 0.42942554869450106, 0.6899404919118851]
Sample: [7.2 3.6 6.1 2.5] expected: [0. 0. 1.] produced: [1.7608508063016193e-06, 0.44028965366845685, 0.9058951732582087]
Sample: [6.7 3.1 5.6 2.4] expected: [0. 0. 1.] produced: [1.7105459206612093e-06, 0.43671657958971094, 0.9094156404401638]
Sample: [5.2 3.5 1.5 0.2] expected: [1. 0. 0.] produced: [0.9768624872334699, 0.22510757652007796, 3.926972672715121e-10]
Sample: [7.3 2.9 6.3 1.8] expected: [0. 0. 1.] produced: [1.8340228914627196e-06, 0.4308454311521615, 0.9011055560363679]
Sample: [6.  2.2 5.  1.5] expected: [0. 0. 1.] produced: [2.244601916093541e-06, 0.4242094064697444, 0.8728564158959534]
Sample: [7.2 3.2 6.  1.8] expected: [0. 0. 1.] produced: [2.4696166472377405e-06, 0.4191718000034233, 0.8574169442768923]
Sample: [6.3 3.3 4.7 1.6] expected: [0. 1. 0.] produced: [3.930231277129816e-05, 0.3785462239182548, 0.10909314259828498]
Sample: [5.7 2.6 3.5 1. ] expected: [0. 1. 0.] produced: [0.02723693194336263, 0.30039729090621553, 1.1762870416722106e-05]
Sample: [5.2 3.4 1.4 0.2] expected: [1. 0. 0.] produced: [0.9767977907959676, 0.22413611621246032, 3.954955213568322e-10]
Sample: [6.  2.2 4.  1. ] expected: [0. 1. 0.] produced: [0.00011549441655382885, 0.3719772101749079, 0.026099745393572262]
Sample: [4.7 3.2 1.6 0.2] expected: [1. 0. 0.] produced: [0.975657186387604, 0.22651187471846146, 4.241109101414649e-10]
Sample: [4.3 3.  1.1 0.1] expected: [1. 0. 0.] produced: [0.9762667657300242, 0.22559259802887652, 4.1013943125878406e-10]
Sample: [6.9 3.1 5.1 2.3] expected: [0. 0. 1.] produced: [2.1112244536839853e-06, 0.42897819167238876, 0.8824553052015269]
Sample: [6.7 3.3 5.7 2.5] expected: [0. 0. 1.] produced: [1.6510645879114523e-06, 0.4285660451010638, 0.914030859664506]
Sample: [6.2 2.9 4.3 1.3] expected: [0. 1. 0.] produced: [0.00014486545022735482, 0.3648565645293415, 0.01915493087954166]
Sample: [6.7 3.3 5.7 2.1] expected: [0. 0. 1.] produced: [1.9140485704796883e-06, 0.4276034955846066, 0.8962943465275175]
Sample: [6.2 3.4 5.4 2.3] expected: [0. 0. 1.] produced: [1.8484318921121828e-06, 0.42428162847219575, 0.9009001966427725]
Epoch 7000 RMSE =  0.28216553175174647
Epoch 7100 RMSE =  0.2815571969447038
Epoch 7200 RMSE =  0.2833829961732334
Epoch 7300 RMSE =  0.2825226122768832
Epoch 7400 RMSE =  0.28638974083955965
Epoch 7500 RMSE =  0.28093463906255595
Epoch 7600 RMSE =  0.2812662882714993
Epoch 7700 RMSE =  0.2805871523685584
Epoch 7800 RMSE =  0.2827816138465527
Epoch 7900 RMSE =  0.2853466665436227
Sample: [5.5 3.5 1.3 0.2] expected: [1. 0. 0.] produced: [0.9787449131302228, 0.21220514752175096, 9.722446750069057e-11]
Sample: [6.7 3.3 5.7 2.5] expected: [0. 0. 1.] produced: [1.5058765520274933e-06, 0.4165757676421607, 0.9043089443659033]
Sample: [6.7 3.  5.  1.7] expected: [0. 1. 0.] produced: [8.641959480297818e-06, 0.389217836162796, 0.42126655187584156]
Sample: [6.4 3.2 4.5 1.5] expected: [0. 1. 0.] produced: [0.02458228785485785, 0.2923705054695647, 5.897333516185953e-06]
Sample: [5.8 2.7 5.1 1.9] expected: [0. 0. 1.] produced: [2.892757112168089e-06, 0.41326184202446437, 0.7814710386343509]
Sample: [5.  3.2 1.2 0.2] expected: [1. 0. 0.] produced: [0.9786557979908976, 0.2131697581538812, 9.736003421350866e-11]
Sample: [5.1 3.5 1.4 0.3] expected: [1. 0. 0.] produced: [0.978690952640582, 0.2125492787619939, 9.70688936720349e-11]
Sample: [7.2 3.2 6.  1.8] expected: [0. 0. 1.] produced: [5.597469766176377e-06, 0.3989969226248108, 0.5771078339974659]
Sample: [6.4 2.9 4.3 1.3] expected: [0. 1. 0.] produced: [0.0005085399695858061, 0.33670053212825934, 0.0018422640176826595]
Sample: [5.5 2.6 4.4 1.2] expected: [0. 1. 0.] produced: [1.744022394882241e-05, 0.3849730454896337, 0.20704802831650657]
Sample: [7.6 3.  6.6 2.1] expected: [0. 0. 1.] produced: [1.403813852759199e-06, 0.4244013199474507, 0.9130286922773284]
Sample: [5.5 2.4 3.8 1.1] expected: [0. 1. 0.] produced: [0.0004382461199443906, 0.3437313953486218, 0.0022834531953474686]
Sample: [5.6 3.  4.5 1.5] expected: [0. 1. 0.] produced: [2.2022179251995192e-05, 0.38751167027116556, 0.15591859291899243]
Sample: [4.4 3.  1.3 0.2] expected: [1. 0. 0.] produced: [0.9777023093307312, 0.21898478027362103, 1.047277305774434e-10]
Sample: [6.3 2.8 5.1 1.5] expected: [0. 0. 1.] produced: [6.160542616470675e-06, 0.40885098585550744, 0.5444733718956273]
Sample: [7.9 3.8 6.4 2. ] expected: [0. 0. 1.] produced: [1.831516128324454e-06, 0.422317595028058, 0.878179154348305]
Sample: [7.3 2.9 6.3 1.8] expected: [0. 0. 1.] produced: [1.3725819446958684e-06, 0.42260995723451195, 0.9168841304528619]
Sample: [6.6 3.  4.4 1.4] expected: [0. 1. 0.] produced: [4.250765152947269e-05, 0.3724092368896014, 0.0666741378480945]
Sample: [6.2 2.9 4.3 1.3] expected: [0. 1. 0.] produced: [4.0432481885572856e-05, 0.3776073617197502, 0.07135991783330985]
Sample: [6.1 2.8 4.7 1.2] expected: [0. 1. 0.] produced: [9.624224750812465e-06, 0.4017551025947069, 0.38719795472487795]
Sample: [6.9 3.1 4.9 1.5] expected: [0. 1. 0.] produced: [6.66296477205767e-05, 0.37994876201296013, 0.035148541596973275]
Sample: [6.3 2.7 4.9 1.8] expected: [0. 0. 1.] produced: [2.6642235154837927e-06, 0.4294421091387551, 0.80431108116926]
Sample: [6.7 3.  5.2 2.3] expected: [0. 0. 1.] produced: [1.6635127021634377e-06, 0.43234757452948186, 0.891812686348462]
Sample: [6.1 2.6 5.6 1.4] expected: [0. 0. 1.] produced: [1.9536956976563243e-06, 0.4262314916571714, 0.867021904975091]
Sample: [6.8 2.8 4.8 1.4] expected: [0. 1. 0.] produced: [1.9887776957917113e-05, 0.3905195011389731, 0.17804919512116266]
Sample: [5.3 3.7 1.5 0.2] expected: [1. 0. 0.] produced: [0.97873261374222, 0.21968026851084685, 9.79025661184316e-11]
Sample: [5.1 2.5 3.  1.1] expected: [0. 1. 0.] produced: [0.05178248156524366, 0.2923052284876068, 1.9317205855936605e-06]
Sample: [5.  2.  3.5 1. ] expected: [0. 1. 0.] produced: [9.411520419958796e-05, 0.37711349853424125, 0.021554556247012902]
Sample: [6.1 3.  4.9 1.8] expected: [0. 0. 1.] produced: [3.5918614693136514e-06, 0.42706167670599077, 0.7269147772937001]
Sample: [5.7 2.6 3.5 1. ] expected: [0. 1. 0.] produced: [0.011088477278025212, 0.3158704209125817, 1.9858266499650488e-05]
Sample: [4.9 2.5 4.5 1.7] expected: [0. 0. 1.] produced: [1.7979682876275142e-06, 0.437635989573213, 0.8811054463302536]
Sample: [6.4 3.2 5.3 2.3] expected: [0. 0. 1.] produced: [1.5188412243398944e-06, 0.436106272181564, 0.9048588483555781]
Sample: [7.7 2.8 6.7 2. ] expected: [0. 0. 1.] produced: [1.2929975794916634e-06, 0.4344492283569619, 0.9234384234080976]
Sample: [5.2 3.5 1.5 0.2] expected: [1. 0. 0.] produced: [0.9783267560674365, 0.2194675496715126, 1.0150946183783348e-10]
Sample: [7.2 3.  5.8 1.6] expected: [0. 0. 1.] produced: [2.1099830137158244e-06, 0.4227849118919187, 0.8546840424181015]
Sample: [6.3 2.5 4.9 1.5] expected: [0. 1. 0.] produced: [2.363188904937541e-06, 0.4175267423939765, 0.8331430850137194]
Sample: [5.5 2.4 3.7 1. ] expected: [0. 1. 0.] produced: [0.0008059361243781831, 0.3441215892668182, 0.0009372020237228318]
Sample: [6.4 2.8 5.6 2.1] expected: [0. 0. 1.] produced: [1.4049380816217136e-06, 0.43459703055037646, 0.9132938752220248]
Sample: [6.5 2.8 4.6 1.5] expected: [0. 1. 0.] produced: [1.2300137564010954e-05, 0.4006493829648625, 0.3037252290968483]
Sample: [5.9 3.  4.2 1.5] expected: [0. 1. 0.] produced: [0.0004508375638354307, 0.3567262098227017, 0.002180980680624516]
Sample: [6.3 3.4 5.6 2.4] expected: [0. 0. 1.] produced: [1.6819020908928732e-06, 0.4380125588488126, 0.8891650234464746]
Sample: [6.5 3.  5.5 1.8] expected: [0. 0. 1.] produced: [2.689273328476595e-06, 0.4274823767374842, 0.8013706302131409]
Sample: [6.5 3.  5.8 2.2] expected: [0. 0. 1.] produced: [1.4699268482824415e-06, 0.432196449876163, 0.9077332889548194]
Sample: [4.8 3.1 1.6 0.2] expected: [1. 0. 0.] produced: [0.9776042213864615, 0.22005970795204438, 1.0559358404218489e-10]
Sample: [6.  2.2 5.  1.5] expected: [0. 0. 1.] produced: [2.0820616656674625e-06, 0.42256836609008636, 0.8552525468125541]
Sample: [4.4 2.9 1.4 0.2] expected: [1. 0. 0.] produced: [0.9770425328183605, 0.2179095740799465, 1.0998543242194315e-10]
Sample: [5.1 3.4 1.5 0.2] expected: [1. 0. 0.] produced: [0.9784283860511501, 0.2166449257088373, 9.995849481389172e-11]
Sample: [4.6 3.4 1.4 0.3] expected: [1. 0. 0.] produced: [0.9781403294581157, 0.2161603634905953, 1.0205980957575799e-10]
Sample: [6.7 3.1 4.7 1.5] expected: [0. 1. 0.] produced: [0.00012193475141858541, 0.3616225366365153, 0.014827672638854615]
Sample: [6.  2.9 4.5 1.5] expected: [0. 1. 0.] produced: [2.2988009900968365e-05, 0.3882110467406944, 0.14847212017150552]
Sample: [6.3 3.3 6.  2.5] expected: [0. 0. 1.] produced: [1.3626590948305134e-06, 0.43196342752146244, 0.9167445886567154]
Sample: [7.7 3.  6.1 2.3] expected: [0. 0. 1.] produced: [1.4865489706410422e-06, 0.42687146086594474, 0.9065409010230477]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.] produced: [0.97813970584974, 0.21693191810119375, 1.0198864726755672e-10]
Sample: [5.4 3.9 1.3 0.4] expected: [1. 0. 0.] produced: [0.9788024151139318, 0.21599720676848413, 9.730920498674192e-11]
Sample: [6.3 3.3 4.7 1.6] expected: [0. 1. 0.] produced: [0.00010046284130170198, 0.36453666997577117, 0.019602188426684763]
Sample: [7.1 3.  5.9 2.1] expected: [0. 0. 1.] produced: [1.584095025797857e-06, 0.4253200262029715, 0.8984451794220605]
Sample: [5.7 3.8 1.7 0.3] expected: [1. 0. 0.] produced: [0.9786743017066949, 0.21616726870326514, 9.826343219268902e-11]
Sample: [6.6 2.9 4.6 1.3] expected: [0. 1. 0.] produced: [0.0002092635090989758, 0.35519365048880513, 0.006769649322593041]
Sample: [5.8 2.7 3.9 1.2] expected: [0. 1. 0.] produced: [0.0008323298979835667, 0.341555302485909, 0.0008962956644977437]
Sample: [6.1 2.8 4.  1.3] expected: [0. 1. 0.] produced: [0.00048211537776018313, 0.352641778630905, 0.001996852074171321]
Sample: [5.6 2.8 4.9 2. ] expected: [0. 0. 1.] produced: [1.6566430044765897e-06, 0.4344310001800027, 0.892427161575181]
Sample: [4.5 2.3 1.3 0.3] expected: [1. 0. 0.] produced: [0.9707330127148589, 0.22454960309339034, 1.5908478612844563e-10]
Sample: [5.2 3.4 1.4 0.2] expected: [1. 0. 0.] produced: [0.9783978404095154, 0.22076154528993266, 1.0035932167460746e-10]
Sample: [5.7 2.9 4.2 1.3] expected: [0. 1. 0.] produced: [9.143769596976909e-05, 0.37370008795485066, 0.022506266271667918]
Sample: [5.2 4.1 1.5 0.1] expected: [1. 0. 0.] produced: [0.9788606391251428, 0.22241461198990828, 9.712047314744588e-11]
Sample: [4.9 3.  1.4 0.2] expected: [1. 0. 0.] produced: [0.9773825840149308, 0.2224461328237551, 1.0760897534023646e-10]
Sample: [6.1 3.  4.6 1.4] expected: [0. 1. 0.] produced: [2.1078820350377557e-05, 0.3962758396154549, 0.16566253217570134]
Sample: [5.9 3.2 4.8 1.8] expected: [0. 1. 0.] produced: [5.050114528536214e-06, 0.4209263755010471, 0.6172642695046271]
Sample: [5.6 2.9 3.6 1.3] expected: [0. 1. 0.] produced: [0.11827125413811918, 0.290887843396932, 5.070525851292653e-07]
Sample: [5.2 2.7 3.9 1.4] expected: [0. 1. 0.] produced: [0.0002531434289608381, 0.3751448031077856, 0.005013781702640967]
Sample: [5.5 4.2 1.4 0.2] expected: [1. 0. 0.] produced: [0.9789411115451008, 0.23115411118126786, 9.477782707487626e-11]
Sample: [6.7 3.1 5.6 2.4] expected: [0. 0. 1.] produced: [1.6406218511286622e-06, 0.44990923753641576, 0.891423491960354]
Sample: [5.  3.3 1.4 0.2] expected: [1. 0. 0.] produced: [0.9784828950119291, 0.2286899157355408, 9.807263165154494e-11]
Sample: [6.7 3.3 5.7 2.1] expected: [0. 0. 1.] produced: [2.464033743458603e-06, 0.43898521861924733, 0.819065233772999]
Sample: [6.9 3.1 5.1 2.3] expected: [0. 0. 1.] produced: [2.6031466190931558e-06, 0.4343171388602816, 0.8074198907728491]
Sample: [5.7 3.  4.2 1.2] expected: [0. 1. 0.] produced: [0.002946754916396032, 0.3350531720796531, 0.00013796286710639018]
Sample: [6.3 2.9 5.6 1.8] expected: [0. 0. 1.] produced: [1.91546506022753e-06, 0.43946956429491063, 0.8685259060341901]
Sample: [7.  3.2 4.7 1.4] expected: [0. 1. 0.] produced: [0.0011302687657268875, 0.34823894566592545, 0.0005655192723514579]
Sample: [6.4 2.8 5.6 2.2] expected: [0. 0. 1.] produced: [1.4020532169409236e-06, 0.4447145970680945, 0.9127933133652214]
Sample: [6.8 3.  5.5 2.1] expected: [0. 0. 1.] produced: [1.7328524017353256e-06, 0.4376689714774745, 0.8847606393306306]
Sample: [5.8 4.  1.2 0.2] expected: [1. 0. 0.] produced: [0.9789202780784785, 0.22348381439862275, 9.589255600233465e-11]
Sample: [6.9 3.2 5.7 2.3] expected: [0. 0. 1.] produced: [1.5597878194765541e-06, 0.43427513999203676, 0.8997606625736566]
Sample: [5.  2.3 3.3 1. ] expected: [0. 1. 0.] produced: [0.001826922900984308, 0.3356699923342959, 0.0002803582623688795]
Sample: [4.7 3.2 1.6 0.2] expected: [1. 0. 0.] produced: [0.9774607143060776, 0.22422142091088043, 1.0624739562066123e-10]
Sample: [4.6 3.1 1.5 0.2] expected: [1. 0. 0.] produced: [0.9773200923781594, 0.22361207946667716, 1.0730206162849257e-10]
Sample: [5.  3.4 1.5 0.2] expected: [1. 0. 0.] produced: [0.9782417272643702, 0.2225101212082371, 1.0073574076919516e-10]
Sample: [5.4 3.7 1.5 0.2] expected: [1. 0. 0.] produced: [0.9786617257278512, 0.22164414144680392, 9.779302596056363e-11]
Sample: [6.5 3.2 5.1 2. ] expected: [0. 0. 1.] produced: [2.7783057835220057e-06, 0.4231875558384813, 0.7939211705798999]
Sample: [6.  2.2 4.  1. ] expected: [0. 1. 0.] produced: [7.259550412777543e-05, 0.3751042824327871, 0.031161277097109034]
Sample: [5.1 3.5 1.4 0.2] expected: [1. 0. 0.] produced: [0.9784180608772793, 0.2218483608774852, 9.984500797423707e-11]
Sample: [5.4 3.9 1.7 0.4] expected: [1. 0. 0.] produced: [0.9783880341312375, 0.22119715116728939, 1.0000707763870195e-10]
Sample: [4.8 3.4 1.6 0.2] expected: [1. 0. 0.] produced: [0.977858488055273, 0.22079317962570114, 1.0379453645713315e-10]
Sample: [4.3 3.  1.1 0.1] expected: [1. 0. 0.] produced: [0.9778601868794191, 0.22017262168845916, 1.0412533281112576e-10]
Sample: [7.2 3.6 6.1 2.5] expected: [0. 0. 1.] produced: [1.4109800478645247e-06, 0.43016322053578115, 0.9128143192459048]
Sample: [4.4 3.2 1.3 0.2] expected: [1. 0. 0.] produced: [0.9777031533732881, 0.2177753329805935, 1.0514809556695263e-10]
Sample: [5.5 2.3 4.  1.3] expected: [0. 1. 0.] produced: [8.93864643927722e-06, 0.3999435301833954, 0.41080939912092385]
Sample: [5.4 3.4 1.7 0.2] expected: [1. 0. 0.] produced: [0.9784659926912457, 0.21923420408363453, 9.859202763568133e-11]
Sample: [6.2 3.4 5.4 2.3] expected: [0. 0. 1.] produced: [2.163311686299577e-06, 0.42357077857105646, 0.8465835777791344]
Sample: [4.8 3.  1.4 0.3] expected: [1. 0. 0.] produced: [0.9776025353850257, 0.21723184748601365, 1.0492736333818905e-10]
Sample: [5.  3.6 1.4 0.2] expected: [1. 0. 0.] produced: [0.9786965188077406, 0.21610258642099625, 9.72624739143649e-11]
Sample: [6.4 3.1 5.5 1.8] expected: [0. 0. 1.] produced: [3.200621541927701e-06, 0.41265569561166426, 0.7569671898721564]
Sample: [6.5 3.  5.2 2. ] expected: [0. 0. 1.] produced: [2.411884159059031e-06, 0.4130387221193832, 0.826059829701215]
Sample: [5.1 3.8 1.5 0.3] expected: [1. 0. 0.] produced: [0.9786416551900949, 0.212139888496965, 9.830034662081149e-11]
Sample: [5.1 3.3 1.7 0.5] expected: [1. 0. 0.] produced: [0.9769166573328077, 0.21229774259680484, 1.1047297577553819e-10]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.] produced: [0.97806420494003, 0.2112207293890671, 1.0240130278957034e-10]
Epoch 8000 RMSE =  0.2818041018034963
Epoch 8100 RMSE =  0.28639910542462504
Epoch 8200 RMSE =  0.28447554164005673
Epoch 8300 RMSE =  0.2846465627675503
Epoch 8400 RMSE =  0.2819034566599393
Epoch 8500 RMSE =  0.27964236396391395
Epoch 8600 RMSE =  0.2822657100922676
Epoch 8700 RMSE =  0.27705143089021117
Epoch 8800 RMSE =  0.2828341376855515
Epoch 8900 RMSE =  0.2812285438464064
Sample: [6.4 2.8 5.6 2.2] expected: [0. 0. 1.] produced: [1.382480400555282e-06, 0.42571823523824953, 0.9095106027468489]
Sample: [5.  2.3 3.3 1. ] expected: [0. 1. 0.] produced: [0.003708615848875915, 0.3156363325872728, 5.840850058494193e-05]
Sample: [5.1 3.5 1.4 0.3] expected: [1. 0. 0.] produced: [0.981518857150359, 0.211643439162696, 2.6602135412882035e-11]
Sample: [6.5 3.2 5.1 2. ] expected: [0. 0. 1.] produced: [3.122592590753102e-06, 0.41406863698694346, 0.7436493329045687]
Sample: [4.3 3.  1.1 0.1] expected: [1. 0. 0.] produced: [0.9811007247472512, 0.20963811879376182, 2.7773981183309126e-11]
Sample: [6.8 3.  5.5 2.1] expected: [0. 0. 1.] produced: [1.5680709017239303e-06, 0.4194158495871538, 0.8931901779851428]
Sample: [6.4 3.2 4.5 1.5] expected: [0. 1. 0.] produced: [0.00010641311562731324, 0.3579939550593825, 0.013206468268750115]
Sample: [5.  3.2 1.2 0.2] expected: [1. 0. 0.] produced: [0.9813552067328887, 0.20965554328442726, 2.7146945606964178e-11]
Sample: [6.6 2.9 4.6 1.3] expected: [0. 1. 0.] produced: [5.923579130383787e-05, 0.36942968135424087, 0.031697606013113215]
Sample: [6.3 2.7 4.9 1.8] expected: [0. 0. 1.] produced: [1.9807162590868668e-06, 0.42141282408975833, 0.8542715898778307]
Sample: [5.4 3.9 1.3 0.4] expected: [1. 0. 0.] produced: [0.9817103100575475, 0.2096145316472788, 2.6369794057409264e-11]
Sample: [5.  3.3 1.4 0.2] expected: [1. 0. 0.] produced: [0.9811949933895554, 0.20931817765983107, 2.7552460003775746e-11]
Sample: [6.7 3.  5.2 2.3] expected: [0. 0. 1.] produced: [1.451884714094583e-06, 0.4204388299926397, 0.9042595960320705]
Sample: [6.1 3.  4.6 1.4] expected: [0. 1. 0.] produced: [1.668600995380855e-05, 0.3828431843599318, 0.1852386616306149]
Sample: [5.8 2.7 5.1 1.9] expected: [0. 0. 1.] produced: [1.5699482539459207e-06, 0.4206600986454031, 0.8931559106493515]
Sample: [6.7 3.  5.  1.7] expected: [0. 1. 0.] produced: [4.881417983005824e-06, 0.4010747673028113, 0.5969969449461915]
Sample: [6.3 3.4 5.6 2.4] expected: [0. 0. 1.] produced: [2.09624939390369e-06, 0.417939544190033, 0.8406065493960858]
Sample: [6.4 2.9 4.3 1.3] expected: [0. 1. 0.] produced: [0.019285238531524695, 0.2938312408740973, 4.57265644921781e-06]
Sample: [5.  2.  3.5 1. ] expected: [0. 1. 0.] produced: [0.0006488127400269676, 0.3402521941525824, 0.0008336098524288474]
Sample: [5.4 3.9 1.7 0.4] expected: [1. 0. 0.] produced: [0.9817538041413679, 0.21306266334097404, 2.5925860701564757e-11]
Sample: [5.7 3.8 1.7 0.3] expected: [1. 0. 0.] produced: [0.9818181099641924, 0.21242871033331326, 2.5783232615513157e-11]
Sample: [5.1 3.4 1.5 0.2] expected: [1. 0. 0.] produced: [0.9816488604348436, 0.21193257170794771, 2.6173285113183845e-11]
Sample: [7.2 3.  5.8 1.6] expected: [0. 0. 1.] produced: [6.4721960941691e-06, 0.4047998921915976, 0.4862836223357674]
Sample: [5.1 3.3 1.7 0.5] expected: [1. 0. 0.] produced: [0.9790027263416409, 0.21104821494736287, 3.269654769053083e-11]
Sample: [6.3 3.3 4.7 1.6] expected: [0. 1. 0.] produced: [1.1204556934342946e-05, 0.39299955509402323, 0.2942918410108538]
Sample: [4.8 3.4 1.6 0.2] expected: [1. 0. 0.] produced: [0.9812309711277113, 0.2117721914530195, 2.729924640505443e-11]
Sample: [4.4 3.2 1.3 0.2] expected: [1. 0. 0.] produced: [0.9811045959885942, 0.2112650202232018, 2.763035298458159e-11]
Sample: [7.3 2.9 6.3 1.8] expected: [0. 0. 1.] produced: [1.5003235385687806e-06, 0.42458092208173936, 0.8990296583863319]
Sample: [6.  2.2 5.  1.5] expected: [0. 0. 1.] produced: [1.862165603977832e-06, 0.4177514413011427, 0.8650665746274068]
Sample: [6.1 2.8 4.7 1.2] expected: [0. 1. 0.] produced: [4.033296433206284e-05, 0.37158733347343437, 0.05546833619676793]
Sample: [7.6 3.  6.6 2.1] expected: [0. 0. 1.] produced: [1.30367882332235e-06, 0.42414041426071747, 0.9171515942743441]
Sample: [6.  2.9 4.5 1.5] expected: [0. 1. 0.] produced: [1.7443242582742382e-05, 0.38417606818485067, 0.17441514228427904]
Sample: [5.2 3.5 1.5 0.2] expected: [1. 0. 0.] produced: [0.981634162714903, 0.21010382880971284, 2.6393396508727245e-11]
Sample: [5.  3.6 1.4 0.2] expected: [1. 0. 0.] produced: [0.9817073252054849, 0.209488827508474, 2.623689981014442e-11]
Sample: [4.5 2.3 1.3 0.3] expected: [1. 0. 0.] produced: [0.9767335890734666, 0.2113263088631527, 3.829635159528448e-11]
Sample: [6.5 3.  5.5 1.8] expected: [0. 0. 1.] produced: [2.3348359236196195e-06, 0.4145696335937407, 0.8193832010981753]
Sample: [7.2 3.2 6.  1.8] expected: [0. 0. 1.] produced: [2.2307010757515942e-06, 0.4116456137633885, 0.8300154939977323]
Sample: [6.4 3.1 5.5 1.8] expected: [0. 0. 1.] produced: [2.2328960831580336e-06, 0.40809981239752663, 0.8302994141196174]
Sample: [5.6 2.8 4.9 2. ] expected: [0. 0. 1.] produced: [1.6619568705867132e-06, 0.4087202962671108, 0.8851514756763872]
Sample: [5.8 2.7 3.9 1.2] expected: [0. 1. 0.] produced: [0.0006782143716614729, 0.32553418927332334, 0.0007954716229109057]
Sample: [5.4 3.7 1.5 0.2] expected: [1. 0. 0.] produced: [0.9817151824309013, 0.2042375406263262, 2.643916177933936e-11]
Sample: [5.1 2.5 3.  1.1] expected: [0. 1. 0.] produced: [0.017515058817346447, 0.28861733415711743, 5.417996382554014e-06]
Sample: [5.1 3.5 1.4 0.2] expected: [1. 0. 0.] produced: [0.9815302437614899, 0.2061790141626371, 2.6866256243477576e-11]
Sample: [5.6 2.9 3.6 1.3] expected: [0. 1. 0.] produced: [0.00205027649748167, 0.31765715512456705, 0.0001467082265002003]
Sample: [6.3 3.3 6.  2.5] expected: [0. 0. 1.] produced: [1.2426345654189602e-06, 0.421305033857636, 0.9232859595397686]
Sample: [6.8 2.8 4.8 1.4] expected: [0. 1. 0.] produced: [9.836367432749876e-06, 0.38883479854916614, 0.33844910110919063]
Sample: [5.5 3.5 1.3 0.2] expected: [1. 0. 0.] produced: [0.9818205322353821, 0.20859934924751092, 2.6022866812705783e-11]
Sample: [6.3 2.8 5.1 1.5] expected: [0. 0. 1.] produced: [6.766311292317681e-06, 0.398006722858973, 0.47265455034046205]
Sample: [5.7 2.6 3.5 1. ] expected: [0. 1. 0.] produced: [0.001936037145012388, 0.319851136320464, 0.00016141494964367638]
Sample: [5.2 4.1 1.5 0.1] expected: [1. 0. 0.] produced: [0.9818020547571302, 0.20890884604610818, 2.6407795276531842e-11]
Sample: [5.9 3.  4.2 1.5] expected: [0. 1. 0.] produced: [1.1070246659755549e-05, 0.3916065865719902, 0.3009644526985193]
Sample: [7.2 3.6 6.1 2.5] expected: [0. 0. 1.] produced: [1.3500672598121305e-06, 0.42615002796202994, 0.9138215510520361]
Sample: [6.6 3.  4.4 1.4] expected: [0. 1. 0.] produced: [0.0001164536975379383, 0.36070124875989656, 0.011615142246323968]
Sample: [6.7 3.3 5.7 2.5] expected: [0. 0. 1.] produced: [1.3030278129631389e-06, 0.42778297116905506, 0.9180663450142861]
Sample: [5.1 3.8 1.5 0.3] expected: [1. 0. 0.] produced: [0.9815593004994165, 0.20985040895110385, 2.6827094974166254e-11]
Sample: [5.2 3.4 1.4 0.2] expected: [1. 0. 0.] produced: [0.9814261676736588, 0.2093465377062579, 2.7135075776685327e-11]
Sample: [4.9 3.  1.4 0.2] expected: [1. 0. 0.] produced: [0.9805517534589296, 0.20923140211511618, 2.917073469128948e-11]
Sample: [6.2 3.4 5.4 2.3] expected: [0. 0. 1.] produced: [1.4722748879739074e-06, 0.4196845188187426, 0.9029972578536232]
Sample: [4.4 3.  1.3 0.2] expected: [1. 0. 0.] produced: [0.9803198831469541, 0.20710310708627197, 2.9786391321795236e-11]
Sample: [6.9 3.2 5.7 2.3] expected: [0. 0. 1.] produced: [1.367811960137683e-06, 0.4161927008156106, 0.912510149942124]
Sample: [4.6 3.1 1.5 0.2] expected: [1. 0. 0.] produced: [0.9802817452387695, 0.204896305232165, 2.9863883995989874e-11]
Sample: [5.5 2.6 4.4 1.2] expected: [0. 1. 0.] produced: [9.428270373857899e-06, 0.3850829961943449, 0.35389691586559713]
Sample: [6.1 3.  4.9 1.8] expected: [0. 0. 1.] produced: [4.576921690675752e-06, 0.3998913125118446, 0.6201370647416831]
Sample: [6.4 3.2 5.3 2.3] expected: [0. 0. 1.] produced: [1.4743653915070392e-06, 0.41235466227013323, 0.9031429875504698]
Sample: [6.7 3.1 4.7 1.5] expected: [0. 1. 0.] produced: [2.6050114226530252e-05, 0.36958127969951265, 0.10420521051143623]
Sample: [6.4 2.8 5.6 2.1] expected: [0. 0. 1.] produced: [1.3317745762346078e-06, 0.41518819227570347, 0.915896372221023]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.] produced: [0.9809957081601385, 0.20392697250957004, 2.8222852742076557e-11]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.] produced: [0.9809966162354267, 0.20339000670077254, 2.8222280150802068e-11]
Sample: [6.5 3.  5.8 2.2] expected: [0. 0. 1.] produced: [1.3113813094528737e-06, 0.410151857305392, 0.9177749057588336]
Sample: [5.7 3.  4.2 1.2] expected: [0. 1. 0.] produced: [0.0003182163650184209, 0.3335175757610327, 0.0025405355790344597]
Sample: [5.4 3.4 1.7 0.2] expected: [1. 0. 0.] produced: [0.9811856136865191, 0.20354630884607494, 2.7782797904508625e-11]
Sample: [5.  3.4 1.5 0.2] expected: [1. 0. 0.] produced: [0.9812505114031377, 0.2029838811577086, 2.7647574244211128e-11]
Sample: [6.9 3.1 4.9 1.5] expected: [0. 1. 0.] produced: [1.4664079557953606e-05, 0.376670400777417, 0.21871270781912341]
Sample: [5.5 2.4 3.8 1.1] expected: [0. 1. 0.] produced: [0.00026421898515814477, 0.34277545558426414, 0.0033565207681857277]
Sample: [7.9 3.8 6.4 2. ] expected: [0. 0. 1.] produced: [2.5012118718384197e-06, 0.410557233941309, 0.8056933819855051]
Sample: [5.7 2.9 4.2 1.3] expected: [0. 1. 0.] produced: [9.174260636621881e-05, 0.3578744976130092, 0.01672504422092612]
Sample: [7.  3.2 4.7 1.4] expected: [0. 1. 0.] produced: [0.0001797009264919115, 0.3532511638290675, 0.006055025986744134]
Sample: [7.1 3.  5.9 2.1] expected: [0. 0. 1.] produced: [1.3539838391698507e-06, 0.4254730066065252, 0.91402642234792]
Sample: [4.8 3.1 1.6 0.2] expected: [1. 0. 0.] produced: [0.9802113122286402, 0.20930140361459296, 3.0081433569124335e-11]
Sample: [5.8 4.  1.2 0.2] expected: [1. 0. 0.] produced: [0.9819337353811148, 0.20782908862655314, 2.608216207838798e-11]
Sample: [5.3 3.7 1.5 0.2] expected: [1. 0. 0.] produced: [0.981643642715056, 0.2074238757736693, 2.6743373628726737e-11]
Sample: [5.5 2.4 3.7 1. ] expected: [0. 1. 0.] produced: [0.0002836416461991622, 0.34588648297471236, 0.003028224316589916]
Sample: [6.2 2.9 4.3 1.3] expected: [0. 1. 0.] produced: [5.703537392116039e-05, 0.37167302841108363, 0.03399077535374807]
Sample: [6.7 3.3 5.7 2.1] expected: [0. 0. 1.] produced: [1.4472661292662302e-06, 0.42803614315548894, 0.9057739152287393]
Sample: [6.1 2.6 5.6 1.4] expected: [0. 0. 1.] produced: [1.5380422248859272e-06, 0.42335219470122687, 0.8976515356550667]
Sample: [6.7 3.1 5.6 2.4] expected: [0. 0. 1.] produced: [1.2836710848778415e-06, 0.42217461442248033, 0.9204783130836204]
Sample: [7.7 3.  6.1 2.3] expected: [0. 0. 1.] produced: [1.268571755798403e-06, 0.41858906335333285, 0.921853547225429]
Sample: [4.7 3.2 1.6 0.2] expected: [1. 0. 0.] produced: [0.9802027932199119, 0.20566109774643432, 3.0192650415514787e-11]
Sample: [6.3 2.9 5.6 1.8] expected: [0. 0. 1.] produced: [1.4302234047226958e-06, 0.41239120313243355, 0.9076729772557458]
Sample: [5.2 2.7 3.9 1.4] expected: [0. 1. 0.] produced: [1.24434284872075e-05, 0.37917325135744084, 0.26562307469542684]
Sample: [6.1 2.8 4.  1.3] expected: [0. 1. 0.] produced: [0.00040717406964394074, 0.3375326104959006, 0.0017420310723381693]
Sample: [4.8 3.  1.4 0.3] expected: [1. 0. 0.] produced: [0.9801912760784073, 0.20831362056814223, 3.009631470284846e-11]
Sample: [7.7 2.8 6.7 2. ] expected: [0. 0. 1.] produced: [1.2550936387787723e-06, 0.41964166939091463, 0.9226428092420886]
Sample: [4.6 3.4 1.4 0.3] expected: [1. 0. 0.] produced: [0.9810227990406409, 0.20565379953022567, 2.816875972102649e-11]
Sample: [4.9 2.5 4.5 1.7] expected: [0. 0. 1.] produced: [1.7688155465127137e-06, 0.41030631495236547, 0.8760831049638026]
Sample: [6.  2.2 4.  1. ] expected: [0. 1. 0.] produced: [6.684854423832241e-05, 0.3575564537712895, 0.026891013588708625]
Sample: [5.5 4.2 1.4 0.2] expected: [1. 0. 0.] produced: [0.9819244301584392, 0.20541444639771203, 2.6115667210676503e-11]
Sample: [6.5 3.  5.2 2. ] expected: [0. 0. 1.] produced: [1.7167228411842993e-06, 0.4112800479433564, 0.8811497376694386]
Sample: [5.9 3.2 4.8 1.8] expected: [0. 1. 0.] produced: [3.3720904860021064e-06, 0.39838166628707966, 0.7259666083910803]
Sample: [6.9 3.1 5.1 2.3] expected: [0. 0. 1.] produced: [2.395308862215658e-06, 0.40814894414053204, 0.8139033814015075]
Sample: [4.4 2.9 1.4 0.2] expected: [1. 0. 0.] produced: [0.9804457796073536, 0.20476569194085162, 2.9286597565338725e-11]
Sample: [6.3 2.5 4.9 1.5] expected: [0. 1. 0.] produced: [4.275963841610675e-06, 0.39586060417704955, 0.644510771665537]
Sample: [6.5 2.8 4.6 1.5] expected: [0. 1. 0.] produced: [0.0009469776093980417, 0.328444273788014, 0.00046620103198534554]
Sample: [5.5 2.3 4.  1.3] expected: [0. 1. 0.] produced: [0.00020728960088049137, 0.35227815662094, 0.004723758903327814]
Sample: [5.6 3.  4.5 1.5] expected: [0. 1. 0.] produced: [0.00023793383877732655, 0.35465519838623194, 0.003830331783578699]
Epoch 9000 RMSE =  0.281525738434023
Epoch 9100 RMSE =  0.28062599001608546
Epoch 9200 RMSE =  0.2806772436200928
Epoch 9300 RMSE =  0.28008144518066586
Epoch 9400 RMSE =  0.2815352271133665
Epoch 9500 RMSE =  0.2826285329932771
Epoch 9600 RMSE =  0.2806312542037913
Epoch 9700 RMSE =  0.27760669993628473
Epoch 9800 RMSE =  0.2808298397756116
Epoch 9900 RMSE =  0.28213030229006936
Sample: [4.3 3.  1.1 0.1] expected: [1. 0. 0.] produced: [0.9822994342072672, 0.20743391621506574, 7.484909937626913e-12]
Sample: [6.3 2.7 4.9 1.8] expected: [0. 0. 1.] produced: [2.298075994847922e-06, 0.4288671630325297, 0.7781145732589629]
Sample: [5.4 3.9 1.3 0.4] expected: [1. 0. 0.] produced: [0.98287976665603, 0.20477872568627595, 7.095892811146638e-12]
Sample: [6.9 3.1 5.1 2.3] expected: [0. 0. 1.] produced: [1.6690195985925008e-06, 0.4291516285573551, 0.8539640062950642]
Sample: [6.4 3.2 4.5 1.5] expected: [0. 1. 0.] produced: [0.00010714952208565821, 0.36415165354791384, 0.008064059038403135]
Sample: [6.5 3.  5.8 2.2] expected: [0. 0. 1.] produced: [1.1903220499631723e-06, 0.43540299343656264, 0.9091391019412728]
Sample: [7.2 3.  5.8 1.6] expected: [0. 0. 1.] produced: [2.0152926546465875e-06, 0.4235322282268829, 0.8133197895703725]
Sample: [7.7 2.8 6.7 2. ] expected: [0. 0. 1.] produced: [1.122528012614758e-06, 0.4286105821932799, 0.9169192356081576]
Sample: [4.8 3.4 1.6 0.2] expected: [1. 0. 0.] produced: [0.98207048101794, 0.20029323694268206, 7.694534585353131e-12]
Sample: [6.7 3.1 5.6 2.4] expected: [0. 0. 1.] produced: [1.1926555360406204e-06, 0.42307856129449184, 0.9094041027802815]
Sample: [6.4 2.9 4.3 1.3] expected: [0. 1. 0.] produced: [0.00011575794420376381, 0.35349439531743826, 0.007191454732666522]
Sample: [4.5 2.3 1.3 0.3] expected: [1. 0. 0.] produced: [0.9745338854802506, 0.20405458334333687, 1.3617228721048879e-11]
Sample: [6.8 3.  5.5 2.1] expected: [0. 0. 1.] produced: [1.3100102913106522e-06, 0.4220237707585915, 0.8965235005059021]
Sample: [5.2 3.5 1.5 0.2] expected: [1. 0. 0.] produced: [0.9824775307777721, 0.1980884647783158, 7.426194836649605e-12]
Sample: [6.9 3.1 4.9 1.5] expected: [0. 1. 0.] produced: [1.059420127148843e-05, 0.3869748444000685, 0.24144619216256166]
Sample: [6.1 3.  4.6 1.4] expected: [0. 1. 0.] produced: [4.131593849883724e-05, 0.37209743818630486, 0.035473295199014904]
Sample: [4.6 3.1 1.5 0.2] expected: [1. 0. 0.] produced: [0.9816814951528318, 0.20272567366323657, 7.949883584582616e-12]
Sample: [5.2 2.7 3.9 1.4] expected: [0. 1. 0.] produced: [3.5742336379368984e-05, 0.37794194936945374, 0.0442033602893384]
Sample: [6.2 2.9 4.3 1.3] expected: [0. 1. 0.] produced: [0.0001772483968425613, 0.35940319928427616, 0.0036617245505553937]
Sample: [5.8 2.7 3.9 1.2] expected: [0. 1. 0.] produced: [0.0003187961624589302, 0.3552759134647596, 0.0014503135739934866]
Sample: [4.9 2.5 4.5 1.7] expected: [0. 0. 1.] produced: [1.5925431367673428e-06, 0.4385440082562841, 0.8635645614516685]
Sample: [7.2 3.6 6.1 2.5] expected: [0. 0. 1.] produced: [1.211640571416712e-06, 0.4387943959189521, 0.9071873954310266]
Sample: [6.1 2.8 4.7 1.2] expected: [0. 1. 0.] produced: [1.4262820088045214e-05, 0.39781853200852096, 0.16547951537620803]
Sample: [5.1 3.8 1.5 0.3] expected: [1. 0. 0.] produced: [0.9827253151004827, 0.20761201853106298, 7.224701564147556e-12]
Sample: [6.7 3.3 5.7 2.5] expected: [0. 0. 1.] produced: [1.2207047182257226e-06, 0.438925977125672, 0.906016751292335]
Sample: [6.6 3.  4.4 1.4] expected: [0. 1. 0.] produced: [0.00016052895531732864, 0.362760739307855, 0.004286048493845009]
Sample: [5.4 3.9 1.7 0.4] expected: [1. 0. 0.] produced: [0.9825874732570469, 0.20777577382826792, 7.32187315784752e-12]
Sample: [6.5 3.  5.5 1.8] expected: [0. 0. 1.] produced: [1.6265003067231295e-06, 0.43464428443936237, 0.8597673453756727]
Sample: [5.1 3.3 1.7 0.5] expected: [1. 0. 0.] produced: [0.9805984546407872, 0.2065808423381222, 8.733628408412623e-12]
Sample: [4.4 2.9 1.4 0.2] expected: [1. 0. 0.] produced: [0.9807635010426736, 0.20595865597168322, 8.633424833433209e-12]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.] produced: [0.9820008540599533, 0.20471130822518183, 7.744511212268103e-12]
Sample: [5.1 3.5 1.4 0.3] expected: [1. 0. 0.] produced: [0.9824636398806444, 0.20389914007936022, 7.423465559413644e-12]
Sample: [6.9 3.2 5.7 2.3] expected: [0. 0. 1.] produced: [1.2627532904656934e-06, 0.4313210259787651, 0.9017082978864238]
Sample: [5.6 2.8 4.9 2. ] expected: [0. 0. 1.] produced: [1.4084497112959056e-06, 0.4258138191220376, 0.8854406702341127]
Sample: [6.3 3.4 5.6 2.4] expected: [0. 0. 1.] produced: [1.247787274617856e-06, 0.4238507180990282, 0.9036307150203114]
Sample: [5.  3.6 1.4 0.2] expected: [1. 0. 0.] produced: [0.9826619983715693, 0.19821549945208455, 7.310600565791125e-12]
Sample: [5.7 3.  4.2 1.2] expected: [0. 1. 0.] produced: [0.00023556393477590178, 0.34408608624218895, 0.002356589711644532]
Sample: [6.4 3.1 5.5 1.8] expected: [0. 0. 1.] produced: [1.5906465480554279e-06, 0.42054721320403426, 0.8647858878972109]
Sample: [5.1 3.5 1.4 0.2] expected: [1. 0. 0.] produced: [0.9825430258476087, 0.19850626603125285, 7.40319794187247e-12]
Sample: [7.7 3.  6.1 2.3] expected: [0. 0. 1.] produced: [1.1561408026961438e-06, 0.42082862304764, 0.9139274221511964]
Sample: [7.3 2.9 6.3 1.8] expected: [0. 0. 1.] produced: [1.21268752083213e-06, 0.41639052803529353, 0.9078830815629031]
Sample: [5.  3.4 1.5 0.2] expected: [1. 0. 0.] produced: [0.9822525669781932, 0.19494576156204976, 7.61448536282428e-12]
Sample: [5.  2.  3.5 1. ] expected: [0. 1. 0.] produced: [2.4435681468547498e-05, 0.3688385618199506, 0.07871415917804615]
Sample: [4.9 3.  1.4 0.2] expected: [1. 0. 0.] produced: [0.9814902592667275, 0.1971976784523258, 8.148287322218755e-12]
Sample: [6.3 2.5 4.9 1.5] expected: [0. 1. 0.] produced: [2.021134141337911e-06, 0.4087260941853526, 0.8146751230085031]
Sample: [5.2 3.4 1.4 0.2] expected: [1. 0. 0.] produced: [0.9826331489450395, 0.19833991530536296, 7.259087756599605e-12]
Sample: [6.4 2.8 5.6 2.1] expected: [0. 0. 1.] produced: [1.2445379651885852e-06, 0.4202339313359749, 0.9028569016186193]
Sample: [6.1 3.  4.9 1.8] expected: [0. 0. 1.] produced: [2.9611884140957785e-06, 0.40375739706783076, 0.7026776435426213]
Sample: [6.6 2.9 4.6 1.3] expected: [0. 1. 0.] produced: [3.43420596450874e-05, 0.36530988279039717, 0.0471367599493829]
Sample: [6.7 3.  5.  1.7] expected: [0. 1. 0.] produced: [2.998718359547717e-06, 0.4050805924895508, 0.7003237987089395]
Sample: [6.7 3.  5.2 2.3] expected: [0. 0. 1.] produced: [1.7420855297637784e-06, 0.4181564775205233, 0.8437262483858748]
Sample: [4.8 3.1 1.6 0.2] expected: [1. 0. 0.] produced: [0.9819865568480092, 0.19803256619632373, 7.658664651948433e-12]
Sample: [4.8 3.  1.4 0.3] expected: [1. 0. 0.] produced: [0.9818484361979913, 0.19761213407646708, 7.756351959946469e-12]
Sample: [6.4 2.8 5.6 2.2] expected: [0. 0. 1.] produced: [1.28987612948197e-06, 0.4174446741718631, 0.8970154177873445]
Sample: [6.8 2.8 4.8 1.4] expected: [0. 1. 0.] produced: [5.426595406985178e-05, 0.35995377412369645, 0.023066275716055826]
Sample: [5.7 2.9 4.2 1.3] expected: [0. 1. 0.] produced: [0.00037209062762461845, 0.33761492999797427, 0.0011232839136208132]
Sample: [7.9 3.8 6.4 2. ] expected: [0. 0. 1.] produced: [2.823944502348073e-06, 0.41185419032339765, 0.7164639205497836]
Sample: [6.5 3.2 5.1 2. ] expected: [0. 0. 1.] produced: [2.1829516266205717e-06, 0.412167940753205, 0.7928845194148647]
Sample: [4.4 3.  1.3 0.2] expected: [1. 0. 0.] produced: [0.9813986582054497, 0.19727832365469075, 8.161305401033528e-12]
Sample: [6.1 2.6 5.6 1.4] expected: [0. 0. 1.] produced: [1.486556903732548e-06, 0.41353834832183234, 0.8759864402851518]
Sample: [5.5 3.5 1.3 0.2] expected: [1. 0. 0.] produced: [0.9827606123485537, 0.19444248143346707, 7.213924031604423e-12]
Sample: [5.8 4.  1.2 0.2] expected: [1. 0. 0.] produced: [0.9830357542802148, 0.1938090837681266, 7.0282408538297085e-12]
Sample: [5.9 3.  4.2 1.5] expected: [0. 1. 0.] produced: [3.0095241002104574e-05, 0.3653089557209726, 0.057362409566228706]
Sample: [7.1 3.  5.9 2.1] expected: [0. 0. 1.] produced: [1.2216424705059357e-06, 0.4163542398080182, 0.9061031901790446]
Sample: [7.  3.2 4.7 1.4] expected: [0. 1. 0.] produced: [9.067423959809721e-05, 0.3511802802776343, 0.010532430351828416]
Sample: [5.7 2.6 3.5 1. ] expected: [0. 1. 0.] produced: [0.003553754863657685, 0.3060705117980401, 3.2021958831028833e-05]
Sample: [5.8 2.7 5.1 1.9] expected: [0. 0. 1.] produced: [1.2664228645100555e-06, 0.4216390898469723, 0.9012521669902575]
Sample: [5.2 4.1 1.5 0.1] expected: [1. 0. 0.] produced: [0.9829401459747049, 0.19713837732331196, 7.103065717835219e-12]
Sample: [5.7 3.8 1.7 0.3] expected: [1. 0. 0.] produced: [0.9825476096599137, 0.19686589368821253, 7.366677491773184e-12]
Sample: [5.  3.3 1.4 0.2] expected: [1. 0. 0.] produced: [0.9821070867961096, 0.19662564037846694, 7.67473127247782e-12]
Sample: [5.5 2.4 3.7 1. ] expected: [0. 1. 0.] produced: [0.00010924204166787836, 0.3516184907155166, 0.007876990680060288]
Sample: [5.4 3.4 1.7 0.2] expected: [1. 0. 0.] produced: [0.9819316105466838, 0.19857110161115624, 7.792100636374985e-12]
Sample: [6.  2.9 4.5 1.5] expected: [0. 1. 0.] produced: [4.655990759655954e-06, 0.4005199516075835, 0.5384015575367437]
Sample: [5.9 3.2 4.8 1.8] expected: [0. 1. 0.] produced: [6.842649598454354e-06, 0.3997420438989887, 0.38371456798014986]
Sample: [5.  2.3 3.3 1. ] expected: [0. 1. 0.] produced: [0.04591928487278739, 0.28200066415164476, 5.098426903167893e-07]
Sample: [5.5 2.6 4.4 1.2] expected: [0. 1. 0.] produced: [0.00020267389425219828, 0.3590475436370707, 0.0028906718439803154]
Sample: [5.3 3.7 1.5 0.2] expected: [1. 0. 0.] produced: [0.9829667081069178, 0.20685440130229626, 6.927370835204565e-12]
Sample: [5.5 2.3 4.  1.3] expected: [0. 1. 0.] produced: [7.63236150522717e-05, 0.37669937108670565, 0.013395801188046924]
Sample: [6.3 2.8 5.1 1.5] expected: [0. 0. 1.] produced: [1.0906924462442355e-05, 0.4102765173179994, 0.22725748738621324]
Sample: [4.7 3.2 1.6 0.2] expected: [1. 0. 0.] produced: [0.9808229833840696, 0.20831789211822418, 8.506056215669728e-12]
Sample: [4.6 3.4 1.4 0.3] expected: [1. 0. 0.] produced: [0.9815889047790224, 0.20732803185222645, 7.967626109709182e-12]
Sample: [6.2 3.4 5.4 2.3] expected: [0. 0. 1.] produced: [1.2271023058070694e-06, 0.43855973901727097, 0.904579413641851]
Sample: [5.5 4.2 1.4 0.2] expected: [1. 0. 0.] produced: [0.9829629844310889, 0.2041898249008033, 7.02994661061309e-12]
Sample: [7.6 3.  6.6 2.1] expected: [0. 0. 1.] produced: [1.1008691050038544e-06, 0.4354384861888804, 0.9184958127335934]
Sample: [6.7 3.3 5.7 2.1] expected: [0. 0. 1.] produced: [1.2379378692597573e-06, 0.42972217878882646, 0.9035633623663671]
Sample: [5.6 2.9 3.6 1.3] expected: [0. 1. 0.] produced: [0.00021610671554113845, 0.35063153078361164, 0.002667717277747522]
Sample: [6.4 3.2 5.3 2.3] expected: [0. 0. 1.] produced: [1.194104329481315e-06, 0.4313036882723212, 0.9085215095773457]
Sample: [6.3 3.3 4.7 1.6] expected: [0. 1. 0.] produced: [5.044330882231198e-06, 0.4058942957530411, 0.5046548922133283]
Sample: [5.  3.2 1.2 0.2] expected: [1. 0. 0.] produced: [0.982574045688549, 0.20345250066517684, 7.2220416952376946e-12]
Sample: [6.5 2.8 4.6 1.5] expected: [0. 1. 0.] produced: [1.5945917445728584e-05, 0.3928880932834993, 0.13959600883031648]
Sample: [6.  2.2 5.  1.5] expected: [0. 0. 1.] produced: [1.7650093904602193e-06, 0.4308111097437526, 0.8400343905184774]
Sample: [6.7 3.1 4.7 1.5] expected: [0. 1. 0.] produced: [8.501718326772684e-05, 0.3695663318423208, 0.011387485970262604]
Sample: [6.  2.2 4.  1. ] expected: [0. 1. 0.] produced: [0.00011667481457660806, 0.3693711380272625, 0.006935028249987844]
Sample: [5.1 3.4 1.5 0.2] expected: [1. 0. 0.] produced: [0.9825023728566852, 0.2083992519915853, 7.269556801805906e-12]
Sample: [5.1 2.5 3.  1.1] expected: [0. 1. 0.] produced: [0.018827940168165552, 0.3017807946153714, 2.192761067451315e-06]
Sample: [4.4 3.2 1.3 0.2] expected: [1. 0. 0.] produced: [0.9820449051232398, 0.21055622670393395, 7.591242962837085e-12]
Sample: [7.2 3.2 6.  1.8] expected: [0. 0. 1.] produced: [1.6460024003514982e-06, 0.44060235138171383, 0.8547054065027099]
Sample: [6.1 2.8 4.  1.3] expected: [0. 1. 0.] produced: [0.00023656009962417002, 0.36240596255697555, 0.0022844715585854055]
Sample: [5.4 3.7 1.5 0.2] expected: [1. 0. 0.] produced: [0.9827624562687329, 0.21016692674827148, 7.107262146080717e-12]
Sample: [5.6 3.  4.5 1.5] expected: [0. 1. 0.] produced: [7.650498953811047e-06, 0.41705723644194276, 0.34204322221642436]
Sample: [6.3 3.3 6.  2.5] expected: [0. 0. 1.] produced: [1.1944309703711667e-06, 0.4506743752515251, 0.9064097698634855]
Sample: [6.5 3.  5.2 2. ] expected: [0. 0. 1.] produced: [2.267178559717501e-06, 0.4366337084428505, 0.778805635886707]
Sample: [5.5 2.4 3.8 1.1] expected: [0. 1. 0.] produced: [0.0003053467644386014, 0.35965907765521543, 0.0015211476985677401]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.] produced: [0.982239920122802, 0.21107906247816413, 7.440298607817692e-12]
Sample: [6.3 2.9 5.6 1.8] expected: [0. 0. 1.] produced: [1.5541453758429107e-06, 0.44256637410048044, 0.8654440510479372]
Epoch 10000 RMSE =  0.28225754371051576
Final Epoch RMSE =  0.28225754371051576
Sample: [5.6 2.5 3.9 1.1] expected: [0. 1. 0.] produced: [0.00027203105876081875, 0.36134606205399483, 0.0018293624770491393]
Sample: [5.9 3.  5.1 1.8] expected: [0. 0. 1.] produced: [2.0566438981673383e-06, 0.4391108203501707, 0.8054487790359287]
Sample: [4.9 3.1 1.5 0.1] expected: [1. 0. 0.] produced: [0.9820899870814929, 0.2094204847705445, 7.576971121577842e-12]
Sample: [6.3 2.5 5.  1.9] expected: [0. 0. 1.] produced: [1.4241463948422362e-06, 0.4400103239348716, 0.8814486381930486]
Sample: [4.8 3.  1.4 0.1] expected: [1. 0. 0.] produced: [0.9819892995598128, 0.20712061392082923, 7.658644526259726e-12]
Sample: [5.7 2.8 4.5 1.3] expected: [0. 1. 0.] produced: [1.4196988337549457e-05, 0.4004637880662485, 0.16420041777819877]
Sample: [6.2 2.8 4.8 1.8] expected: [0. 0. 1.] produced: [2.6918241323298694e-06, 0.4304640694917558, 0.73074655842117]
Sample: [5.1 3.8 1.9 0.4] expected: [1. 0. 0.] produced: [0.9821035607746608, 0.20711648433723534, 7.594660639092601e-12]
Sample: [5.1 3.7 1.5 0.4] expected: [1. 0. 0.] produced: [0.9824818651882465, 0.20633905245678472, 7.339867733005175e-12]
Sample: [5.1 3.8 1.6 0.2] expected: [1. 0. 0.] produced: [0.9827440621249881, 0.20562998661620774, 7.163779633351854e-12]
Sample: [6.7 2.5 5.8 1.8] expected: [0. 0. 1.] produced: [1.2083930027587403e-06, 0.43645410569885335, 0.9065783913649018]
Sample: [5.7 2.5 5.  2. ] expected: [0. 0. 1.] produced: [1.2461629441716824e-06, 0.43204887369103134, 0.9024798436397974]
Sample: [5.  3.5 1.6 0.6] expected: [1. 0. 0.] produced: [0.9812198731350635, 0.20248102979496402, 8.224858468801327e-12]
Sample: [6.1 2.9 4.7 1.4] expected: [0. 1. 0.] produced: [9.58347771640187e-06, 0.39692505107678655, 0.26913210229741374]
Sample: [4.8 3.4 1.9 0.2] expected: [1. 0. 0.] produced: [0.9820305913946822, 0.20384521361951286, 7.621111761562355e-12]
Sample: [7.4 2.8 6.1 1.9] expected: [0. 0. 1.] produced: [1.4194139355614134e-06, 0.4296555865916031, 0.8821612675719303]
Sample: [5.8 2.6 4.  1.2] expected: [0. 1. 0.] produced: [0.0004226965526308797, 0.34292139804161004, 0.0009184583022185615]
Sample: [5.6 3.  4.1 1.3] expected: [0. 1. 0.] produced: [0.0006693953630908619, 0.3405750003148578, 0.00044405484203337937]
Sample: [5.8 2.8 5.1 2.4] expected: [0. 0. 1.] produced: [1.2062620155203967e-06, 0.4379091719238855, 0.9065436802318598]
Sample: [6.  3.4 4.5 1.6] expected: [0. 1. 0.] produced: [0.00011318147608466415, 0.36656132330099456, 0.007339610210623872]
Sample: [5.7 4.4 1.5 0.4] expected: [1. 0. 0.] produced: [0.9830353978474491, 0.20646212865925312, 6.957434976081957e-12]
Sample: [6.  2.7 5.1 1.6] expected: [0. 1. 0.] produced: [2.010946186292846e-06, 0.4302492555289265, 0.8123458111682492]
Sample: [4.9 2.4 3.3 1. ] expected: [0. 1. 0.] produced: [0.007190780164467186, 0.3158110479001829, 1.0158281490375075e-05]
Sample: [7.7 3.8 6.7 2.2] expected: [0. 0. 1.] produced: [1.562889451698925e-06, 0.4436999197159921, 0.8636112340473269]
Sample: [5.8 2.7 4.1 1. ] expected: [0. 1. 0.] produced: [0.0023194268669760316, 0.33202331719799216, 6.13320723365244e-05]
Sample: [6.4 2.7 5.3 1.9] expected: [0. 0. 1.] produced: [1.5049010711637937e-06, 0.44496252965056177, 0.8707671851942577]
Sample: [7.7 2.6 6.9 2.3] expected: [0. 0. 1.] produced: [1.0870923658018913e-06, 0.4459602323956362, 0.9186403223834074]
Sample: [5.4 3.  4.5 1.5] expected: [0. 1. 0.] produced: [1.1912018271465306e-05, 0.40557284603748295, 0.20430807613421204]
Sample: [5.  3.5 1.3 0.3] expected: [1. 0. 0.] produced: [0.982743574296977, 0.21020303121138836, 7.071106346043897e-12]
Sample: [5.  3.4 1.6 0.4] expected: [1. 0. 0.] produced: [0.9821711967394621, 0.20996687428182023, 7.4510936468188e-12]
Sample: [6.9 3.1 5.4 2.1] expected: [0. 0. 1.] produced: [2.108074982190663e-06, 0.43502759662758844, 0.7979288825679469]
Sample: [5.7 2.8 4.1 1.3] expected: [0. 1. 0.] produced: [0.0002530948751224557, 0.3601193708544099, 0.0020433285984455514]
Sample: [6.8 3.2 5.9 2.3] expected: [0. 0. 1.] produced: [1.2774520257591818e-06, 0.4437284784737705, 0.8975236677745609]
Sample: [6.3 2.3 4.4 1.3] expected: [0. 1. 0.] produced: [1.03743385146403e-05, 0.40791123102873833, 0.24239210172806505]
Sample: [5.8 2.7 5.1 1.9] expected: [0. 0. 1.] produced: [1.791347480356556e-06, 0.4395394467150999, 0.8362520892011066]
Sample: [5.4 3.4 1.5 0.4] expected: [1. 0. 0.] produced: [0.9825544913030966, 0.20859113523781495, 7.207932102540239e-12]
Sample: [5.6 2.7 4.2 1.3] expected: [0. 1. 0.] produced: [0.00013974869251352087, 0.3696803164541245, 0.005201615575045796]
Sample: [6.2 2.2 4.5 1.5] expected: [0. 1. 0.] produced: [4.0712230549879175e-06, 0.42705313990654425, 0.5832562363723113]
Sample: [4.6 3.2 1.4 0.2] expected: [1. 0. 0.] produced: [0.9826923193694738, 0.21272302140989954, 7.025673858073766e-12]
Sample: [6.  3.  4.8 1.8] expected: [0. 0. 1.] produced: [3.431954715730234e-05, 0.39854443869497436, 0.045079409954008054]
Sample: [4.6 3.6 1.  0.2] expected: [1. 0. 0.] produced: [0.982976617183003, 0.21036084094999843, 6.871022681610496e-12]
Sample: [5.5 2.5 4.  1.3] expected: [0. 1. 0.] produced: [0.00018111977912450353, 0.37005986767151106, 0.0034121330256050037]
Sample: [6.7 3.1 4.4 1.4] expected: [0. 1. 0.] produced: [0.007811824227914071, 0.32057805400942513, 8.81881979130548e-06]
Sample: [5.  3.  1.6 0.2] expected: [1. 0. 0.] produced: [0.9819469679383641, 0.21533966920734163, 7.544635230269887e-12]
Sample: [4.7 3.2 1.3 0.2] expected: [1. 0. 0.] produced: [0.982515164416769, 0.21438560021209774, 7.1716875194523205e-12]
Final Epoch RMSE =  0.29873453114831877
Sample: [0.35] expected: [0.34289781] produced: [0.8072005772532275]
Sample: [0.23] expected: [0.22797752] produced: [0.7996813350029295]
Sample: [0.48] expected: [0.46177918] produced: [0.8133451651527559]
Sample: [0.68] expected: [0.62879302] produced: [0.8236063578182216]
Sample: [1.05] expected: [0.86742323] produced: [0.8412209240618593]
Sample: [0.57] expected: [0.53963205] produced: [0.8176241308665952]
Sample: [1.22] expected: [0.93909936] produced: [0.8482871341268394]
Sample: [0.63] expected: [0.58914476] produced: [0.8205836772721719]
Sample: [0.94] expected: [0.8075581] produced: [0.8357030827459507]
Sample: [1.5] expected: [0.99749499] produced: [0.8591078470715681]
Sample: [0.67] expected: [0.62098599] produced: [0.8225246850507228]
Sample: [0.11] expected: [0.1097783] produced: [0.7906791138184089]
Sample: [0.33] expected: [0.32404303] produced: [0.8029392098861111]
Sample: [0.18] expected: [0.17902957] produced: [0.7936064062984225]
Sample: [0.43] expected: [0.4168708] produced: [0.8073444643718823]
Epoch 0 RMSE =  0.3766035052710372
Epoch 100 RMSE =  0.2653811659188588
Epoch 200 RMSE =  0.2646438626936347
Epoch 300 RMSE =  0.2640489363257199
Epoch 400 RMSE =  0.26342671569935344
Epoch 500 RMSE =  0.26265537712686604
Epoch 600 RMSE =  0.2615490761515839
Epoch 700 RMSE =  0.2598035527391258
Epoch 800 RMSE =  0.2567928260165425
Epoch 900 RMSE =  0.251280790723245
Sample: [0.68] expected: [0.62879302] produced: [0.5681668131217642]
Sample: [0.23] expected: [0.22797752] produced: [0.529429280605207]
Sample: [0.43] expected: [0.4168708] produced: [0.5465517458543119]
Sample: [1.5] expected: [0.99749499] produced: [0.618737804793789]
Sample: [1.05] expected: [0.86742323] produced: [0.5954010507699694]
Sample: [0.94] expected: [0.8075581] produced: [0.5891676983680204]
Sample: [0.57] expected: [0.53963205] produced: [0.5614926276229825]
Sample: [0.48] expected: [0.46177918] produced: [0.5537109650147568]
Sample: [0.18] expected: [0.17902957] produced: [0.5261981990836918]
Sample: [0.11] expected: [0.1097783] produced: [0.5187143029869633]
Sample: [0.35] expected: [0.34289781] produced: [0.5397643666446531]
Sample: [1.22] expected: [0.93909936] produced: [0.6043698605721831]
Sample: [0.67] expected: [0.62098599] produced: [0.5677679507462171]
Sample: [0.33] expected: [0.32404303] produced: [0.538760511178236]
Sample: [0.63] expected: [0.58914476] produced: [0.5640249811764878]
Epoch 1000 RMSE =  0.24132472800514992
Epoch 1100 RMSE =  0.22562717881125824
Epoch 1200 RMSE =  0.2061200622263878
Epoch 1300 RMSE =  0.18640871496785175
Epoch 1400 RMSE =  0.1686219123632159
Epoch 1500 RMSE =  0.15326274461543818
Epoch 1600 RMSE =  0.1401397347529191
Epoch 1700 RMSE =  0.1289034728697293
Epoch 1800 RMSE =  0.11922560764816614
Epoch 1900 RMSE =  0.11081954987151463
Sample: [0.48] expected: [0.46177918] produced: [0.5010609360430435]
Sample: [0.68] expected: [0.62879302] produced: [0.6100105484030048]
Sample: [0.57] expected: [0.53963205] produced: [0.5533038409367053]
Sample: [0.94] expected: [0.8075581] produced: [0.7103316523138109]
Sample: [0.67] expected: [0.62098599] produced: [0.6055931350009586]
Sample: [0.33] expected: [0.32404303] produced: [0.4062723559860571]
Sample: [0.18] expected: [0.17902957] produced: [0.3100679300089904]
Sample: [1.5] expected: [0.99749499] produced: [0.8106809163702108]
Sample: [0.63] expected: [0.58914476] produced: [0.5856032898745309]
Sample: [0.35] expected: [0.34289781] produced: [0.41915100024997504]
Sample: [0.23] expected: [0.22797752] produced: [0.3413394872426219]
Sample: [0.11] expected: [0.1097783] produced: [0.2679390102755524]
Sample: [0.43] expected: [0.4168708] produced: [0.4694082455701097]
Sample: [1.05] expected: [0.86742323] produced: [0.7393566356319096]
Sample: [1.22] expected: [0.93909936] produced: [0.7744284225315471]
Epoch 2000 RMSE =  0.1034651230176702
Epoch 2100 RMSE =  0.09698544459718815
Epoch 2200 RMSE =  0.09124057753862014
Epoch 2300 RMSE =  0.08611917063797095
Epoch 2400 RMSE =  0.08153068912067615
Epoch 2500 RMSE =  0.07740130190633905
Epoch 2600 RMSE =  0.07367251217793552
Epoch 2700 RMSE =  0.07029590622328075
Epoch 2800 RMSE =  0.06722273441195552
Epoch 2900 RMSE =  0.06442343000036292
Sample: [0.48] expected: [0.46177918] produced: [0.48652594822380607]
Sample: [0.35] expected: [0.34289781] produced: [0.3766049392084891]
Sample: [0.63] expected: [0.58914476] produced: [0.5993467269050933]
Sample: [0.94] expected: [0.8075581] produced: [0.7555418548805075]
Sample: [0.67] expected: [0.62098599] produced: [0.6256542193747905]
Sample: [0.33] expected: [0.32404303] produced: [0.3596058354555179]
Sample: [1.05] expected: [0.86742323] produced: [0.7891608979693506]
Sample: [0.68] expected: [0.62879302] produced: [0.6320199875917107]
Sample: [0.11] expected: [0.1097783] produced: [0.19276921285867363]
Sample: [1.5] expected: [0.99749499] produced: [0.8616584526113438]
Sample: [0.23] expected: [0.22797752] produced: [0.2779145748749651]
Sample: [1.22] expected: [0.93909936] produced: [0.8262462670990901]
Sample: [0.18] expected: [0.17902957] produced: [0.2403707689652055]
Sample: [0.57] expected: [0.53963205] produced: [0.5572045807848923]
Sample: [0.43] expected: [0.4168708] produced: [0.4449732319920241]
Epoch 3000 RMSE =  0.061865510749939435
Epoch 3100 RMSE =  0.05952116856277867
Epoch 3200 RMSE =  0.05737052258736119
Epoch 3300 RMSE =  0.05539218845715132
Epoch 3400 RMSE =  0.05356976480388555
Epoch 3500 RMSE =  0.05188776966377005
Epoch 3600 RMSE =  0.05033343708761887
Epoch 3700 RMSE =  0.048893970859648625
Epoch 3800 RMSE =  0.04756109396490051
Epoch 3900 RMSE =  0.046323219827852175
Sample: [0.48] expected: [0.46177918] produced: [0.47808850713079054]
Sample: [0.18] expected: [0.17902957] produced: [0.21195336982663696]
Sample: [1.22] expected: [0.93909936] produced: [0.850453138461209]
Sample: [0.94] expected: [0.8075581] produced: [0.777001703279154]
Sample: [0.67] expected: [0.62098599] produced: [0.6344656386260268]
Sample: [0.35] expected: [0.34289781] produced: [0.3565915283809702]
Sample: [0.68] expected: [0.62879302] produced: [0.641320484975542]
Sample: [0.33] expected: [0.32404303] produced: [0.3380056608042734]
Sample: [0.11] expected: [0.1097783] produced: [0.16448480479978983]
Sample: [0.43] expected: [0.4168708] produced: [0.4314963601521218]
Sample: [0.57] expected: [0.53963205] produced: [0.5569845087666591]
Sample: [0.63] expected: [0.58914476] produced: [0.6047102201232268]
Sample: [1.5] expected: [0.99749499] produced: [0.8859388605072053]
Sample: [0.23] expected: [0.22797752] produced: [0.25070838973746984]
Sample: [1.05] expected: [0.86742323] produced: [0.812340713829732]
Epoch 4000 RMSE =  0.04517268499189931
Epoch 4100 RMSE =  0.04410094325870515
Epoch 4200 RMSE =  0.04310469424919024
Epoch 4300 RMSE =  0.04217428511984039
Epoch 4400 RMSE =  0.04130523074591088
Epoch 4500 RMSE =  0.040492688824942245
Epoch 4600 RMSE =  0.03973235946506489
Epoch 4700 RMSE =  0.039019292828123095
Epoch 4800 RMSE =  0.03835102219805117
Epoch 4900 RMSE =  0.037722405393026674
Sample: [0.43] expected: [0.4168708] produced: [0.42377161805657665]
Sample: [0.35] expected: [0.34289781] produced: [0.34534637245020405]
Sample: [0.48] expected: [0.46177918] produced: [0.4726568306864379]
Sample: [0.23] expected: [0.22797752] produced: [0.23715008506370155]
Sample: [0.94] expected: [0.8075581] produced: [0.7886997287798897]
Sample: [1.5] expected: [0.99749499] produced: [0.9003427661674278]
Sample: [0.67] expected: [0.62098599] produced: [0.6384985953415834]
Sample: [0.33] expected: [0.32404303] produced: [0.32628882540808024]
Sample: [0.68] expected: [0.62879302] produced: [0.6458044784061713]
Sample: [1.22] expected: [0.93909936] produced: [0.8645866913043189]
Sample: [1.05] expected: [0.86742323] produced: [0.8257335526413274]
Sample: [0.18] expected: [0.17902957] produced: [0.1984397759839195]
Sample: [0.11] expected: [0.1097783] produced: [0.15151329077816358]
Sample: [0.57] expected: [0.53963205] produced: [0.5566645332768776]
Sample: [0.63] expected: [0.58914476] produced: [0.6073639513779366]
Epoch 5000 RMSE =  0.03713234772185714
Epoch 5100 RMSE =  0.03657649421297887
Epoch 5200 RMSE =  0.03605274299392332
Epoch 5300 RMSE =  0.03555846492862252
Epoch 5400 RMSE =  0.03509174483136411
Epoch 5500 RMSE =  0.034650551851829726
Epoch 5600 RMSE =  0.0342328844812109
Epoch 5700 RMSE =  0.03383737672591507
Epoch 5800 RMSE =  0.03346218196140324
Epoch 5900 RMSE =  0.033105710404913065
Sample: [0.43] expected: [0.4168708] produced: [0.41884193988655066]
Sample: [0.33] expected: [0.32404303] produced: [0.3194795351084339]
Sample: [0.35] expected: [0.34289781] produced: [0.33888382903350106]
Sample: [1.5] expected: [0.99749499] produced: [0.909729402614008]
Sample: [0.18] expected: [0.17902957] produced: [0.19126065013670762]
Sample: [0.23] expected: [0.22797752] produced: [0.22998918324108117]
Sample: [0.57] expected: [0.53963205] produced: [0.5557038587576113]
Sample: [1.05] expected: [0.86742323] produced: [0.8339576056017539]
Sample: [0.11] expected: [0.1097783] produced: [0.14495783124470016]
Sample: [0.63] expected: [0.58914476] produced: [0.608278442890231]
Sample: [0.48] expected: [0.46177918] produced: [0.46909707100883086]
Sample: [0.94] expected: [0.8075581] produced: [0.796087255333984]
Sample: [0.68] expected: [0.62879302] produced: [0.6480960594369647]
Sample: [1.22] expected: [0.93909936] produced: [0.8736143268351385]
Sample: [0.67] expected: [0.62098599] produced: [0.6404881260560334]
Epoch 6000 RMSE =  0.032767439377357106
Epoch 6100 RMSE =  0.03244511172888544
Epoch 6200 RMSE =  0.03213843849704182
Epoch 6300 RMSE =  0.031845161238510916
Epoch 6400 RMSE =  0.031566566839238704
Epoch 6500 RMSE =  0.03130000423237009
Epoch 6600 RMSE =  0.03104530439954688
Epoch 6700 RMSE =  0.030801238039466362
Epoch 6800 RMSE =  0.030567649069832753
Epoch 6900 RMSE =  0.030343724759452663
Sample: [1.5] expected: [0.99749499] produced: [0.916293999743347]
Sample: [0.18] expected: [0.17902957] produced: [0.18742889866777818]
Sample: [0.48] expected: [0.46177918] produced: [0.46663325987350884]
Sample: [0.23] expected: [0.22797752] produced: [0.2259027424885816]
Sample: [0.67] expected: [0.62098599] produced: [0.6414072521178599]
Sample: [0.35] expected: [0.34289781] produced: [0.33490085030637984]
Sample: [0.63] expected: [0.58914476] produced: [0.6083308171984606]
Sample: [0.57] expected: [0.53963205] produced: [0.5544592326943917]
Sample: [1.05] expected: [0.86742323] produced: [0.8393366444542492]
Sample: [0.43] expected: [0.4168708] produced: [0.4155155118546069]
Sample: [0.94] expected: [0.8075581] produced: [0.8007781150664329]
Sample: [0.11] expected: [0.1097783] produced: [0.14157175062341013]
Sample: [1.22] expected: [0.93909936] produced: [0.8798138714814661]
Sample: [0.33] expected: [0.32404303] produced: [0.31547494245360297]
Sample: [0.68] expected: [0.62879302] produced: [0.6492765004672328]
Epoch 7000 RMSE =  0.030128816270173574
Epoch 7100 RMSE =  0.029922418937794765
Epoch 7200 RMSE =  0.02972395647322933
Epoch 7300 RMSE =  0.02953289662280196
Epoch 7400 RMSE =  0.02934938765425818
Epoch 7500 RMSE =  0.02917244131094188
Epoch 7600 RMSE =  0.02900172257064195
Epoch 7700 RMSE =  0.028837085056629342
Epoch 7800 RMSE =  0.02867813672222387
Epoch 7900 RMSE =  0.028524478158408553
Sample: [0.67] expected: [0.62098599] produced: [0.6416465270603487]
Sample: [0.57] expected: [0.53963205] produced: [0.5534968182149327]
Sample: [0.18] expected: [0.17902957] produced: [0.18527125122157698]
Sample: [0.33] expected: [0.32404303] produced: [0.3127749444419449]
Sample: [0.11] expected: [0.1097783] produced: [0.13990275783932252]
Sample: [1.05] expected: [0.86742323] produced: [0.8431912611037955]
Sample: [0.48] expected: [0.46177918] produced: [0.4645167347978584]
Sample: [1.22] expected: [0.93909936] produced: [0.8842704685262124]
Sample: [0.63] expected: [0.58914476] produced: [0.6081345459563954]
Sample: [0.68] expected: [0.62879302] produced: [0.6495333874947358]
Sample: [0.43] expected: [0.4168708] produced: [0.4131506722513457]
Sample: [0.35] expected: [0.34289781] produced: [0.3322587317677739]
Sample: [0.23] expected: [0.22797752] produced: [0.22351218830708366]
Sample: [1.5] expected: [0.99749499] produced: [0.9211125410256503]
Sample: [0.94] expected: [0.8075581] produced: [0.804034573940747]
Epoch 8000 RMSE =  0.02837584261360824
Epoch 8100 RMSE =  0.02823228132758664
Epoch 8200 RMSE =  0.028093212338161856
Epoch 8300 RMSE =  0.0279584703117564
Epoch 8400 RMSE =  0.027828014558022213
Epoch 8500 RMSE =  0.027701331152203013
Epoch 8600 RMSE =  0.027578369521489045
Epoch 8700 RMSE =  0.027459113741633533
Epoch 8800 RMSE =  0.027343239108180102
Epoch 8900 RMSE =  0.027230720628856512
Sample: [0.33] expected: [0.32404303] produced: [0.3113189134298294]
Sample: [0.63] expected: [0.58914476] produced: [0.6078139053472331]
Sample: [0.94] expected: [0.8075581] produced: [0.8062361667728404]
Sample: [0.67] expected: [0.62098599] produced: [0.6415888340742006]
Sample: [0.68] expected: [0.62879302] produced: [0.6495554862338221]
Sample: [1.5] expected: [0.99749499] produced: [0.9247859739636234]
Sample: [0.23] expected: [0.22797752] produced: [0.22223883396553315]
Sample: [0.43] expected: [0.4168708] produced: [0.4116148048710106]
Sample: [1.22] expected: [0.93909936] produced: [0.8876395453105934]
Sample: [0.11] expected: [0.1097783] produced: [0.1392117072206075]
Sample: [0.48] expected: [0.46177918] produced: [0.46317388963269746]
Sample: [1.05] expected: [0.86742323] produced: [0.8461238673814216]
Sample: [0.57] expected: [0.53963205] produced: [0.5527816917441593]
Sample: [0.18] expected: [0.17902957] produced: [0.18428047365312664]
Sample: [0.35] expected: [0.34289781] produced: [0.3307362606103951]
Epoch 9000 RMSE =  0.02712113215426569
Epoch 9100 RMSE =  0.027014624032773105
Epoch 9200 RMSE =  0.026910827958592336
Epoch 9300 RMSE =  0.02681014988592872
Epoch 9400 RMSE =  0.02671187976509795
Epoch 9500 RMSE =  0.026615888708418012
Epoch 9600 RMSE =  0.026522808789573422
Epoch 9700 RMSE =  0.026431307418598716
Epoch 9800 RMSE =  0.026342914570763168
Epoch 9900 RMSE =  0.0262562030285633
Sample: [1.22] expected: [0.93909936] produced: [0.8903369858710752]
Sample: [0.35] expected: [0.34289781] produced: [0.3298004458252062]
Sample: [0.23] expected: [0.22797752] produced: [0.2217249330358165]
Sample: [0.33] expected: [0.32404303] produced: [0.3104417455633037]
Sample: [0.68] expected: [0.62879302] produced: [0.649988490367413]
Sample: [0.48] expected: [0.46177918] produced: [0.46226655684270257]
Sample: [0.57] expected: [0.53963205] produced: [0.5521589562706539]
Sample: [0.94] expected: [0.8075581] produced: [0.8080175025547177]
Sample: [0.67] expected: [0.62098599] produced: [0.6416119028244719]
Sample: [1.5] expected: [0.99749499] produced: [0.9278259119754673]
Sample: [0.11] expected: [0.1097783] produced: [0.13896453629361546]
Sample: [1.05] expected: [0.86742323] produced: [0.8483288202549224]
Sample: [0.43] expected: [0.4168708] produced: [0.4106780269878748]
Sample: [0.18] expected: [0.17902957] produced: [0.1838433938554965]
Sample: [0.63] expected: [0.58914476] produced: [0.6075027626758194]
Epoch 10000 RMSE =  0.026171485657366943
Final Epoch RMSE =  0.026171485657366943
Sample: [1.21] expected: [0.935616] produced: [0.8883878843892367]
Sample: [0.1] expected: [0.09983342] produced: [0.13333561455531948]
Sample: [0.16] expected: [0.15931821] produced: [0.17003002222923724]
Sample: [0.5] expected: [0.47942554] produced: [0.4826229025926599]
Sample: [0.99] expected: [0.83602598] produced: [0.8279056119585586]
Sample: [0.75] expected: [0.68163876] produced: [0.7024980385561774]
Sample: [1.2] expected: [0.93203909] produced: [0.8863629510175149]
Sample: [0.91] expected: [0.78950374] produced: [0.7946872794980968]
Sample: [0.55] expected: [0.52268723] produced: [0.5326658446471161]
Sample: [1.47] expected: [0.99492435] produced: [0.9250357554732883]
Sample: [1.24] expected: [0.945784] produced: [0.8941367289843009]
Sample: [0.85] expected: [0.75128041] produced: [0.7646403025595963]
Sample: [0.34] expected: [0.33348709] produced: [0.32007518009672914]
Sample: [0.61] expected: [0.57286746] produced: [0.5897170553109566]
Sample: [1.46] expected: [0.99386836] produced: [0.9241028407077168]
Sample: [0.79] expected: [0.71035327] produced: [0.7293301549811969]
Sample: [1.48] expected: [0.99588084] produced: [0.9260606019676215]
Sample: [0.44] expected: [0.42593947] produced: [0.42119514796402535]
Sample: [0.32] expected: [0.31456656] produced: [0.3009870645861194]
Sample: [0.21] expected: [0.2084599] produced: [0.20610376528425958]
Sample: [1.54] expected: [0.99952583] produced: [0.9313981827226345]
Sample: [0.04] expected: [0.03998933] produced: [0.10340257778966447]
Sample: [1.15] expected: [0.91276394] produced: [0.8757110004242636]
Sample: [0.28] expected: [0.27635565] produced: [0.26432966203375813]
Sample: [0.69] expected: [0.63653718] produced: [0.6583400017068819]
Sample: [0.03] expected: [0.0299955] produced: [0.09899456654487661]
Sample: [1.38] expected: [0.98185353] produced: [0.9153049318456205]
Sample: [1.51] expected: [0.99815247] produced: [0.928914222459635]
Sample: [0.76] expected: [0.68892145] produced: [0.7099797553150269]
Sample: [1.39] expected: [0.98370081] produced: [0.9165777986461082]
Sample: [1.3] expected: [0.96355819] produced: [0.9044097349661062]
Sample: [0.02] expected: [0.01999867] produced: [0.09481589541778859]
Sample: [1.53] expected: [0.99916795] produced: [0.9306911096405244]
Sample: [1.07] expected: [0.8772005] produced: [0.8549473476455153]
Sample: [1.41] expected: [0.9871001] produced: [0.9190631364935973]
Sample: [0.45] expected: [0.43496553] produced: [0.4322560566912137]
Sample: [0.98] expected: [0.83049737] produced: [0.8249297024887076]
Sample: [0.12] expected: [0.11971221] produced: [0.14507600253060202]
Sample: [0.54] expected: [0.51413599] produced: [0.5238938079051575]
Sample: [0.31] expected: [0.30505864] produced: [0.29202864204808693]
Sample: [1.49] expected: [0.99673775] produced: [0.927292579780976]
Sample: [1.43] expected: [0.99010456] produced: [0.9213630978309294]
Sample: [0.46] expected: [0.44394811] produced: [0.4428257842133754]
Sample: [1.26] expected: [0.95209034] produced: [0.8982311430816087]
Sample: [0.66] expected: [0.61311685] produced: [0.6347969469972198]
Sample: [1.34] expected: [0.97348454] produced: [0.9104825849702326]
Sample: [0.88] expected: [0.77073888] produced: [0.7813905730738457]
Sample: [1.31] expected: [0.96618495] produced: [0.9062421808432729]
Sample: [0.97] expected: [0.82488571] produced: [0.8214243581953661]
Sample: [0.15] expected: [0.14943813] produced: [0.1639336577209563]
Sample: [1.] expected: [0.84147098] produced: [0.8326424577227101]
Sample: [1.16] expected: [0.91680311] produced: [0.878647091045113]
Sample: [0.64] expected: [0.59719544] produced: [0.6179743430572899]
Sample: [1.33] expected: [0.97114838] produced: [0.9091943843294016]
Sample: [1.1] expected: [0.89120736] produced: [0.8639038791584576]
Sample: [0.53] expected: [0.50553334] produced: [0.5147158063629677]
Sample: [0.73] expected: [0.66686964] produced: [0.689948534161077]
Sample: [1.04] expected: [0.86240423] produced: [0.8462380146365851]
Sample: [0.49] expected: [0.47062589] produced: [0.47406340229637456]
Sample: [1.19] expected: [0.92836897] produced: [0.8851947189947579]
Sample: [0.71] expected: [0.65183377] produced: [0.6751037516391366]
Sample: [0.24] expected: [0.23770263] produced: [0.2305990266089781]
Sample: [0.87] expected: [0.76432894] produced: [0.776475272867262]
Sample: [1.11] expected: [0.89569869] produced: [0.86649460431905]
Sample: [0.9] expected: [0.78332691] produced: [0.7913404880037879]
Sample: [1.12] expected: [0.90010044] produced: [0.8691033045318128]
Sample: [1.42] expected: [0.98865176] produced: [0.9204853546835834]
Sample: [1.29] expected: [0.96083506] produced: [0.9033982326536765]
Sample: [0.06] expected: [0.05996401] produced: [0.11299287347460474]
Sample: [1.27] expected: [0.95510086] produced: [0.9002112781011361]
Sample: [0.7] expected: [0.64421769] produced: [0.6677689324349466]
Sample: [1.06] expected: [0.87235548] produced: [0.8526784366432082]
Sample: [1.23] expected: [0.9424888] produced: [0.8932373790133467]
Sample: [1.44] expected: [0.99145835] produced: [0.9227738758984365]
Sample: [1.45] expected: [0.99271299] produced: [0.9238480698622235]
Sample: [1.01] expected: [0.84683184] produced: [0.8367897275682458]
Sample: [0.51] expected: [0.48817725] produced: [0.4952696550510393]
Sample: [0.86] expected: [0.75784256] produced: [0.7719223806555965]
Sample: [0.14] expected: [0.13954311] produced: [0.15774786511672764]
Sample: [1.32] expected: [0.9687151] produced: [0.9080891562862008]
Sample: [1.4] expected: [0.98544973] produced: [0.9184846844634683]
Sample: [0.38] expected: [0.37092047] produced: [0.36170777149916405]
Sample: [1.18] expected: [0.92460601] produced: [0.8836044935597908]
Sample: [0.47] expected: [0.45288629] produced: [0.45456615555184826]
Sample: [0.07] expected: [0.06994285] produced: [0.11807408008862778]
Sample: [0.93] expected: [0.80161994] produced: [0.8057828909567546]
Sample: [1.03] expected: [0.85729899] produced: [0.8437053705857601]
Sample: [1.28] expected: [0.95801586] produced: [0.9021780152829533]
Sample: [1.52] expected: [0.99871014] produced: [0.9305170237990723]
Sample: [0.56] expected: [0.5311862] produced: [0.5455323615980193]
Sample: [0.01] expected: [0.00999983] produced: [0.09102797952665696]
Sample: [0.74] expected: [0.67428791] produced: [0.6982256183774228]
Sample: [0.8] expected: [0.71735609] produced: [0.7378820827654214]
Sample: [0.19] expected: [0.18885889] produced: [0.1920222599393758]
Sample: [1.14] expected: [0.9086335] produced: [0.8745482074509516]
Sample: [0.22] expected: [0.21822962] produced: [0.21493366242013562]
Sample: [0.84] expected: [0.74464312] produced: [0.7613156762864908]
Sample: [0.4] expected: [0.38941834] produced: [0.38207824277107383]
Sample: [0.26] expected: [0.25708055] produced: [0.24800490758273602]
Sample: [0.89] expected: [0.77707175] produced: [0.7873412171399641]
Sample: [1.17] expected: [0.9207506] produced: [0.8814378406193738]
Sample: [0.77] expected: [0.69613524] produced: [0.7187453146285049]
Sample: [0.] expected: [0.] produced: [0.08704929260999612]
Sample: [0.05] expected: [0.04997917] produced: [0.10830123493228765]
Sample: [0.83] expected: [0.73793137] produced: [0.7555014149435061]
Sample: [0.29] expected: [0.28595223] produced: [0.2744391556857495]
Sample: [0.78] expected: [0.70327942] produced: [0.7250117738746487]
Sample: [0.52] expected: [0.49688014] produced: [0.5052088969463827]
Sample: [0.82] expected: [0.73114583] produced: [0.7495227485470487]
Sample: [0.13] expected: [0.12963414] produced: [0.15139816520712965]
Sample: [0.81] expected: [0.72428717] produced: [0.7434637852282893]
Sample: [0.36] expected: [0.35227423] produced: [0.3411273262561055]
Sample: [0.27] expected: [0.26673144] produced: [0.25636358482967797]
Sample: [0.3] expected: [0.29552021] produced: [0.2834379590100697]
Sample: [1.08] expected: [0.88195781] produced: [0.8586664632598541]
Sample: [1.55] expected: [0.99978376] produced: [0.932777444630537]
Sample: [0.09] expected: [0.08987855] produced: [0.12837194292663928]
Sample: [1.36] expected: [0.9778646] produced: [0.9135396911056901]
Sample: [0.25] expected: [0.24740396] produced: [0.23933439743880905]
Sample: [0.59] expected: [0.55636102] produced: [0.5736227443054549]
Sample: [0.37] expected: [0.36161543] produced: [0.35139239906109937]
Sample: [1.02] expected: [0.85210802] produced: [0.840212555028252]
Sample: [1.37] expected: [0.97990806] produced: [0.9148709417609198]
Sample: [0.92] expected: [0.79560162] produced: [0.8012969322473535]
Sample: [0.41] expected: [0.39860933] produced: [0.3923033037676139]
Sample: [1.35] expected: [0.97572336] produced: [0.9123489576433731]
Sample: [0.95] expected: [0.8134155] produced: [0.8142980183048165]
Sample: [1.13] expected: [0.90441219] produced: [0.8722209613018855]
Sample: [0.72] expected: [0.65938467] produced: [0.6837027452346249]
Sample: [0.58] expected: [0.54802394] produced: [0.5644371224793039]
Sample: [0.39] expected: [0.38018842] produced: [0.37183716888675333]
Sample: [0.62] expected: [0.58103516] produced: [0.6013223233709465]
Sample: [1.57] expected: [0.99999968] produced: [0.9343896131611003]
Sample: [0.17] expected: [0.16918235] produced: [0.17774437552831712]
Sample: [0.65] expected: [0.60518641] produced: [0.6275460444631505]
Sample: [0.42] expected: [0.40776045] produced: [0.40252000949068095]
Sample: [0.6] expected: [0.56464247] produced: [0.5829558759843183]
Sample: [1.56] expected: [0.99994172] produced: [0.9335988913397574]
Sample: [0.2] expected: [0.19866933] produced: [0.1993961641381533]
Sample: [0.08] expected: [0.07991469] produced: [0.12310195078409145]
Sample: [0.96] expected: [0.81919157] produced: [0.8181356117222687]
Sample: [1.09] expected: [0.88662691] produced: [0.8616355572980537]
Sample: [1.25] expected: [0.94898462] produced: [0.8970754857211972]
Final Epoch RMSE =  0.03888371987580243
Sample: [1. 0.] expected: [1.] produced: [0.5905471783104774]
Sample: [1. 1.] expected: [0.] produced: [0.6049491040184231]
Sample: [0. 0.] expected: [0.] produced: [0.570764484017694]
Sample: [0. 1.] expected: [1.] produced: [0.5833218933914439]
Epoch 0 RMSE =  0.5081849496947644
Epoch 100 RMSE =  0.5012834770065099
Epoch 200 RMSE =  0.5008285212961834
Epoch 300 RMSE =  0.5007939922474459
Epoch 400 RMSE =  0.5007844485997185
Epoch 500 RMSE =  0.5007761531988807
Epoch 600 RMSE =  0.5007663939763731
Epoch 700 RMSE =  0.50075395266663
Epoch 800 RMSE =  0.5007464130654645
Epoch 900 RMSE =  0.5007330002334228
Sample: [1. 0.] expected: [1.] produced: [0.5095770871011716]
Sample: [0. 1.] expected: [1.] produced: [0.49562273908682547]
Sample: [0. 0.] expected: [0.] produced: [0.5025065702782913]
Sample: [1. 1.] expected: [0.] produced: [0.5054345983578541]
Epoch 1000 RMSE =  0.5007214897556487
Epoch 1100 RMSE =  0.5007127220870974
Epoch 1200 RMSE =  0.5006962299608497
Epoch 1300 RMSE =  0.5006849812189178
Epoch 1400 RMSE =  0.5006699679012074
Epoch 1500 RMSE =  0.5006511534707504
Epoch 1600 RMSE =  0.5006367019353691
Epoch 1700 RMSE =  0.5006178800287786
Epoch 1800 RMSE =  0.5005945943418254
Epoch 1900 RMSE =  0.5005751665626428
Sample: [1. 0.] expected: [1.] produced: [0.511495376527746]
Sample: [1. 1.] expected: [0.] produced: [0.5064115883108915]
Sample: [0. 1.] expected: [1.] produced: [0.4925256576106532]
Sample: [0. 0.] expected: [0.] produced: [0.4995862552691042]
Epoch 2000 RMSE =  0.5005512207972914
Epoch 2100 RMSE =  0.500521231638487
Epoch 2200 RMSE =  0.500492838821988
Epoch 2300 RMSE =  0.5004604419106606
Epoch 2400 RMSE =  0.5004299872871715
Epoch 2500 RMSE =  0.5003880837352696
Epoch 2600 RMSE =  0.5003497428672926
Epoch 2700 RMSE =  0.5003034173360181
Epoch 2800 RMSE =  0.5002532031123367
Epoch 2900 RMSE =  0.5001963223134748
Sample: [1. 0.] expected: [1.] produced: [0.5152043046013727]
Sample: [1. 1.] expected: [0.] produced: [0.5093778507023711]
Sample: [0. 1.] expected: [1.] produced: [0.4912575140658819]
Sample: [0. 0.] expected: [0.] produced: [0.4972178112101527]
Epoch 3000 RMSE =  0.5001342644336236
Epoch 3100 RMSE =  0.5000601972906964
Epoch 3200 RMSE =  0.49998640525044946
Epoch 3300 RMSE =  0.4998982419766805
Epoch 3400 RMSE =  0.49979782406242734
Epoch 3500 RMSE =  0.49968793042756726
Epoch 3600 RMSE =  0.49956696804884404
Epoch 3700 RMSE =  0.4994279894359432
Epoch 3800 RMSE =  0.49927107006447113
Epoch 3900 RMSE =  0.49909321474211393
Sample: [0. 1.] expected: [1.] produced: [0.4909388303420119]
Sample: [0. 0.] expected: [0.] produced: [0.4926630991277703]
Sample: [1. 1.] expected: [0.] produced: [0.5143731648385665]
Sample: [1. 0.] expected: [1.] produced: [0.5213258665974586]
Epoch 4000 RMSE =  0.4988909906959774
Epoch 4100 RMSE =  0.49866224459069103
Epoch 4200 RMSE =  0.4983982070058509
Epoch 4300 RMSE =  0.49810566678890256
Epoch 4400 RMSE =  0.49776893836775976
Epoch 4500 RMSE =  0.4973821539225119
Epoch 4600 RMSE =  0.4969496435756368
Epoch 4700 RMSE =  0.49645497286492296
Epoch 4800 RMSE =  0.49589172300948664
Epoch 4900 RMSE =  0.4952558517653999
Sample: [1. 0.] expected: [1.] produced: [0.5401612433191968]
Sample: [0. 0.] expected: [0.] produced: [0.4804134356964067]
Sample: [1. 1.] expected: [0.] produced: [0.5250260910199132]
Sample: [0. 1.] expected: [1.] produced: [0.4897361845343483]
Epoch 5000 RMSE =  0.4945377409684187
Epoch 5100 RMSE =  0.4937290073537973
Epoch 5200 RMSE =  0.4928229661117221
Epoch 5300 RMSE =  0.4918066982838865
Epoch 5400 RMSE =  0.4906886262988218
Epoch 5500 RMSE =  0.4894436250987961
Epoch 5600 RMSE =  0.48808935551733124
Epoch 5700 RMSE =  0.4866062692320424
Epoch 5800 RMSE =  0.48500189918400066
Epoch 5900 RMSE =  0.48327559613479587
Sample: [1. 0.] expected: [1.] produced: [0.5743333089369982]
Sample: [0. 0.] expected: [0.] produced: [0.45542976275783276]
Sample: [0. 1.] expected: [1.] produced: [0.49918366254825786]
Sample: [1. 1.] expected: [0.] produced: [0.5363472776161596]
Epoch 6000 RMSE =  0.4814285530457827
Epoch 6100 RMSE =  0.4794705114410989
Epoch 6200 RMSE =  0.4774026728189167
Epoch 6300 RMSE =  0.4752295054403434
Epoch 6400 RMSE =  0.4729652803155814
Epoch 6500 RMSE =  0.47062290038692023
Epoch 6600 RMSE =  0.4681958097990438
Epoch 6700 RMSE =  0.46569232046674947
Epoch 6800 RMSE =  0.4631399897547433
Epoch 6900 RMSE =  0.46052075192058173
Sample: [1. 0.] expected: [1.] produced: [0.6127952670239051]
Sample: [1. 1.] expected: [0.] produced: [0.5244141652248168]
Sample: [0. 1.] expected: [1.] produced: [0.5131808745060068]
Sample: [0. 0.] expected: [0.] produced: [0.42023089232607447]
Epoch 7000 RMSE =  0.4578549403848984
Epoch 7100 RMSE =  0.4551408838223377
Epoch 7200 RMSE =  0.45238544887380366
Epoch 7300 RMSE =  0.44959464260818555
Epoch 7400 RMSE =  0.44676585611458436
Epoch 7500 RMSE =  0.443908295675124
Epoch 7600 RMSE =  0.4410216239223103
Epoch 7700 RMSE =  0.43810189575448
Epoch 7800 RMSE =  0.43517504319457456
Epoch 7900 RMSE =  0.432208945797332
Sample: [0. 1.] expected: [1.] produced: [0.5529610918077602]
Sample: [0. 0.] expected: [0.] produced: [0.3828090052438809]
Sample: [1. 0.] expected: [1.] produced: [0.630522098079068]
Sample: [1. 1.] expected: [0.] produced: [0.5040694725536503]
Epoch 8000 RMSE =  0.4292395814408104
Epoch 8100 RMSE =  0.4262547817050125
Epoch 8200 RMSE =  0.42324732131772647
Epoch 8300 RMSE =  0.42024734875895997
Epoch 8400 RMSE =  0.41722861826083646
Epoch 8500 RMSE =  0.4142032766862537
Epoch 8600 RMSE =  0.41117248102002873
Epoch 8700 RMSE =  0.4081224187944483
Epoch 8800 RMSE =  0.4050525548658715
Epoch 8900 RMSE =  0.40196610661036547
Sample: [0. 0.] expected: [0.] produced: [0.3424523303830785]
Sample: [1. 0.] expected: [1.] produced: [0.6305134312413353]
Sample: [0. 1.] expected: [1.] produced: [0.6063731772983085]
Sample: [1. 1.] expected: [0.] produced: [0.47701572414959875]
Epoch 9000 RMSE =  0.3988358057252531
Epoch 9100 RMSE =  0.3956547402071033
Epoch 9200 RMSE =  0.3923913606939939
Epoch 9300 RMSE =  0.3890525407467729
Epoch 9400 RMSE =  0.3855846138318524
Epoch 9500 RMSE =  0.3819685197287504
Epoch 9600 RMSE =  0.37817530509899616
Epoch 9700 RMSE =  0.3741531499717167
Epoch 9800 RMSE =  0.36987302421309665
Epoch 9900 RMSE =  0.36528352033957745
Sample: [1. 1.] expected: [0.] produced: [0.4236673432472585]
Sample: [0. 1.] expected: [1.] produced: [0.6470096733601199]
Sample: [0. 0.] expected: [0.] produced: [0.3060163425733581]
Sample: [1. 0.] expected: [1.] produced: [0.65116212075073]
Epoch 10000 RMSE =  0.3603574809673801
Final Epoch RMSE =  0.3603574809673801

Process finished with exit code 0
"""
