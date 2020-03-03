"""
Feedfoward and Backpropogation Neural Network with menu
Author:Peter Wright
Beginning of project: 3/26/19 - presently working on.

This a neural network that has been a school project
and is steadily developing new features over time. (Guidance provided by foothill college, Professor Eric Reed)
"""

import random
import collections
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod
import json
from math import sqrt
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
import operator

"""
This class is used to prepare the data (split into loaded training/test lists) while keeping
track of indices, and primes-data (check enums)  `

"""


#  This exception class is for mismatched x (example, predictor, independent variable) and y (label, dependent)


class DataMismatchError(Exception):
    pass


class NNData:
    class Order(Enum):
        RANDOM = 0
        SEQUENTIAL = 1

    class Set(Enum):
        TRAIN = 0
        TEST = 1

    def __init__(self, x=None, y=None, train_percentage=100):
        if x is None:
            x = []
        if y is None:
            y = []
        self.train_percentage = NNData.percentage_limiter(train_percentage)
        self.x = None
        self.y = None
        self.train_indices = None
        self.test_indices = None
        self.train_pool = None
        self.test_pool = None
        self.load_data(x, y)
        pass

    def load_data(self, x, y):
        if not len(x) == len(y):
            raise DataMismatchError
        self.x = x
        self.y = y
        self.split_set()

    # splits data into train and testing data while keeping track of the indices
    def split_set(self, new_train_percentage=None):
        if new_train_percentage is not None:
            self.train_percentage = NNData.percentage_limiter(new_train_percentage)
        train_size = int(len(self.x) * (self.train_percentage * .01))
        self.train_indices = random.sample(range(0, len(self.x)), train_size)
        self.test_indices = list(set(range(0, len(self.x))) - set(self.train_indices))
        self.prime_data()

    def get_one_item(self, my_set=None):
        if my_set is None:
            my_set = NNData.Set.TRAIN
        try:
            if my_set == NNData.Set.TRAIN:
                index = self.train_pool.popleft()
            else:
                index = self.test_pool.popleft()
            return self.x[index], self.y[index]
        except IndexError:
            return None

    def get_number_samples(self, my_set=None):
        if my_set is None:
            return len(self.x)
        # or return (len(self.test_pool) + len(self.train_pool)) this made more sense to change
        if my_set == NNData.Set.TRAIN:
            return len(self.train_pool)
        if my_set == NNData.Set.TEST:
            return len(self.test_pool)

    def empty_pool(self, my_set=None):
        if my_set is None:
            my_set = NNData.Set.TRAIN
        if ((my_set == NNData.Set.TRAIN) and (len(self.train_pool) == 0)) or \
                ((my_set == NNData.Set.TEST) and (len(self.test_pool) == 0)):
            return True
        else:
            return False

    def prime_data(self, my_set=None, order=None):
        no_case = 0
        # if order is None, set order to sequential
        if my_set is None:
            no_case = 1
        if order is None:
            order = NNData.Order.SEQUENTIAL
        if order == NNData.Order.RANDOM:
            # 2. make copies of self.test_indices and self.train_indices
            # 3. shuffle the copies of both *only if* order is random
            # 4. finally, populate test or train pool or both as needed
            if (my_set == NNData.Set.TRAIN) or (no_case == 1):
                train_indices_temp = list(self.train_indices)
                random.shuffle(train_indices_temp)
                self.train_pool = collections.deque(train_indices_temp)
            if (my_set == NNData.Set.TEST) or (no_case == 1):
                test_indices_temp = list(self.test_indices)
                random.shuffle(test_indices_temp)
                self.test_pool = collections.deque(test_indices_temp)
        else:
            if (my_set == NNData.Set.TRAIN) or (no_case == 1):
                self.train_pool = collections.deque(self.train_indices)
            if (my_set == NNData.Set.TEST) or (no_case == 1):
                self.test_pool = collections.deque(self.test_indices)

    #  rounds the percentage down if exceeding or not
    @staticmethod
    def percentage_limiter(percentage):
        if percentage < 0:
            return 0
        if percentage > 100:
            return 100
        if 0 <= percentage <= 100:
            return percentage


"""
Class Neurode is inherited from MultiLinkNode. This MultiLinkNode class gives
attributes to each nodes' inputs and outputs -- whether the node is full, or 
which node is connected to which via dictionary. The Neuorode are holding the values, type (hidden,
input, output). (Nuerode class is most granular part to the list of classes, but obviously crucial.)
"""


# creates an Enum for the Types of Layers


class LayerType(Enum):
    INPUT = 0
    OUTPUT = 1
    HIDDEN = 2


class MultiLinkNode(ABC):

    def __init__(self):
        self.num_inputs = 0
        self.num_outputs = 0
        # keep track of which input connections have provided to node
        self.reporting_inputs = 0
        # keep track of which output connections have provided information to our node
        self.reporting_outputs = 0
        # Both compare attributes use the a binary encoded method for inticading if full or not
        # For example 11111 is full and 00011 is not full
        self.compare_inputs_full = 0
        self.compare_outputs_full = 0
        # the input node side and holds key and value in dictionary
        self.input_nodes = collections.OrderedDict()
        # the output node side and holds key and value in dictionary
        self.output_nodes = collections.OrderedDict()

    def __str__(self):
        ret_str = "-->Node " + str(id(self)) + "\n"
        ret_str = ret_str + "   Input Nodes:\n"
        for key in self.input_nodes:
            ret_str = ret_str + "   " + str(id(key)) + "\n"
        ret_str = ret_str + "   Output Nodes\n"
        for key in self.output_nodes:
            ret_str = ret_str + "   " + str(id(key)) + "\n"
        return ret_str

    @abstractmethod
    def process_new_input_node(self, node):
        pass

    @abstractmethod
    def process_new_output_node(self, node):
        pass

    # adds number of input and adds to compare fullness attribute
    def add_input_node(self, node):
        self.input_nodes[node] = None
        self.process_new_input_node(node)
        self.num_inputs += 1
        self.compare_inputs_full = 2 ** self.num_inputs - 1

    # it clears the entire dictionary of input nodes, and add indicated input nodes
    def clear_and_add_input_nodes(self, nodes):
        self.clear_inputs()
        for node in nodes:
            self.add_input_node(node)

    # It should clear the entire dictionary of output nodes, and add indicated output nodes
    def clear_and_add_output_nodes(self, nodes):
        self.clear_outputs()
        for node in nodes:
            self.add_output_node(node)

    def clear_inputs(self):
        self.input_nodes = collections.OrderedDict()
        self.num_inputs = 0
        self.compare_inputs_full = 0

    # adding an output node and to the number of outputs
    def add_output_node(self, node):
        self.output_nodes[node] = None
        self.process_new_output_node(node)
        self.num_outputs += 1
        self.compare_outputs_full = 2 ** self.num_outputs - 1

    def clear_outputs(self):
        self.output_nodes = collections.OrderedDict()
        self.num_outputs = 0
        self.compare_outputs_full = 0


class Neurode(MultiLinkNode):

    def __init__(self, my_type, rate):
        super().__init__()
        self.value = 0
        self.my_type = my_type
        self.learning_rate = rate

    def get_value(self):
        return self.value

    def get_type(self):
        return self.my_type

    # add random 0-1 random number to input node and store it in the input_nodes dictionary
    def process_new_input_node(self, node):
        rand_data = random.uniform(0.0, 1.0)
        self.input_nodes[node] = rand_data

    def process_new_output_node(self, node):
        pass


"""
FFNeurode(Neurode) takes the data from the input side
and then processes that data, Then it calculates its own
value and fires by passing (firing) its 
data or value to the output side. 
"""


class FFNeurode(Neurode):

    def __init__(self, my_type, rate):  # rate
        super().__init__(my_type, rate)  # rate

    #  First update our binary encoding reporting_inputs.  We
    #  will use the index of from_node in the input_nodes ordered dictionary
    #  to determine which bit position in reporting_inputs to change to 1
    def register_input(self, from_node):
        self.reporting_inputs = self.reporting_inputs | (2 ** list(self.input_nodes.keys()).index(from_node))
        #  should reset reporting_inputs to 0 and return True
        if self.reporting_inputs == self.compare_inputs_full:
            self.reporting_inputs = 0
            return True
        else:
            return False

    def receive_input(self, from_node=None, input_value=0):
        if self.my_type is LayerType.INPUT:
            self.value = input_value
            #  call receive_input on each neurode in output_nodes, pass on the data
            for key in self.output_nodes:
                key.receive_input(self)
        # could be a hidden Layer, output, register_input() responds to having all of the input connections
        # have data available, then fire the through
        elif self.register_input(from_node):
            self.fire()

    # sigmoid function, used in logistic regression, brings back a number 0-1
    @staticmethod
    def activate_sigmoid(value):
        return 1 / (1 + np.exp(-value))

    # gets value of node and multiplies by the weights coming from input_nodes
    # add these values to to a sigmoid function.
    # Example: 1/(1 + (value * weight + value * weight))
    # Then it sets the value for node and lets nodes know it has a value available.
    def fire(self):
        input_sum = 0
        for key, node_data in self.input_nodes.items():
            input_sum += key.get_value() * node_data
        self.value = FFNeurode.activate_sigmoid(input_sum)
        for key in self.output_nodes:
            key.receive_input(self)


"""
Class BPNeurode is inherited from class Neurode. It does the updating and 
feedback (changing weights, delta) to the nodes behind it. Hence backpropogation.
"""


class BPNeurode(Neurode):
    def __init__(self, my_type, rate):
        self.my_type = my_type
        self.delta = 0
        self.learning_rate = rate
        super().__init__(my_type, rate)  # note this changes the parent class attribute

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    # Calculates the delta by adding the sum of the weights of input nodes and
    # the previous delta value and multiplying that with the signmoid, will give
    # a new delta value.

    def calculate_delta(self, expected=None):
        output_sum = 0
        if self.my_type is LayerType.HIDDEN:
            for key, node_data in self.output_nodes.items():
                output_sum += key.get_delta() * key.get_weight_for_input_node(self)
            self.delta = output_sum * self.sigmoid_derivative(self.value)
        elif self.my_type is LayerType.OUTPUT:
            self.delta = (expected - self.value) * (self.sigmoid_derivative(self.value))

    # update_weights is able to update the input nodes with new weight
    # New weight = current weight + value of incoming node * delta * learning rate

    def update_weights(self):
        for key, node_data in self.output_nodes.items():
            adjustment = key.get_learning_rate() * key.get_delta() * self.value
            key.adjust_input_node(self, adjustment)

    # receive_back_layer gives a new delta if full or an output node,

    def receive_back_input(self, from_node, expected=None):
        if self.register_back_input(from_node):
            self.calculate_delta(expected)
            self.back_fire()
            if self.my_type is not LayerType.OUTPUT:
                self.update_weights()

    # based on fullness or if it is an outputlayer, register_back_inpuf
    # returns true or false

    def register_back_input(self, from_node):
        if self.get_type() == LayerType.OUTPUT:
            return True
        else:
            self.reporting_outputs = self.reporting_outputs | (2 ** list(self.output_nodes.keys()).index(from_node))
            if self.reporting_outputs == self.compare_outputs_full:
                self.reporting_outputs = 0
                return True
            else:
                return False

    # back fire initiates a call receive_back_input on each of the neurode's input nodes.

    def back_fire(self):
        for key in self.input_nodes:
            key.receive_back_input(self)

    def get_learning_rate(self):
        return self.learning_rate

    def get_weight_for_input_node(self, from_node):
        return self.input_nodes[from_node]

    def get_delta(self):
        return self.delta

    def adjust_input_node(self, node, value):
        self.input_nodes[node] += value


class FFBPNeurode(FFNeurode, BPNeurode):

    def __init__(self, my_type, rate):  # rate
        super().__init__(my_type, rate)  # rate


"""
The class DoublinkedList is a two linked list that incorporates the 
DLLNodes. (Double Linked nodes) These classes allow the user to iterate
through the node graphs going back and forth. 
"""


# DLLNode class with basic linklist node structure

class DLLNode:

    def __init__(self):
        self.prev = None
        self.next = None

    def set_next(self, next_node):
        self.next = next_node

    def get_next(self):
        return self.next

    def set_prev(self, prev_node):
        self.prev = prev_node

    def get_prev(self):
        return self.prev

    def __str__(self):
        return "(generic node)"


# DoublyLinkedList allows for easy manipulate of node such
# as removing, iterating, and adding the nodes,


class DoublyLinkedList:

    def __init__(self):
        self.head = None
        self.tail = None
        self.current = None

    def reset_cur(self):
        self.current = self.head
        return self.current

    def check_iterate(self):
        if self.current is not None:
            self.current = self.current.get_next()
        return self.current

    def iterate(self):
        if self.current.get_next() is None:
            print("\"Alert: Unable to Iterate\"")
        elif self.current is not None:
            self.current = self.current.get_next()
        return self.current

    def rev_iterate(self):
        if self.current.get_prev() is None:
            print("\"Alert: Unable to Rev-iterate\"")
        elif self.current is not None:
            self.current = self.current.get_prev()
        return self.current

    def add_to_head(self, new_node):
        if isinstance(new_node, DLLNode):
            new_node.set_next(self.head)
            self.head = new_node
        if new_node.get_next() is not None:
            (new_node.get_next()).set_prev(new_node)
        else:
            self.tail = new_node

    def remove_from_head(self):
        ret_node = self.head
        if self.tail is None:
            return None
        if self.tail == self.head:
            self.tail = None
            self.head = None
            return ret_node
        if ret_node is not None:
            self.head = ret_node.get_next()  # unlink
            ret_node.set_next(None)  # don't give client way in
            (self.head).set_prev(None)  # don't give client way in
        if self.current is ret_node:
            self.current = None
        return ret_node

    def insert_after_cur(self, new_node):
        if isinstance(new_node, DLLNode) and self.current:
            new_node.set_next(self.current.get_next())
            self.current.set_next(new_node)
            new_node.set_prev(self.current)
            if new_node.get_next() is None:
                self.tail = new_node
            else:
                (new_node.get_next()).set_prev(new_node)
            return True
        else:
            return False

    def remove_after_cur(self):
        if self.current is None or self.current.get_next() is None:
            return False
        else:
            if self.current.get_next().get_next() is None:
                (self.current.get_next().set_next(None))
                (self.current.get_next().set_prev(None))
                self.current.set_next(None)
                self.tail = self.current
            else:
                self.current.set_next(self.current.get_next().get_next())
                (self.current.get_next().get_next()).set_prev(self.current)
                (self.current.get_next().set_next(None))
                (self.current.get_next().set_prev(None))


"""
Layer class inherits the attributes of DLLNode and adds some data functionality specific to our Neural Net project
"""


class Layer(DLLNode):

    def __init__(self, num_neurodes=5, my_type=LayerType.HIDDEN, rate=.05):
        self.my_type = my_type
        self.neurodes = []
        self.learning_rate = rate
        super().__init__()

        for neurode in range(num_neurodes):
            self.add_neurode()

    def add_neurode(self):
        self.neurodes.append(FFBPNeurode(self.my_type, self.learning_rate))

    def get_my_neurodes(self):
        return self.neurodes

    def get_my_type(self):
        return self.my_type

    def get_layer_info(self):
        return (self.my_type, len(self.neurodes))


"""
LayerList inherits DoublyLinkedLists and is the controlling object that 
will handle adding and removing layers
"""


class LayerList(DoublyLinkedList):
    def __init__(self, num_inputs, num_outputs, rate):  # rate
        self.input_layer = Layer(num_inputs, LayerType.INPUT, rate)  # rate
        self.output_layer = Layer(num_outputs, LayerType.OUTPUT, rate)  # rate
        super().__init__()
        self.add_to_head(self.input_layer)
        self.reset_cur()
        self.insert_after_cur(self.output_layer)

    def get_input_nodes(self):
        return self.input_layer.get_my_neurodes()

    def get_output_nodes(self):
        return self.output_layer.get_my_neurodes()

    # insert_after_cur does as implied, it connecrs the neurodes and clears and
    # adds neurodes

    def insert_after_cur(self, new_layer):
        # if new_layer is LayerType.OUTPUT | new_layer is LayerType.HIDDEN:
        if new_layer.get_my_type() is LayerType.OUTPUT:
            for neurode in self.current.get_my_neurodes():
                neurode.clear_and_add_output_nodes(new_layer.get_my_neurodes())
            # Each neurode in the current layer has new_layer's neurodes as their output nodes.
            for neurode in new_layer.get_my_neurodes():
                neurode.clear_and_add_input_nodes(self.current.get_my_neurodes())

        else:  # elif(new_layer.get_my_type() is LayerType.HIDDEN): (to be specific)
            # Make sure each neurode in the current layer has new_layer's neurodes as their output nodes.
            for neurode in self.current.get_my_neurodes():
                neurode.clear_and_add_output_nodes(new_layer.get_my_neurodes())
            # Make sure each neurode in new_layer has the current layer's neurodes as their input nodes.
            for neurode in new_layer.get_my_neurodes():
                neurode.clear_and_add_input_nodes(self.current.get_my_neurodes())
            # Make sure each neurode in the next layer has new_layer's neurodes as their input nodes.
            for neurode in self.current.get_next().get_my_neurodes():
                neurode.clear_and_add_input_nodes(new_layer.get_my_neurodes())
            # Make sure each neurode in new_layer has the next layer's neurodes as their output nodes.
            for neurode in new_layer.get_my_neurodes():
                neurode.clear_and_add_output_nodes(self.current.get_next().get_my_neurodes())
        super().insert_after_cur(new_layer)

    def insert_hidden_layer(self, num_neurode, rate):
        hidden_layer = Layer(num_neurode, LayerType.HIDDEN, rate)
        if self.current == self.tail:
            print()
            print("\"Alert: Cannot add layer here\"")
            print()
        else:
            self.insert_after_cur(hidden_layer)

    def remove_after_cur(self):
        for neurode in self.current.get_my_neurodes():
            neurode.clear_and_add_output_nodes(self.current.get_next().get_next().get_my_neurodes())
        for neurode in self.current.get_next().get_next().get_my_neurodes():
            neurode.clear_and_add_input_nodes(self.current.get_my_neurodes())
        super().remove_after_cur()

    def remove_hidden_layer(self):
        if self.current.get_next().get_my_type() is not LayerType.HIDDEN:
            print()
            print("\"Alert: Cannot remove layer here\"")
            print()
        else:
            self.remove_after_cur()

    def next_layer_false(self):
        if self.current.get_next().get_my_type() is not LayerType.HIDDEN:
            return True
        else:
            return False


class NodePositionError(Exception):
    pass


"""
class FFBPNetwork is able to wrap the classes all together, specifically it intoduces the
training and testing functions which graph the results.
"""


class FFBPNetwork:
    class EmptyLayerException(Exception):
        pass

    class EmptySetException(Exception):
        pass

    def __init__(self, num_inputs, num_outputs, rate=.05):
        self.layers = LayerList(num_inputs, num_outputs, rate)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        # final tested-graph #todo make it an attribute for each graph
        self.graph_test_x = []
        self.graph_test_y = []

    def add_hidden_layer(self, num_neurodes=5, rate=.05):
        if num_neurodes < 1:
            raise FFBPNetwork.EmptyLayerException
        self.layers.insert_hidden_layer(num_neurodes, rate)

    def remove_hidden_layer(self):
        return self.layers.remove_hidden_layer()

    def iterate(self):
        return self.layers.iterate()

    def rev_iterate(self):
        return self.layers.rev_iterate()

    def reset_cur(self):
        return self.layers.reset_cur()

    def get_layer_info(self):
        return self.layers.current.get_layer_info()

    def next_layer_false(self):
        return self.layers.next_layer_false()

    @staticmethod
    def show_confusion_matrix(graph_x, graph_y):
        length = len(graph_x)
        multiclass = np.array([[0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]])
        for i in range(length):
            index_y, valueB = max(enumerate(graph_x[i]), key=operator.itemgetter(1))
            index_x, valueA = max(enumerate(graph_y[i]), key=operator.itemgetter(1))
            multiclass[index_y][index_x] += 1
        print(multiclass[1][1])

        print(multiclass)
        class_names = ['virginica', 'versicolor', 'iriscolor']
        fig, ax = plot_confusion_matrix(conf_mat=multiclass,
                                        colorbar=True,
                                        show_absolute=False,
                                        show_normed=True,
                                        class_names=class_names)
        plt.title("Confusion Matrix for Iris data set")
        plt.show()

        print("this is the confusion matrix")

    def show_sin_graph(self, x_axis, y_axis, labeling):
        plt.scatter(x_axis, y_axis)
        plt.title('Sin Wave ' + labeling)
        plt.xlabel('independent output')
        plt.ylabel('dependent input')
        plt.show()

    # train literally trains the network by inputting data and passing it into the input nodes,
    # Raw data is then published indicating the predicted and expected values of the output
    # layer nodes.

    def train(self, data_set: NNData, epochs=1000, verbosity=0, graph=0, order=NNData.Order.SEQUENTIAL):
        # If there are no examples loaded in data_set, raise an EmptySetException.
        if data_set.x is None:
            raise FFBPNetwork.EmptySetException
        RMSE = 0
        temp_rsme = 0
        # loop through a training algorithm for the number of times specified in epochs
        for epoch in range(epochs):
            graph_x = []
            graph_y = []
            #    todo finish making an original train plot
            re = 0
            # Prime the dataset, using random or sequential according to the order passed to train().
            data_set.prime_data(None, order)
            n = len(data_set.train_pool) * len(self.layers.get_output_nodes())
            while not data_set.empty_pool():
                # Use get_one_item from NNData to fetch one example and label, (x,y)
                item = data_set.get_one_item()
                # Present the example to the input layer neurodes so the network can perform its feedforward function
                for i, node in enumerate(self.layers.get_input_nodes()):
                    node.receive_input(None, item[0][i])
                # Calculate the error for this example and add it to the running error
                for count, nodes in enumerate(self.layers.get_output_nodes()):
                    predicted = nodes.get_value()
                    expected = item[1][count]
                    re = re + (1 / n * (predicted - expected) ** 2)
                    # Provides the label to the output layer neurodes so the network can perform its backpropagation
                    nodes.receive_back_input(None, item[1][count])
                    # if verbosity > 1 and epoch % 1000 == 0 and epoch != 0:
                    #     print("sample", item[0][count], "epoch#", epoch, "expected", expected, "produced ", predicted)
                # provides verbosity with given conditions
                if verbosity > 1 and epoch % 1000 == 0:
                    output = []
                    for num, nod in enumerate(self.layers.get_output_nodes()):
                        output.append(nod.get_value())
                    # print(item[0], "expects", item[1], "and produced", output)
                    print(item[0], item[1], output)
                # if verbosity > 1 and epoch % 100 == 0 and epoch != 0:
                if verbosity > 1 and epoch % 100 == 0:
                    output = []
                    for num, nod in enumerate(self.layers.get_output_nodes()):
                        output.append(nod.get_value())
                    # print(item[0], "expects", item[1], "and produced", output)
                    # print(item[0], item[1], output)
                    if (graph == 1):  # ---->temporary graph
                        graph_x.append(item[0])
                        graph_y.append(output)
            if verbosity > 0 and epoch % 100 == 0:
                temp_rsme = sqrt(re)
                print("Epoch#", epoch, "RSME", temp_rsme)
                if (graph == 1):
                    self.show_sin_graph(graph_x, graph_y, "Training")
        # presenting the total root mean square value
        rsme = sqrt(re)
        return rsme

    # test method is like the train method except it tests the rest of the
    # of the data pool that is not training.

    def test(self, data_set: NNData, order=NNData.Order.SEQUENTIAL, graph=0):
        graph_x = []
        graph_y = []
        if data_set.x is None:
            raise FFBPNetwork.EmptySetException
        re = 0  # running error
        # Prime the dataset, using random or sequential according to the order passed to test().
        data_set.prime_data(NNData.Set.TEST, order)
        n = len(data_set.test_pool * len(self.layers.get_output_nodes()))
        # loop through the test set data and test which is similar to training function up above
        while not data_set.empty_pool(NNData.Set.TEST):
            item = data_set.get_one_item(NNData.Set.TEST)
            for i, node in enumerate(self.layers.get_input_nodes()):
                node.receive_input(None, item[0][i])
            for count, nodes in enumerate(self.layers.get_output_nodes()):
                predicted = nodes.get_value()
                expected = item[1][count]
                re = re + (1 / n * (predicted - expected) ** 2)
            output = []
            for num, nod in enumerate(self.layers.get_output_nodes()):
                output.append(nod.get_value())
            print(item[0], item[1], output)
            if (graph == 1):         # temporary graph for just sin
                graph_x.append(item[0])
                graph_y.append(output)
            elif (graph == 2):
                graph_x.append(item[1])
                graph_y.append(output)
            # x predictor expected actual
        if (graph == 1):
            self.show_sin_graph(graph_x, graph_y, "Testing")
        if (graph == 2):
            self.show_confusion_matrix(graph_x, graph_y)
            # show the confusion matrix
        # presenting the total root mean square value
        rsme = sqrt(re)
        print("The Final RSME is", rsme)


# testing script for node iteration


def run_tests():
    network = FFBPNetwork(1, 1)
    network.add_hidden_layer(3)
    network.add_hidden_layer(4)
    network.layers.reset_cur()
    while True:
        print(network.layers.current.get_layer_info())
        if not network.layers.iterate():
            break
    network.layers.reset_cur()
    network.layers.iterate()
    network.layers.remove_hidden_layer()
    network.layers.reset_cur()
    while True:
        print(network.layers.current.get_layer_info())
        if not network.layers.iterate():
            break


class MultiTypeEncorder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, NNData):
            return {'__NNData__': obj.__dict__}
        if isinstance(obj, collections.deque):
            return {'__deque__': list(obj)}
        return obj.__dict__


def multi_type_decoder(obj):
    if "__NNData__" in obj:
        item = obj["__NNData__"]
        return_obj = NNData(item["x"], item["y"], item["train_percentage"])
        trainpool_redeque = item["train_pool"]
        testpool_redeque = item["test_pool"]
        return_obj.train_pool = collections.deque(trainpool_redeque["__deque__"])
        return_obj.test_pool = collections.deque(testpool_redeque["__deque__"])
        return_obj.train_indices = item["train_indices"]
        return_obj.test_indices = item["test_indices"]

        return return_obj
    return obj


def load_sin():
    sin_x = [[0], [0.01], [0.02], [0.03], [0.04], [0.05], [0.06], [0.07], [0.08], [0.09], [0.1], [0.11], [0.12],
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
    sin_y = [[0], [0.00999983333416666], [0.0199986666933331], [0.0299955002024957], [0.0399893341866342],
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
    while True:
        try:
            train = int(input("What data percentage would you like to train?(1-100)"))
            if train > 100 or train < 1:
                print("Needs to be in Range")
                continue
            rate = float(input("Enter the learning rate(0.01-1)?"))
            if rate > 1 or rate < 0.01:
                print("Needs to be in Range")
                continue
        except ValueError:
            print("Invalid Entry")
        else:
            break
    # Load data into a NNData object
    object_data = NNData(sin_x, sin_y, train)  # set up data
    # JSON Encode data into an object data_encoded
    data_encoded = json.dumps(object_data, cls=MultiTypeEncorder)
    # Decode data_encoded into an NNData object called data_decoded
    data_decoded = json.loads(data_encoded, object_hook=multi_type_decoder)

    # Train your network on data_decoded by:
    # setting up network based on examples (independent) and label (dependents) length.
    network_sin = FFBPNetwork(1, 1, rate)
    print("Adding the first hiddenLayer:")

    # adding a particular amount of layer nodes
    ask_hidden_layer_add(network_sin)
    while True:
        try:
            epoch = int(input("How many Epochs: "))
            verb = int(input("Choose verbosity(1 or greater than 2: "))
        except ValueError:
            print("Invalid Choice")
            continue
        else:
            break
    print("Training...")

    # Decoded data is put in and the network is being trained! (phew!)
    network_sin.train(data_decoded, epoch, verb, 1, order=NNData.Order.RANDOM)
    return network_sin, data_decoded


# dataset from https://www.kaggle.com/uciml/iris#Iris.csv

def load_Iris():
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
    # monkey_in = [[0, 1], [2, 2], [0, 1], [2, 1]]
    # monkey_out = [[1], [0], [1], [1]]
    while True:
        try:
            train = int(input("What data percentage would you like to train?(1-100)"))
            if train > 100 or train < 1:
                print("Needs to be in Range")
                continue
            rate = float(input("Enter the learning rate(0.01-1)?"))
            if rate > 1 or rate < 0.01:
                print("Needs to be in Range")
                continue
        except ValueError:
            print("Invalid Entry")
        else:
            break
    # object_data = NNData(monkey_in, monkey_out, train)
    object_data = NNData(Iris_X, Iris_Y, train)
    data_encoded = json.dumps(object_data, cls=MultiTypeEncorder)
    data_decoded = json.loads(data_encoded, object_hook=multi_type_decoder)
    # network_monkey = FFBPNetwork(2, 1, rate)
    network_iris = FFBPNetwork(4, 3, rate)
    print("Adding the first hiddenLayer:")
    ask_hidden_layer_add(network_iris)
    while True:
        try:
            epoch = int(input("How many Epochs: "))
            verb = int(input("Choose verbosity(1 or greater than 2: "))
        except ValueError:
            print("Invalid Choice")
            continue
        else:
            break
    print("Training...")
    graph_data = 0      # temporary for graphing data
    network_iris.train(data_decoded, epoch, verb, graph_data, order=NNData.Order.RANDOM)
    return network_iris, data_decoded


def load_XOR():
    xor_in = [[0, 0], [1, 0], [0, 1], [1, 1]]
    xor_out = [[0], [1], [1], [0]]
    while True:
        try:
            train = int(input("What data percentage would you like to train?(1-100)"))
            if train > 100 or train < 1:
                print("Needs to be in Range")
                continue
            rate = float(input("Enter the learning rate(0.01-1)?"))
            if rate > 1 or rate < 0.01:
                print("Needs to be in Range")
                continue
        except ValueError:
            print("Invalid Entry")
        else:
            break
    object_data = NNData(xor_in, xor_out, train)
    data_encoded = json.dumps(object_data, cls=MultiTypeEncorder)
    data_decoded = json.loads(data_encoded, object_hook=multi_type_decoder)
    network_xor = FFBPNetwork(2, 1, rate)
    print("Adding the first hiddenLayer:")
    ask_hidden_layer_add(network_xor)
    while True:
        try:
            epoch = int(input("How many Epochs: "))
            verb = int(input("Choose verbosity(1 or greater than 2: "))
        except ValueError:
            print("Invalid Choice")
            continue
        else:
            break
    print("Training...")
    network_xor.train(data_decoded, epoch, verb, 0, order=NNData.Order.RANDOM)
    return network_xor, data_decoded


def ask_hidden_layer_add(network: FFBPNetwork):
    add = True
    while add:
        try:
            print()
            num_node = int(input("How many nodes would you like to put in the hidden layer?"))
            h_rate = float(input("What would you like the learning rate to be?"))
            network.add_hidden_layer(num_node, h_rate)
            add = int(input("Would you like to continue adding(1 for yes, 0 for no)"))
            print()
            if add == 1:
                add = True
            elif add == 0:
                add = False
        except ValueError:
            print("Invalid Entry")
            continue

    return network


def initial_menu():
    print()
    print("Welcome to the Neural network")
    print("______________________________")
    print("1 - Sin data")
    print("2 - XOR data")
    print("3 - Iris data")
    print("4 - Quit")


def second_menu():
    print("1- Reload/load Data and Train Network")
    print("2- Test Network")
    print("3- Browse Layers and add/remove Layers")
    print("4- Change back to main menu to change data/exit")


"""
# Menu for interacting with neural network. #Todo, needs to be compartmentalized
"""


def main():
    # object spots
    list_data = [0, 0, 0]
    decode_data = [0, 0, 0]
    done = True
    while done:
        initial_menu()  # starting menu tp pick data
        while True:
            try:
                print()
                load = int(input("Which data would you like to load?"))
                if 1 > load or load > 4:
                    print("Invalid range")
                else:
                    data_num = load - 1
                    break
            except ValueError:
                print("Invalid choice")
        # if load == 1 or load == 2 or load == 3:
        if load <= 3 and load >= 1:
            menu_out = True
            while menu_out:
                print()
                if data_num == 0:
                    print("Sin Data")
                elif data_num == 1:
                    print("Xor Data")
                elif data_num == 2:
                    print("Iris Data (Fake data)")
                second_menu()
                while True:
                    try:
                        print()
                        response = int(input("Enter a option (remember to load data first):"))
                    except ValueError:
                        print("Invalid Choice")
                        continue
                    if response < 1 or response > 4:
                        print("Needs to be in Range")
                        continue
                    else:
                        break
                if response == 1:
                    if data_num == 0:  # 1-take sin data
                        sin_data, sin_decoded = load_sin()
                        list_data[data_num] = sin_data
                        decode_data[data_num] = sin_decoded
                    elif data_num == 1:  # 2-load xor_data
                        xor_data, xor_decoded = load_XOR()
                        list_data[data_num] = xor_data
                        decode_data[data_num] = xor_decoded
                    elif data_num == 2:  # 3-load Iris_data
                        iris_data, iris_decoded = load_Iris()
                        list_data[data_num] = iris_data
                        decode_data[data_num] = iris_decoded
                    else:
                        print("no Data loaded")
                elif response == 2:  # Test the data

                    if list_data[data_num] == 0:
                        print("no data loaded")
                        continue
                    print("Testing Network")
                    if (0 == data_num):     # this is temporary graphing for
                        graph = 1
                    elif(2 == data_num):
                        graph = 2
                    else:
                        graph = 0
                    list_data[data_num].test(decode_data[data_num], NNData.Order.RANDOM, graph)

                elif response == 3:  # browse the NNT

                    if list_data[data_num] == 0:
                        print("no data loaded")
                        continue
                    browse = True

                    while browse:
                        print("Current layer is:")
                        print("-----------------")
                        print("LayerType, number of nodes in layer")
                        print(list_data[data_num].get_layer_info())
                        print()
                        print("1 - Add hidden layer(s)")
                        print("2 - remove hidden layer(s)")
                        print("3 - iterate forward a layer")
                        print("4 - iterate backward a layer")
                        print("5 - reset to input layer")
                        print("6 - printing the network layers (resets current cursor to input layer)")
                        print("7 - Return to Menu")

                        while True:
                            try:
                                print()
                                choice = int(input("Please enter an option: "))
                                if choice < 1 or choice > 7:
                                    print("Invalid range")
                                else:
                                    break
                            except ValueError:
                                print("Invalid choice")
                        if choice == 1:
                            ask_hidden_layer_add(list_data[data_num])
                        elif choice == 2:
                            list_data[data_num].remove_hidden_layer()
                        elif choice == 3:
                            list_data[data_num].iterate()
                        elif choice == 4:
                            list_data[data_num].rev_iterate()
                        elif choice == 5:
                            list_data[data_num].reset_cur()
                        elif choice == 6:
                            list_data[data_num].reset_cur()
                            print()
                            print("The Layers and their nodes:")
                            while True:
                                print(list_data[data_num].get_layer_info())
                                if not list_data[data_num].layers.check_iterate():  # checks for last node
                                    list_data[data_num].reset_cur()
                                    print()
                                    break
                        elif choice == 7:
                            break
                elif response == 4:
                    break
        elif load == 4:
            break


if __name__ == "__main__":
    main()

"""

Welcome to the Neural network
______________________________
1 - Sin data
2 - XOR data
3 - Monkey data
4 - Quit

Which data would you like to load?1

Sin Data
1- Reload/load Data and Train Network
2- Test Network
3- Browse Layers and add/remove Layers
4- Change back to main menu to change data/exit

Enter a option (remember to load data first):1
What data percentage would you like to train?(1-100)50
Enter the learning rate(0.01-1)?.05
Adding the first hiddenLayer:

How many nodes would you like to put in the hidden layer?3
What would you like the learning rate to be?.05
Would you like to continue adding(1 for yes, 0 for no)1


How many nodes would you like to put in the hidden layer?.05
Invalid Entry

How many nodes would you like to put in the hidden layer?1
What would you like the learning rate to be?.05
Would you like to continue adding(1 for yes, 0 for no)0

How many Epochs: 1001
Choose verbosity(1 or greater than 2: 2
Training...
Epoch# 0 RSME 0.3096712890054866
Epoch# 100 RSME 0.3084187846784271
Epoch# 200 RSME 0.3074106152965534
Epoch# 300 RSME 0.30613974504668295
Epoch# 400 RSME 0.3032643750770099
Epoch# 500 RSME 0.2940862688839347
Epoch# 600 RSME 0.276261441193989
Epoch# 700 RSME 0.2597375355333373
Epoch# 800 RSME 0.24536820882949287
Epoch# 900 RSME 0.23229592787542414
[0.21] [0.2084598998461] [0.4696947650478097]
[0.89] [0.777071747526824] [0.6438585399338235]
[0.51] [0.488177246882907] [0.5714646532253576]
[1.44] [0.991458348191686] [0.6848539619178465]
[0.27] [0.266731436688831] [0.4939034968786819]
[0.78] [0.70327941920041] [0.6286948896838406]
[1.33] [0.971148377921045] [0.6809079360570756]
[1.19] [0.928368967249167] [0.6745446944521074]
[1.56] [0.999941720229966] [0.6924385355420318]
[1.51] [0.998152472497548] [0.692498926992866]
[0.41] [0.398609327984423] [0.5489235839209851]
[0.55] [0.522687228930659] [0.5879075367184853]
[1.41] [0.98710010101385] [0.6892602624573279]
[0.57] [0.539632048733969] [0.5941338292413999]
[1.34] [0.973484541695319] [0.6874635062964425]
[0.86] [0.757842562895277] [0.6489545387511655]
[1.3] [0.963558185417193] [0.6875944927477549]
[1] [0.841470984807897] [0.6673214734589876]
[0.13] [0.129634142619695] [0.44627336090965347]
[1.55] [0.999783764189357] [0.6980053099532064]
[0.72] [0.659384671971473] [0.630043378782343]
[0.65] [0.60518640573604] [0.6166607688146027]
[1.54] [0.999525830605479] [0.6993127822186701]
[0.14] [0.139543114644236] [0.45213471811486095]
[1.18] [0.92460601240802] [0.6827924711314223]
[0.12] [0.119712207288919] [0.4429574036730584]
[0] [0] [0.3873626676037323]
[0.24] [0.237702626427135] [0.4897709445429568]
[0.32] [0.314566560616118] [0.51877326832876]
[0.79] [0.710353272417608] [0.6365793117034246]
[0.23] [0.227977523535188] [0.48346334583578165]
[0.35] [0.342897807455451] [0.5270771816859509]
[0.05] [0.0499791692706783] [0.40371885898468457]
[0.74] [0.674287911628145] [0.6241995144474792]
[0.45] [0.43496553411123] [0.5567045987785713]
[1.38] [0.98185353037236] [0.6844318421876551]
[0.48] [0.461779175541483] [0.5664204924519929]
[0.6] [0.564642473395035] [0.596606356755727]
[0.38] [0.370920469412983] [0.5345902341735379]
[1.57] [0.999999682931835] [0.6909338814341746]
[0.2] [0.198669330795061] [0.46780584035967143]
[0.82] [0.731145829726896] [0.6365464078290268]
[0.92] [0.795601620036366] [0.6504154530712006]
[1.37] [0.979908061398614] [0.6852964404867382]
[1.29] [0.960835064206073] [0.6829995030707273]
[0.77] [0.696135238627357] [0.6334120540146336]
[0.36] [0.35227423327509] [0.5320403144080166]
[0.04] [0.0399893341866342] [0.40056771036274047]
[0.9] [0.783326909627483] [0.6493917081254909]
[0.63] [0.58914475794227] [0.6050573117872503]
[0.37] [0.361615431964962] [0.5330700097133083]
[0.22] [0.218229623080869] [0.47600100851230087]
[0.69] [0.636537182221968] [0.6146126408382998]
[1.04] [0.862404227243338] [0.6627030872204546]
[0.16] [0.159318206614246] [0.45084984349413787]
[0.66] [0.613116851973434] [0.6082535355876586]
[0.53] [0.505533341204847] [0.5780416670127048]
[0.85] [0.751280405140293] [0.6401534673946689]
[0.56] [0.531186197920883] [0.585906071074005]
[0.8] [0.717356090899523] [0.6331627612085644]
[1.4] [0.98544972998846] [0.685376056671576]
[0.25] [0.247403959254523] [0.4881424623055652]
[0.18] [0.179029573425824] [0.4582736497915138]
[1.21] [0.935616001553386] [0.6743047028451629]
[0.71] [0.651833771021537] [0.6185345989663995]
[0.31] [0.305058636443443] [0.5095683056157545]
[0.08] [0.0799146939691727] [0.4140429287954418]
[0.83] [0.737931371109963] [0.63538931695565]
[1.06] [0.872355482344986] [0.6623247287462858]
[0.15] [0.149438132473599] [0.44444674118718985]
[1.23] [0.942488801931697] [0.6743728764718532]
[0.93] [0.801619940883777] [0.6501218723441626]
[0.02] [0.0199986666933331] [0.3880146941563046]
[0.47] [0.452886285379068] [0.5597080931708395]
[0.88] [0.770738878898969] [0.6422789590747071]
[0.58] [0.548023936791874] [0.5888938063388334]
[0.26] [0.257080551892155] [0.48806968010994634]
[0.46] [0.44394810696552] [0.5551837665338593]
[1.07] [0.877200504274682] [0.6610798707473605]
Epoch# 1000 RSME 0.22019223583572015

Sin Data
1- Reload/load Data and Train Network
2- Test Network
3- Browse Layers and add/remove Layers
4- Change back to main menu to change data/exit

Enter a option (remember to load data first):2
Testing Network
[0.01] [0.00999983333416666] [0.38102506346753046]
[0.03] [0.0299955002024957] [0.38990750528027146]
[0.06] [0.0599640064794446] [0.40320133832650423]
[0.07] [0.0699428473375328] [0.40761632593632036]
[0.09] [0.089878549198011] [0.41640964493606913]
[0.1] [0.0998334166468282] [0.4207840105968226]
[0.11] [0.109778300837175] [0.4251409314618225]
[0.17] [0.169182349066996] [0.45081387150029917]
[0.19] [0.188858894976501] [0.4591522688280317]
[0.28] [0.276355648564114] [0.4949170197598044]
[0.29] [0.285952225104836] [0.49869044674233864]
[0.3] [0.29552020666134] [0.5024203842870586]
[0.33] [0.324043028394868] [0.5133422668368127]
[0.34] [0.333487092140814] [0.5168915687089162]
[0.39] [0.380188415123161] [0.5339327234489821]
[0.4] [0.389418342308651] [0.5371977105677672]
[0.42] [0.40776045305957] [0.5435829186041005]
[0.43] [0.416870802429211] [0.5467029833144672]
[0.44] [0.425939465066] [0.5497746613119423]
[0.49] [0.470625888171158] [0.56440987433454]
[0.5] [0.479425538604203] [0.5671933460443108]
[0.52] [0.496880137843737] [0.5726185608764942]
[0.54] [0.514135991653113] [0.5778568508475794]
[0.59] [0.556361022912784] [0.5901542249737399]
[0.61] [0.572867460100481] [0.5947628471585452]
[0.62] [0.581035160537305] [0.5970025526702064]
[0.64] [0.597195441362392] [0.6013552963405263]
[0.67] [0.62098598703656] [0.6075756985388893]
[0.68] [0.628793024018469] [0.6095690243488208]
[0.7] [0.644217687237691] [0.6134388169399225]
[0.73] [0.666869635003698] [0.6189598947360354]
[0.75] [0.681638760023334] [0.6224582395136908]
[0.76] [0.688921445110551] [0.6241544299011378]
[0.81] [0.724287174370143] [0.6321285675422027]
[0.84] [0.744643119970859] [0.636528089119562]
[0.87] [0.764328937025505] [0.6406573285723423]
[0.91] [0.78950373968995] [0.6457678012160427]
[0.94] [0.807558100405114] [0.64932256937689]
[0.95] [0.813415504789374] [0.6504572200223014]
[0.96] [0.819191568300998] [0.6515675525701781]
[0.97] [0.82488571333845] [0.6526540475289185]
[0.98] [0.83049737049197] [0.6537171784535534]
[0.99] [0.836025978600521] [0.6547574118819061]
[1.01] [0.846831844618015] [0.6567710170241158]
[1.02] [0.852108021949363] [0.6577452863338644]
[1.03] [0.857298989188603] [0.6586984532968567]
[1.05] [0.867423225594017] [0.6605431967515251]
[1.08] [0.881957806884948] [0.6631625855145951]
[1.09] [0.886626914449487] [0.6639979382669527]
[1.1] [0.891207360061435] [0.6648150557314512]
[1.11] [0.895698685680048] [0.6656143193082509]
[1.12] [0.900100442176505] [0.6663961034828568]
[1.13] [0.904412189378826] [0.6671607758831113]
[1.14] [0.908633496115883] [0.6679086973406023]
[1.15] [0.912763940260521] [0.6686402219560766]
[1.16] [0.916803108771767] [0.6693556971684749]
[1.17] [0.920750597736136] [0.6700554638272236]
[1.2] [0.932039085967226] [0.6720638237305083]
[1.22] [0.939099356319068] [0.6733301469648345]
[1.24] [0.945783999449539] [0.6745412743442318]
[1.25] [0.948984619355586] [0.6751268783009621]
[1.26] [0.952090341590516] [0.6756995579592613]
[1.27] [0.955100855584692] [0.6762595927488362]
[1.28] [0.958015860289225] [0.6768072564367476]
[1.31] [0.966184951612734] [0.6783786755171746]
[1.32] [0.968715100118265] [0.6788794823769699]
[1.35] [0.975723357826659] [0.6803163604144346]
[1.36] [0.977864602435316] [0.6807742611930561]
[1.39] [0.983700814811277] [0.6820879645102541]
[1.42] [0.98865176285172] [0.6833161877808167]
[1.43] [0.990104560337178] [0.6837075620198487]
[1.45] [0.992712991037588] [0.6844644385842065]
[1.46] [0.993868363411645] [0.6848303190949205]
[1.47] [0.994924349777581] [0.685188073143762]
[1.48] [0.99588084453764] [0.6855378798356371]
[1.49] [0.996737752043143] [0.6858799144215556]
[1.5] [0.997494986604054] [0.68621434837538]
[1.52] [0.998710143975583] [0.6868610818490946]
[1.53] [0.999167945271476] [0.6871737061059551]
The Final RSME is 0.22927618972760816

Sin Data
1- Reload/load Data and Train Network
2- Test Network
3- Browse Layers and add/remove Layers
4- Change back to main menu to change data/exit

Enter a option (remember to load data first):3
Current layer is:
-----------------
LayerType, number of nodes in layer
(<LayerType.INPUT: 0>, 1)

1 - Add hidden layer(s)
2 - remove hidden layer(s)
3 - iterate forward a layer
4 - iterate backward a layer
5 - reset to input layer
6 - printing the network layers (resets current cursor to input layer)
7 - Return to Menu

Please enter an option: 1

How many nodes would you like to put in the hidden layer?2
What would you like the learning rate to be?.05
Would you like to continue adding(1 for yes, 0 for no)0

Current layer is:
-----------------
LayerType, number of nodes in layer
(<LayerType.INPUT: 0>, 1)

1 - Add hidden layer(s)
2 - remove hidden layer(s)
3 - iterate forward a layer
4 - iterate backward a layer
5 - reset to input layer
6 - printing the network layers (resets current cursor to input layer)
7 - Return to Menu

Please enter an option: 3
Current layer is:
-----------------
LayerType, number of nodes in layer
(<LayerType.HIDDEN: 2>, 2)

1 - Add hidden layer(s)
2 - remove hidden layer(s)
3 - iterate forward a layer
4 - iterate backward a layer
5 - reset to input layer
6 - printing the network layers (resets current cursor to input layer)
7 - Return to Menu

Please enter an option: 3
Current layer is:
-----------------
LayerType, number of nodes in layer
(<LayerType.HIDDEN: 2>, 1)

1 - Add hidden layer(s)
2 - remove hidden layer(s)
3 - iterate forward a layer
4 - iterate backward a layer
5 - reset to input layer
6 - printing the network layers (resets current cursor to input layer)
7 - Return to Menu

Please enter an option: 3
Current layer is:
-----------------
LayerType, number of nodes in layer
(<LayerType.HIDDEN: 2>, 3)

1 - Add hidden layer(s)
2 - remove hidden layer(s)
3 - iterate forward a layer
4 - iterate backward a layer
5 - reset to input layer
6 - printing the network layers (resets current cursor to input layer)
7 - Return to Menu

Please enter an option: 3
Current layer is:
-----------------
LayerType, number of nodes in layer
(<LayerType.OUTPUT: 1>, 1)

1 - Add hidden layer(s)
2 - remove hidden layer(s)
3 - iterate forward a layer
4 - iterate backward a layer
5 - reset to input layer
6 - printing the network layers (resets current cursor to input layer)
7 - Return to Menu

Please enter an option: 3
"Alert: Unable to Iterate"
Current layer is:
-----------------
LayerType, number of nodes in layer
(<LayerType.OUTPUT: 1>, 1)

1 - Add hidden layer(s)
2 - remove hidden layer(s)
3 - iterate forward a layer
4 - iterate backward a layer
5 - reset to input layer
6 - printing the network layers (resets current cursor to input layer)
7 - Return to Menu

Please enter an option: 5
Current layer is:
-----------------
LayerType, number of nodes in layer
(<LayerType.INPUT: 0>, 1)

1 - Add hidden layer(s)
2 - remove hidden layer(s)
3 - iterate forward a layer
4 - iterate backward a layer
5 - reset to input layer
6 - printing the network layers (resets current cursor to input layer)
7 - Return to Menu

Please enter an option: 2
Current layer is:
-----------------
LayerType, number of nodes in layer
(<LayerType.INPUT: 0>, 1)

1 - Add hidden layer(s)
2 - remove hidden layer(s)
3 - iterate forward a layer
4 - iterate backward a layer
5 - reset to input layer
6 - printing the network layers (resets current cursor to input layer)
7 - Return to Menu

Please enter an option: 2
Current layer is:
-----------------
LayerType, number of nodes in layer
(<LayerType.INPUT: 0>, 1)

1 - Add hidden layer(s)
2 - remove hidden layer(s)
3 - iterate forward a layer
4 - iterate backward a layer
5 - reset to input layer
6 - printing the network layers (resets current cursor to input layer)
7 - Return to Menu

Please enter an option: 4
"Alert: Unable to Rev-iterate"
Current layer is:
-----------------
LayerType, number of nodes in layer
(<LayerType.INPUT: 0>, 1)

1 - Add hidden layer(s)
2 - remove hidden layer(s)
3 - iterate forward a layer
4 - iterate backward a layer
5 - reset to input layer
6 - printing the network layers (resets current cursor to input layer)
7 - Return to Menu

Please enter an option: 6

The Layers and their nodes:
(<LayerType.INPUT: 0>, 1)
(<LayerType.HIDDEN: 2>, 3)
(<LayerType.OUTPUT: 1>, 1)

Current layer is:
-----------------
LayerType, number of nodes in layer
(<LayerType.INPUT: 0>, 1)

1 - Add hidden layer(s)
2 - remove hidden layer(s)
3 - iterate forward a layer
4 - iterate backward a layer
5 - reset to input layer
6 - printing the network layers (resets current cursor to input layer)
7 - Return to Menu

Please enter an option: 7

Sin Data
1- Reload/load Data and Train Network
2- Test Network
3- Browse Layers and add/remove Layers
4- Change back to main menu to change data/exit

Enter a option (remember to load data first):4

Welcome to the Neural network
______________________________
1 - Sin data
2 - XOR data
3 - Monkey data
4 - Quit

Which data would you like to load?4

"""
