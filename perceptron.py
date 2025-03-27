import random
from collections import Counter
from matplotlib import pyplot as plt
import numpy as np

class Perceptron:
    def __init__(self, alpha):
        self.alpha = alpha
        self.weights = []
        self.threshold = 0
        self.beta = 0
        self.labels = {}
        self.inverse_labels = {}

    def initialize_weights(self, num_features):
        self.weights = np.random.uniform(-5, 5, num_features)
        self.weights = np.append(self.weights, self.threshold)

    def expand_input(self, vector):
        return vector + [-1.0]

    def build_label_dict(self, dataset):
        labels = {}
        inverse_labels = {}
        label_set = set()

        for vector, label in dataset:
            label_set.add(label)

        for i, label in enumerate(label_set):
            labels[label] = i
            inverse_labels[i] = label

        self.labels = labels
        self.inverse_labels = inverse_labels


    def encode_labels(self, label_name):
        return self.labels[label_name]

    def decode_label(self, label_id):
        return self.inverse_labels[label_id]


    def calculate_net(self, input_vector):
    # net = X' * W'
        net = 0
        for i in range(len(input_vector)):
            net += self.weights[i] * input_vector[i]
        return net

    def compute(self, input_vector):
        net = self.calculate_net(input_vector)
        if net >= 0: return 1
        else: return 0

    def learn(self, input, right_decision):
        predicted_class = self.compute(input)
        true_class = self.encode_labels(right_decision)
        if (predicted_class != true_class):
            for i in range(len(self.weights)):
                self.weights[i] = self.weights[i] + (true_class - predicted_class) * self.alpha * input[i]
            # self.weights = self.weights + (predicted_class - right_decision) * self.alpha * input



