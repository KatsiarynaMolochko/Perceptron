


class Trainer:
    def __init__(self, perceptron, train_set_file, test_set_file):
        self.perceptron = perceptron
        self.train_set = self.load_data(train_set_file)
        self.test_set = self.load_data(test_set_file)

    def load_data(self, train_set, skip_header = True):
        dataset = []
        with open(train_set, 'r') as file:
            lines = file.readlines()
            if skip_header:
                lines = lines[1:]

            for line in lines:
                row = line.strip().split(',')
                label = row[-1]
                vector = [float(x) for x in row[:-1]]
                # vector.append(-1.0)
                dataset.append((vector, label))
        self.perceptron.build_label_dict(dataset)
        return dataset


    def train(self):
        for vector, label in self.train_set:
            expanded = self.perceptron.expand_input(vector)
            self.perceptron.learn(expanded, label)


    def test(self):
        correct = 0
        total = 0
        for vector, label in self.test_set:
            expanded = self.perceptron.expand_input(vector)
            predicted = self.perceptron.compute(expanded)
            encoded = self.perceptron.encode_labels(label)
            if predicted == encoded:
                correct += 1
            total += 1
            accuracy = correct / total
        return accuracy

    def accuracy(self):
        return self.test()

    def accuracy_by_label(self):
        correct = {}
        total = {}

        for vector, label in self.test_set:
            expanded = self.perceptron.expand_input(vector)
            predicted = self.perceptron.compute(expanded)
            encoded = self.perceptron.encode_labels(label)

            if label not in correct:
                correct[label] = 0
                total[label] = 0

            total[label] += 1
            if predicted == encoded:
                correct[label] += 1

        accuracies = {
            label: correct[label] / total [label]
            for label in total
        }
        return accuracies


