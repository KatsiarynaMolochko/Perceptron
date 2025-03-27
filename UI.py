from InvalidInputError import InvalidInputError
from matplotlib import pyplot as plt

class UI:
    def __init__(self, perceptron):
        self.perceptron = perceptron

    def interactive_mode(self):
        num_features = len(self.perceptron.weights) - 1
        print("Enter vector using ';' as separator or exit for escaping")

        while True:
            user_input = input("Enter vector: ")

            if user_input.lower() == "exit":
                break

            try:
                user_vector = [float(x) for x in user_input.split(";")]

                if len(user_vector) != num_features:
                    raise InvalidInputError(f"Error! Expected {num_features} arguments, received {len(user_vector)}.")

                expanded = self.perceptron.expand_input(user_vector)
                predicted = self.perceptron.compute(expanded)
                decoded = self.perceptron.decode_label(predicted)

                print(f"Predicted label: {decoded}")

            except ValueError:
                print("Check if attributes are numerical and separated with ';' ")
            except InvalidInputError as e:
                print(f"{e}")


    def draw(self, trainer):
        acc_by_label = trainer.accuracy_by_label()
        labels = list(acc_by_label.keys())
        values = list(acc_by_label.values())

        plt.bar(labels, values)
        plt.ylim(0, 1)
        plt.title("Accuracy of classification by class")
        plt.xlabel("Class")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.show()

