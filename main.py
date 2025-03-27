import argparse;

from UI import UI
from perceptron import Perceptron
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Perceptron")

    parser.add_argument("a", type=float, help="Alpha parameter (a)")
    parser.add_argument("train_file", type=str, help="Train set file .csv")
    parser.add_argument("test_file", type=str, help="Test set file .csv")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    perceptron = Perceptron(args.a)
    trainer = Trainer(perceptron, args.train_file, args.test_file)
    perceptron.initialize_weights(len(trainer.train_set[0][0]))

    trainer.train()
    acc = trainer.accuracy()

    print(f"Total accuracy: {acc:.2%}")

    ui = UI(perceptron)

    if args.interactive:
        ui.interactive_mode()
    else:
        ui.draw(trainer)


if __name__ == "__main__":
    main()