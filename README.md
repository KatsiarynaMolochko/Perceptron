# Perceptron Classifier (Iris Dataset)

This project implements a binary Perceptron classifier using Python. It trains and evaluates the model on a filtered version of the Iris dataset (containing only two classes). The program includes training, prediction, accuracy evaluation, interactive classification, and graphical visualization of per-class accuracy.

---

## 🔧 Features

- Custom Perceptron model implementation
- Command-line training and testing
- Interactive classification mode
- Visualization of per-class accuracy
- Label encoding / decoding
- Vector expansion with integrated threshold


---

## 📁 Project Structure
```
.
├── main.py # Entry point with CLI interface 
├── perceptron.py # Perceptron logic 
├── trainer.py # Dataset loading, training, evaluation 
├── UI.py # Interactive mode and visualization
├── iris_train_filtered.csv 
├── iris_test_filtered.csv
```

---

## 📄 CSV Format

- Files must be `;` (semicolon) separated
- The **last column** should be the **class label**
- The **first row** should contain a header

**Example `iris_train_filtered.csv`:**
```
SepalLength;SepalWidth;PetalLength;PetalWidth;Species
5.1;3.5;1.4;0.2;Iris-setosa
6.2;3.4;5.4;2.3;Iris-virginica
```

---

## 🚀 How to Run

### Train and test:
```bash
python main.py <alpha> <train_file.csv> <test_file.csv>
```

### Example:
```
python main.py 0.1 iris_train_filtered.csv iris_train_filtered.csv
```
### Interactive mode:
```
python main.py 0.1 iris_train_filtered.csv iris_train_filtered.csv --interactive
```
You can enter vectors like:
```
5.1;3.5;1.4;0.2
```
### 📊 Accuracy Plot
If --interactive is not passed, the program will automatically display a bar chart showing classification accuracy per class using matplotlib.

### ⚠️ Notes
Works only with two-class Iris dataset (e.g. setosa vs versicolor)

Input vectors must match feature count (4 values separated by ;)

Perceptron weights include threshold as an extra feature

---

Made for the PJATK NAI course.

---

