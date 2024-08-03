# Perceptron-Rust

A basic implementation of the Perceptron algorithm, implemented using Rust, for Python.

## Installation
This package is available via PyPI, meaning you can use pip to install it.
```sh
pip install perceptron-rust
```

## Usage
Install the Perceptron-rust package, then import it under the name `perceptron-rust`. A short example:
```python
from perceptron_rust import Perceptron

dimensions = 2
samples = [([1,1], -1), ([-1, 1], +1), ([1, -1], -1), ([-1, -1], +1)]

p = Perceptron(dimensions, samples)
p.train(iterations=10)
```

Have a look at the example file [example.py](example.py) for in-situ usage.

### Training Data format
Training data has to be provided in annotated sets. The data exists as a list, and each element is a tuple. Data, then the label. The data is a list of numbers (with the same dimension as you initialized the Perceptron with), and the label is either a `1` or `-1`. E.g.:
```python
data = [
    ([1, 0, 0], -1),
    ([0, 1, 0],  1),
    ([0, 0, 1], -1),
]
```

### Methods available
```py
p.add_samples(data)
```
Appends samples to the currently stored set. Make sure they follow the data format described above.

```python
p.clear_samples()
```
Removes all samples already stored. (If some training has occurred, this will finalize the training.)

```python
p.replace_samples(samples)
```
Clears all existing samples and adds the provided samples. Make sure these follow the data format described above.

```py
p.train(iterations, should_normalize=True)
```
Trains for number of iterations provided. Calling this method multiple times will train it in steps. Once training has started, you cannot change the samples any more. You can normalize the output, but this will finalize the model. It cannot be trained further after this.

## Development
This package is developed using [Maturin](https://github.com/PyO3/maturin). There are other alternatives available. If you are in doubt, please reach out.

The steps are roughly as follows:
1. Install Maturin
2. Create a virtual environment for Python, and activate it.
3. Run `maturin develop` to automatically build and install the package.
4. Start Python (`perceptron` will be available for import).
