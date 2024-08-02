# Percentron-rs

A basic implementation of the Perceptron algorithm, implemented using Rust, for Python.


## Usage
Install the Perceptron-rs package, then import it under the name `perceptron` as follows:

```python
from perceptron import Perceptron

dimensions = 2
samples = [([1,1], -1), ([-1, 1], +1), ([1, -1], -1), ([-1, -1], +1)]

p = Perceptron(dimensions, samples)
p.train(iterations=10)
```

## Development
This package is developed using [Maturin](https://github.com/PyO3/maturin). There are other alternatives available. If you are in doubt, please reach out.

The steps are roughly as follows:
1. Install Maturin
2. Create a virtual environment for Python, and activate it.
3. Run `maturin develop` to automatically build and install the package.
4. Start Python (`perceptron` will be available for import).
