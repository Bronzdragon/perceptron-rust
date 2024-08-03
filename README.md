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

Training data has to be provided in annotated sets. An n-dimensional vector (list of numbers) and either a `1` or `-1`, combined as a tuple. E.g.:
```python
data = [
    ([1, 0, 0], -1),
    ([0, 1, 0],  1),
    ([0, 0, 1], -1),
]
```
You can provide the data right as you create the Perceptron class, or you can provide it later (using the `.add_samples(samples)`, `.replace_samples(samples)` and `.clear_samples(samples)` methods).

The last method available to you is the `.train(iterations)` which starts training for `iterations` number of iterations. You can call `.train()` multiple times, and it'll continue from where it left off last.

Have a look at `example.py` for an in-situ example.

## Development
This package is developed using [Maturin](https://github.com/PyO3/maturin). There are other alternatives available. If you are in doubt, please reach out.

The steps are roughly as follows:
1. Install Maturin
2. Create a virtual environment for Python, and activate it.
3. Run `maturin develop` to automatically build and install the package.
4. Start Python (`perceptron` will be available for import).
