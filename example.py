from perceptron import Perceptron
import random
import math

def main():
    dimensions = 3
    sample_count = 100

    # For demonstration purposes, we generate a random theta and a bunch of data that matches that theta.
    # This guarantees our data is linearly separable.
    goal_theta = normalize(random_vector(dimensions))
    samples = generate_samples(goal_theta, dimensions, sample_count)

    p = Perceptron(dimensions)
    p.add_training_samples(samples)

    actual_theta = p.train(100)

    print("Generated model:", actual_theta)


## Helper functions
def vector_length(v):
    return math.sqrt(sum(x * x for x in v))

def normalize(v):
    length = vector_length(v)
    return [x / length for x in v]

def signed_distance(point, hyperplane):
    dot_prod = sum(a * b for a, b in zip(hyperplane, point))
    normal = vector_length(hyperplane)
    return dot_prod / normal

def vector_distance(vector1, vector2):
    # A.k.a. Euclidean distance
    return math.sqrt(sum((v1 - v2) ** 2 for v1, v2 in zip(vector1, vector2)))

def random_vector(size = 2,):
    return [random.uniform(-1, 1) for _ in range(size)]

def generate_samples(goal_theta, dimensions, count = 1):
    samples = []
    for _ in range(count):
        point = random_vector(dimensions)
        label = -1 if signed_distance(point, goal_theta) < 0 else 1
        samples.append((point, label))
    return samples


if __name__ == "__main__":
    main()
