from perceptron import Perceptron
import random
import math

## Helper functions

def sign(num):
    return -1 if num < 0 else 1

def signed_distance(point, hyperplane):
    dot_prod = sum(a * b for a, b in zip(hyperplane, point))
    normal = vector_length(hyperplane)
    return dot_prod / normal

def vector_distance(vector1, vector2):
    # A.k.a. Euclidean distance
    return math.sqrt(sum((v1 - v2) ** 2 for v1, v2 in zip(vector1, vector2)))

def vector_length(v):
    return math.sqrt(sum(x * x for x in v))

def normalize(v):
    length = vector_length(v)
    return [x / length for x in v]

def get_random_vector(size = 2,):
    return [random.uniform(-1, 1) for _ in range(size)]

def generate_samples(goal_theta, count = 1):
    samples = []
    for _ in range(count):
        point = get_random_vector(dimensions)
        label = sign(signed_distance(point, goal_theta))
        samples.append((point, label))
    return samples


if __name__ == "__main__":

    dimensions = 3
    sample_count = 100

    # For demonstration purposes, we generate a random theta and a bunch of data that matches that theta.
    # This guarantees our data is linearly separable.
    goal_theta = normalize(get_random_vector(dimensions))
    samples = generate_samples(goal_theta, 100)

    p = Perceptron(dimensions)
    p.add_training_samples(samples)

    actual_theta = p.train(100)

    print(actual_theta)
