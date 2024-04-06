import numpy as np

class ResponseSampler:
    def __init__(self, population, seed=42):
        """
        Initializes the ResponseSampler with a given population and random seed.

        :param population: A numpy array representing the population data.
        :param seed: An integer seed for the random number generator.
        """
        self.population = population
        self.rand_gen = np.random.default_rng(seed)
        self.n_repeat = len(population[:, 0]) // len(np.unique(population[:, 0]))

    def sample_response(self, x_sams):
        """
        Generates samples from the population based on the input x values.

        :param x_sams: A list or array of x indices for which to generate response samples.
        :return: A generator yielding tuples of (x, y) samples.
        """
        for x in x_sams:
            index = self.rand_gen.integers(0, self.n_repeat)
            yield (self.population[x * self.n_repeat + index, 0], self.population[x * self.n_repeat + index, 1])

    def get_samples(self, x_sams):
        """
        Returns an array of sampled responses for the given x indices.

        :param x_sams: A list or array of x indices for which to generate response samples.
        :return: A numpy array of (x, y) samples.
        """
        return np.array(list(self.sample_response(x_sams)))
