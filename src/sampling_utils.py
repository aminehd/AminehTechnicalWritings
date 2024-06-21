import numpy as np

def get_population(file_path):
        files = np.load(file_path)
        return files['arr_0']
        
        
class PopulationSampler:
    def __init__(self, population, seed=42):
        """
        Initializes the ResponseSampler with a given population and random seed.

        :param population: A numpy array representing the population data.
        :param seed: An integer seed for the random number generator.
        """
        self.population = population
        self.rand_gen = np.random.default_rng(seed)
        self.n_repeat = len(population[:, 0]) // len(np.unique(population[:, 0]))
        self.xs = []

    def sample_random_response(self, x_sams=None):
        """
        Generates samples from the population based on the input x values.

        :param x_sams: A list or array of x indices for which to generate response samples.
        :return: A generator yielding tuples of (x, y) samples.

        """

        if(not x_sams):
            x_sams = self.xindeces
            
        for x in x_sams:
            index = self.rand_gen.integers(0, self.n_repeat)
            yield (self.population[x * self.n_repeat + index, 0], self.population[x * self.n_repeat + index, 1])

    def sample_lowest_responses(self, x_sams=None):
        """
        Generates samples of the lowest responses from the population based on the input x values.

        :param x_sams: A list or array of x indices for which to generate lowest response samples.
        :return: A generator yielding tuples of (x, y) samples with the lowest y values for each x.
        """
        if x_sams is None:
            x_sams = self.xindeces

        for x in x_sams:
            start_index = x * self.n_repeat
            end_index = start_index + self.n_repeat
            subset = self.population[start_index:end_index, :]
            min_response_index = subset[:, 1].argmin()
            yield (subset[min_response_index, 0], subset[min_response_index, 1])

def sample_highest_responses(self, x_sams=None):
        """
        Generates samples of the highest responses from the population based on the input x values.
        
        :param x_sams: A list or array of x indices for which to generate lowest response samples.
        :return: A generator yielding tuples of (x, y) samples with the highest y values for each x.
        """
        if x_sams is None:
            x_sams = self.xindeces
        
        for x in x_sams:
            start_index = x * self.n_repeat
            end_index = start_index + self.n_repeat
            subset = self.population[start_index:end_index, :]
            max_response_index = subset[:, 1].argmax()
            yield (subset[max_response_index, 0], subset[max_response_index, 1])

    def get_samples(self, x_sams):
        """
        Returns an array of sampled responses for the given x indices.

        :param x_sams: A list or array of x indices for which to generate response samples.
        :return: A numpy array of (x, y) samples.
        """
        return np.array(list(self.sample_response(x_sams)))
    def set_sample_xs(self, sample_size):
        distinct_xs = np.unique(self.population[:, 0])
        x_samp_index = np.linspace(0, len(distinct_xs)-1, sample_size).astype(int)
        x_samples = distinct_xs[x_samp_index]
        self.xindeces = x_samp_index 

 
        return (x_samp_index, x_samples)
