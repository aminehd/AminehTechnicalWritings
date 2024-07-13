import numpy as np

def get_population(file_path):
        files = np.load(file_path)
        return files['arr_0']
        
import numpy as np
import pandas as pd

class Population:
    def __init__(self, num_samples):
        self.predictors = {}
        self.coefficients = {}
        self.intercept = 0
        self.noise_std = 1
        self.predictor_mesh = None
    
    class Builder:
        def __init__(self, num_samples):
            self.population = Population(num_samples)
        
        def add_predictor(self, name, value_range, coefficient):
            self.population.predictors[name] = value_range
            self.population.coefficients[name] = coefficient
            return self
        
        def set_intercept(self, intercept):
            self.population.intercept = intercept
            return self

        def create_predictor_mesh(self):
            # m faces and each face len(range) dimensonal
            input = (len(
                self.population.predictors[name]
            ) for name in self.population.predictors)
            self.population.predictor_mesh = np.empty(shape=list(input), dtype=object)
            # replace empty with the combination of values in the cell.
            shape = self.population.predictor_mesh.shape
            indices = np.indices(shape)

            preds = list(self.population.predictors.values())

            for idx in np.ndindex(shape):
                    temp = tuple(index[idx] for index in indices)
                    self.population.predictor_mesh[idx] = tuple(preds[i][temp[i]] for i in range(len(preds)))
            
            # TODO: test above
            # TODO: go to set noise and create setnoise and repetion for each ...
            return self
        def set_noise(self, noise_std):
            # Most of implementation is here
            return self
        
        def build(self):
            self.create_predictor_mesh()
            return self.population
    
    def generate_data(self):
        pass

# rename it to simplePopulationSampler. Implement a multi 
# poplulation sampler where input population is not pd but population
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
        x_samp_index = np.random.choice(len(distinct_xs), sample_size, replace=True)
        x_samples = distinct_xs[x_samp_index]
        self.xindeces = x_samp_index 

 
        return (x_samp_index, x_samples)
