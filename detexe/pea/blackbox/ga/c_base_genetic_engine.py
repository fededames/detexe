import copy
import random
import time
from typing import Any, Tuple

import numpy as np
from secml.adv.attacks import CAttackEvasion
from secml.array import CArray
from secml.data import CDataset

from ..c_blackbox_problem import CBlackBoxProblem

try:
    import deap  # noqa
    from deap import algorithms, base, creator, tools  # noqa
except ImportError:
    raise ImportError("Install DEAP to apply black box evasion")


def _pad_sequence_with_last(sequence, until):
    how_many = until - len(sequence)
    if how_many <= 0:
        return sequence
    return sequence + [sequence[-1]] * how_many


class CGeneticAlgorithm(CAttackEvasion):
    """
    The genetic optimizer.
    """

    def _objective_function_gradient(self, x):
        raise NotImplementedError("This function is not used inside secML malware")

    def _objective_function(self, x):
        raise NotImplementedError("This function is not used inside secML malware")

    def objective_function_gradient(self, x):
        raise NotImplementedError("This function is not used inside secML malware")

    def objective_function(self, x):
        raise NotImplementedError("This function is not used inside secML malware")

    def f_eval(self):
        raise NotImplementedError("This function is not used inside secML malware")

    def grad_eval(self):
        raise NotImplementedError("This function is not used inside secML malware")

    def __init__(
        self,
        problem: CBlackBoxProblem,
        is_debug: bool = False,
    ):
        """
        Create and instance of the genetic optimizer.

        Parameters
        ----------
        problem : CBlackBoxProblem
                The problem to optimize
        is_debug : bool, default False
                If True, debug prints will be displayed during the attack.
                Default is False
        """
        CAttackEvasion.__init__(
            self,
            problem.model_wrapper.classifier,
            problem.model_wrapper.classifier,
        )
        self.problem = problem
        self.confidences_ = []
        self.fitness_ = []
        self.sizes_ = []
        self.changes_per_iterations_ = []
        self.model_wrapper = problem.model_wrapper
        self.is_debug = is_debug
        self._original_x = None
        self.minimization_result_ = []
        self.evolved_problem_ = None
        self.stagnation = 5
        self.elapsed_time_ = 0

    def run(self, x, y, ds_init=None) -> Tuple[CArray, CArray, CDataset, Any]:
        """
        Runs the genetic algorithms.

        Parameters
        ----------
        x : CArray
                input sample to perturb
        y : CArray
                original class
        ds_init : CDataset, optional, default None
                the initialization point.
                Default is None
        Returns
        -------
        CArray
                y_pred : the predicted label after the attack
        CArray
                scores : the scores after the attack
        CDataset
                adv_ds : the CDataset containing the adversarial points
        CArray
                f_obj : the mean value for the objective function
        """
        x = CArray(x).atleast_2d()
        y = CArray(y).atleast_2d()
        x_init = None if ds_init is None else CArray(ds_init.X).atleast_2d()

        # only consider samples that can be manipulated
        v = self.is_attack_class(y)
        idx = CArray(v.find(v)).ravel()
        # print(v, idx)

        # number of modifiable samples
        n_mod_samples = idx.size

        adv_ds = CDataset(x.deepcopy(), y.deepcopy())

        # If dataset is sparse, set the proper attribute
        if x.issparse is True:
            self._issparse = True

        # array in which the value of the optimization function are stored
        fs_opt = CArray.zeros(
            n_mod_samples,
        )
        y_pred = CArray.zeros(
            n_mod_samples,
        )
        scores = CArray.zeros((n_mod_samples, 2))
        for i in range(n_mod_samples):
            k = idx[i].item()  # idx of sample that can be modified

            xi = x[k, :] if x_init is None else x_init[k, :]
            x_opt, f_opt = self._run(x[k, :], y[k], x_init=xi)

            self.logger.debug(
                "Point: {:}/{:}, f(x):{:}, eval:{:}/{:}".format(
                    k, x.shape[0], f_opt, self.f_eval, self.grad_eval
                )
            )
            if x_opt.shape[-1] > adv_ds.X.shape[-1]:
                # Need to resize the whole adv dataset, since CDataset can't deal with varying vector sizes
                new_length = x_opt.shape[-1]
                adv_ds.X = adv_ds.X.resize((adv_ds.X.shape[0], new_length), 256)
            adv_ds.X[k, : min(adv_ds.X.shape[-1], x_opt.shape[-1])] = x_opt
            fs_opt[i] = f_opt
            y_p, score = self.problem.model_wrapper.predict(
                x_opt, return_decision_function=True
            )
            scores[i, :] = score[0, :]
            y_pred[i] = y_p

        # Return the mean objective function value on the evasion points (
        # computed from the outputs of the surrogate classifier)
        f_obj = fs_opt.mean()

        return y_pred, scores, adv_ds, f_obj

    def _run(self, x0, y0, x_init=None):

        if x_init is None:
            x_init = copy.deepcopy(x0)

        self._original_x = self.problem.init_starting_point(x_init)
        _ = self.problem.score_step(x_init, 0)

        if self.is_debug:
            print(f"> Original Confidence: {self.confidences_[0]}")
            print("> Beginning new sample evasion...")

        minimization_results = self._compute_black_box_optimization()

        self.minimization_result_ = minimization_results.tondarray()

        x_adv = self.problem.apply_feasible_manipulations(
            self.minimization_result_, self._original_x
        )

        if self.is_debug:
            print(f">AFTER INVERSION, CONFIDENCE SCORE: {self.confidences_[-1]}")
        return x_adv, self.confidences_[-1]

    def _compute_black_box_optimization(self) -> CArray:

        start_t = time.time()

        slice_indexes = [self.problem.population_size]

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.random)
        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            toolbox.attr_float,
            n=self.problem.latent_space_size,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", self.problem.fitness)
        toolbox.register("mate", tools.cxOnePoint)
        toolbox.register("mutate", random_mutation, indpb=0.3)
        toolbox.register(
            "select", tools.selTournament, tournsize=self.problem.population_size
        )

        pop = toolbox.population(n=self.problem.population_size)
        fitness = [self.problem.fitness(np.array(t)) for t in pop]
        for ind, fit in zip(pop, fitness):
            ind.fitness.values = fit

        # CXPB  is the probability with which two individuals are crossed
        # MUTPB is the probability for mutating an individual
        CXPB, MUTPB = 0.9, 0.3
        g = 0
        last_n_best_fits = []
        while g < self.problem.iterations:
            # Select the next generation individuals
            offspring = toolbox.select(pop, self.problem.population_size)
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            slice_indexes.append(len(invalid_ind))
            fitness = [self.problem.fitness(np.array(t)) for t in invalid_ind]
            for ind, fit in zip(invalid_ind, fitness):
                ind.fitness.values = fit

            pop.extend(invalid_ind)
            fits = [ind.fitness.values[0] for ind in pop]
            best_fitness = min(fits)
            last_n_best_fits.insert(0, best_fitness)
            last_n_best_fits = last_n_best_fits[: self.stagnation]
            if self.is_debug:
                print(f">{g} - Global min: {best_fitness}")
            if len(last_n_best_fits) == self.stagnation and (
                all((np.array(last_n_best_fits) - best_fitness) < 1e-6)
                or all(np.array(last_n_best_fits) == np.infty)
            ):
                if self.is_debug:
                    print("Stagnating result!")
                break
            g += 1

        confidences, fitness, sizes = self.problem._export_internal_results(
            slice_indexes
        )
        end_t = time.time()
        self.confidences_ = _pad_sequence_with_last(
            confidences, self.problem.iterations
        )
        self.fitness_ = _pad_sequence_with_last(fitness, self.problem.iterations)
        self.sizes_ = _pad_sequence_with_last(sizes, self.problem.iterations)
        self.elapsed_time_ = end_t - start_t
        best_t = tools.selBest(pop, 1)[0]

        del creator.FitnessMin
        del creator.Individual

        return CArray(best_t)

    @classmethod
    def write_adv_to_file(cls, x_adv: CArray, path: str):
        """
        Write the adversarial malware as a file on disk

        Parameters
        ----------
        x_adv : CArray
                The adversarial malware to dump
        path : str
                The path where to save the executable
        """
        x_real = x_adv.tolist()[0]
        x_real_adv = b"".join([bytes([i]) for i in x_real])
        with open(path, "wb") as f:
            f.write(x_real_adv)


def random_mutation(individual, indpb):
    """
    Apply the mutation operator, that perturb randomly each entry of the individual, with a given probability.
    The mutation is applied in-place.

    Parameters
    ----------
    individual :
            the individual to mutate
    indpb : float
            the probability of altering a single entry
    Returns
    -------
    tuple
            the mutated individual, the mutatio is in-place
    """
    size = len(individual)
    for i in range(size):
        if random.random() < indpb:
            individual[i] = random.random()
    return (individual,)
