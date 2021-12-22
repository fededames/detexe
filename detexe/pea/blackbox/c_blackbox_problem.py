from abc import abstractmethod

import numpy as np
from secml.array import CArray

from detexe.pea.model.c_wrapper_phi import CWrapperPhi


def _slice_sequence(slice_size, sequence, irregular=None):
    if irregular is not None:
        expanded_sequence = [
            sequence[: sum(irregular[: i + 1])]
            for i, _ in enumerate(irregular)
            # sequence[: sum(irregular[:i])] for i, val in enumerate(irregular)
        ]
    else:
        how_many = len(sequence) // slice_size
        expanded_sequence = [sequence[: (i + 1) * slice_size] for i in range(how_many)]
    # expanded_sequence = np.array(expanded_sequence)
    return expanded_sequence


class CBlackBoxProblem:
    """
    Base class for encapsulating a black box optimization problem.
    """

    def __init__(
        self,
        model_wrapper: CWrapperPhi,
        latent_space_size: int,
        population_size: int,
        penalty_regularizer: float,
        iterations: int,
        seed: int = None,
        is_debug: bool = False,
        hard_label: bool = False,
        threshold: float = 0,
        loss: str = "l1",
    ):
        """
        Creates an instance of a black-box problem.

        Parameters
        ----------
        model_wrapper : CWrapperPhi
                the target models to attack
        latent_space_size : int
                the dimensionality of the vectors of parameters used to perturb malware
        population_size : int
                the population size generated at each round by the genetic algorithm
        penalty_regularizer : float
                the regularization parameter that affects the size constraint
        iterations : int
                the number of iterations performed by the algorithm
        seed : int, optional, default None
                the seed to pass to the genetic engine, leave None for random behaviour.
        is_debug : bool, optional, default False
                if True, it prints debug messages while optimizing
        hard_label : bool, optional default False
                if True, the problem will use only binary labels instead. Infinity will be used for non-evasive samples.
        threshold : float, optional, default 0
                the detection threshold. Leave 0 to test the degradation of the models until the end of the algorithm.
        loss : str, optional, default l1
                The loss function used as objective function
        """
        self.model_wrapper = model_wrapper
        self.latent_space_size = latent_space_size
        self.population_size = population_size
        self._original_x = None
        self.penalty_regularizer = penalty_regularizer
        self.iterations = iterations
        self.seed = seed
        self.is_debug = is_debug
        self.hard_label = hard_label
        self.threshold = threshold
        self.loss = loss
        self.confidences_ = []
        self.fitness_ = []
        self.sizes_ = []
        self.advx = []

    def clear_results(self):
        """
        Reset the internal state after computing an attack
        """
        self.confidences_ = []
        self.fitness_ = []
        self.sizes_ = []
        self.advx = []

    def init_starting_point(self, x: CArray) -> CArray:
        """
        Initialize the problem, by setting the starting point.

        Parameters
        ----------
        x : CArray
                the initial point

        Returns
        -------
        CArray
                the initial point (padded accordingly to remove trailing invalid values)
        """
        self._original_x = x.deepcopy()
        padding_positions = x.find(x == 256)
        self.clear_results()
        if padding_positions:
            self._original_x = self._original_x[0, : padding_positions[0]]
        return self._original_x

    def get_bounds(self) -> (list, list):
        """
        Gets the bounds for the genetic algorithm
        Returns
        -------
        list, list
                lower bounds and upper bounds
        """
        return [0] * self.latent_space_size, [1] * self.latent_space_size

    def fitness(self, t: np.ndarray) -> list:
        """
        Compute the objective function.

        Parameters
        ----------
        t : numpy array
                the vector of parameter to test
        Returns
        -------
        list
                the score attributed by the target combined with the the penalty term
        """
        candidate = self.apply_feasible_manipulations(t, self._original_x)
        penalty_term = self.compute_penalty_term(
            self._original_x, candidate, self.penalty_regularizer
        )
        score = self.score_step(candidate, penalty_term)
        return [score]

    def _compute_loss(self, confidence, penalty):
        if self.loss == "l1":
            return confidence + penalty
        if self.loss == "cw":
            return max(confidence - self.threshold + 0.1, 0) + penalty
        if self.loss == "log":
            return -np.log(1 - confidence) + penalty
        raise ValueError("NO LOSS")

    def score_step(self, x: CArray, penalty_term: float) -> float:
        """
        Computes the objective function, combining the penalty term

        Parameters
        ----------
        x : CArray
                the original sample
        penalty_term : float
                the penalty term

        Returns
        -------
        float, float
        """
        _, confidence = self.model_wrapper.predict(x, return_decision_function=True)
        confidence = confidence[0, 1].item()
        if self.hard_label:
            confidence = np.infty if confidence > self.threshold else 0
        fitness_value = self._compute_loss(confidence, penalty_term)
        self.confidences_.append(confidence)
        self.sizes_.append(x.shape[-1])
        self.fitness_.append(fitness_value)
        return fitness_value

    def _export_internal_results(self, irregular=None) -> (list, list, list):
        """
        Exports the results of the attack

        Parameters
        ----------
        irregular : list, optional, default None
                Slices the internal results
        Returns
        -------

        """
        confidence = _slice_sequence(
            self.population_size, self.confidences_[1:], irregular
        )
        fitness = _slice_sequence(self.population_size, self.fitness_[1:], irregular)
        sizes = _slice_sequence(self.population_size, self.sizes_[1:], irregular)
        best_idx = [np.argmin(f) for f in fitness]
        fitness = [self.fitness_[0]] + [f[i] for f, i in zip(fitness, best_idx)]
        confidence = [self.confidences_[0]] + [
            f[i] for f, i in zip(confidence, best_idx)
        ]
        sizes = [self.sizes_[0]] + [f[i] for f, i in zip(sizes, best_idx)]

        return confidence, fitness, sizes

    def compute_penalty_term(
        self, original_x: CArray, adv_x: CArray, par: float
    ) -> float:
        """
        Computes the size penalty term

        Parameters
        ----------
        original_x : CArray
                the original malware sample
        adv_x : CArray
                the adversarial malware
        par : float
                the regularization parameter

        Returns
        -------
        flaot
                the size penalty term, multiplied by the regularization parameter
        """
        return par * abs(adv_x.shape[-1] - original_x.shape[-1])

    @abstractmethod
    def apply_feasible_manipulations(self, t: np.ndarray, x: CArray) -> CArray:
        """
        Applies the manipulation to the malware

        Parameters
        ----------
        t : numpy array
                the vector of parameters
        x : CArray
                the original malware sample

        Returns
        -------
        CArray
                the adversarial malware
        """
        raise NotImplementedError(
            "This method is abstract, you should implement it somewhere else!"
        )
