import copy

from secml.array import CArray

from detexe.pea.model.c_wrapper_phi import CWrapperPhi
from detexe.pea.utils.extend_pe import (create_int_list_from_x_adv,
                                        shift_section_by)

from .c_gamma_evasion import CGammaEvasionProblem


class CGammaShiftEvasionProblem(CGammaEvasionProblem):
    def __init__(
        self,
        section_population: list,
        model_wrapper: CWrapperPhi,
        population_size: int,
        penalty_regularizer: float,
        iterations: int,
        seed: int = None,
        is_debug: bool = False,
        hard_label: bool = False,
        threshold: float = 0.5,
        loss: str = "l1",
    ):
        """
        TODO: not yet tested properly
        Creates the GAMMA section injection attack

        Parameters
        ----------
        section_population : list
                a list containing all the goodware sections to inject
        model_wrapper : CWrapperPhi
                the target models, wrapped inside a CWrapperPhi
        population_size : int
                the population size generated at each round by the genetic algorithm
        penalty_regularizer: float
                the regularization parameter used for the size constraint
        iterations : int, optional, default 100
                the total number of iterations, default 100
        seed : int, optional, default None
                specifies an initialization seed for the random. None for not using determinism
        is_debug : bool, optional, default False
                if True, it prints messages while optimizing. Default is False
        hard_label : bool, optional default False
                if True, the problem will use only binary labels instead. Infinity will be used for non-evasive samples.
        threshold : float, optional, default 0
                the detection threshold. Leave 0 to test the degradation of the models until the end of the algorithm.
        loss : str, optional, default l1
                The loss function used as objective function
        """
        super(CGammaShiftEvasionProblem, self).__init__(
            section_population,
            model_wrapper,
            population_size,
            penalty_regularizer,
            iterations,
            seed,
            is_debug,
            hard_label,
            threshold,
            loss,
        )

    def apply_feasible_manipulations(self, t, x: CArray) -> CArray:
        x_adv = copy.deepcopy(x)
        overall_content = []
        for i in range(t.shape[-1]):
            content = self.section_population[i]
            content_to_append = content[: int(round(len(content) * t[i]))]
            overall_content.extend(content_to_append)
        overall_size = len(overall_content)
        x_adv, indexes_to_perturb = self._craft_perturbed_c_array(x_adv, overall_size)
        x_adv[0, indexes_to_perturb[:overall_size]] = CArray(overall_content)
        x_adv = x_adv.reshape((1, x_adv.shape[-1]))
        return x_adv

    def _craft_perturbed_c_array(self, x0: CArray, size: int):
        x_init, indexes_to_perturb = self._generate_list_adv_example(x0, size)
        self.indexes_to_perturb = indexes_to_perturb
        x_init = CArray([x_init])
        return x_init, indexes_to_perturb

    def _generate_list_adv_example(self, x0, size):
        x_init = create_int_list_from_x_adv(x0, 256, False)
        x_init, index_to_perturb_sections = shift_section_by(
            x_init, preferable_extension_amount=size
        )
        return x_init, index_to_perturb_sections
