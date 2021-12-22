import copy

import lief
from secml.array import CArray

from detexe.pea.model.c_wrapper_phi import CWrapperPhi

from .c_blackbox_problem import CBlackBoxProblem


class CGammaAPIEvasionProblem(CBlackBoxProblem):
    def __init__(
        self,
        api_list: list,
        model_wrapper: CWrapperPhi,
        population_size: int,
        penalty_regularizer: float,
        iterations: int,
        seed: int = None,
        is_debug: bool = False,
    ):
        """
        TODO: not yet tested properly

        Manipulate the sample by adding APIs, using the Gamma approach.

        Parameters
        ----------
        api_list : list
                A list of couples [dll_name, function_name] of each API that can be included during the optimization
        model_wrapper : CWrapperPhi
                The models under attack
        population_size : int
                How many samples will be generated by the genetic strategy at each round
        penalty_regularizer : float
                The value of the regularization parameter
        iterations: int
                Specifies how many generations will be created by the genetic algorithm.
                The overall number of queries will be (iterations + 1) * population_size
        seed : int
                Specify a seed for init the random engine
        is_debug : bool
                Prints additional information during the optimization

        Returns
        ----------
        An evasion object ready to run
        """
        super(CGammaAPIEvasionProblem, self).__init__(
            model_wrapper,
            len(api_list),
            population_size,
            penalty_regularizer,
            iterations,
            seed,
            is_debug,
        )

        self.api_list = api_list
        self._cache_lib = None
        self._names = None

    def _add_import(self, liefpe, dll, func_name):
        lib = [
            file_import
            for file_import in liefpe.imports
            if file_import.name.lower() == dll.lower()
        ]
        if lib == []:
            lib = liefpe.add_library(dll)
        lib = lief.add_library(dll) if lib == [] else lib[0]
        names = set([e.name for e in lib.entries])
        if func_name not in names:
            lib.add_entry(func_name)

    def apply_feasible_manipulations(self, t, x: CArray) -> CArray:
        x_adv = copy.deepcopy(x)
        x_adv = x_adv.tolist()
        lief_adv: lief.PE.Binary = lief.PE.parse(x_adv[0])
        for i in range(t.shape[-1]):
            dll_name, function_name = self.api_list[i]
            self._add_import(lief_adv, dll_name, function_name)
        builder = lief.PE.Builder(lief_adv)
        builder.build_imports(True).patch_imports(True)
        builder.build()
        x_adv = CArray(builder.get_build())
        x_adv = x_adv.reshape((1, x_adv.shape[-1]))
        return x_adv
