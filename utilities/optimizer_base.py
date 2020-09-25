from typing import Any, Dict, List, Optional

from utilities.circuit_model import CircuitModel


class OptimizerBase:
    """
    Base class for all the optimization method.
    """

    def __init__(self, circuit_config: Dict[str, Any], max_iter: int) -> None:
        """
        :param circuit_config: Quantum circuit configuration for the CircuitModel class.
        :param max_iter: Maximum iteration before stopping optimization,
                         if other convergence requirements are still not met.
        """
        self.circuit_model = CircuitModel(**circuit_config)
        self.max_iter = max_iter

        self.angles: List[List[float]] = []
        self.errors: List[float] = []
        self.it: int = 0

    def extract_result_to_json(self, results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate the json with the results.
        If previous results exist, then it will update only values for current nlayer.
        """
        if results is None:
            results = {
                'nqubit': self.circuit_model.nqubit,
                'target_state': self.circuit_model.target_state.data.tolist(),
                'max_iter': self.max_iter,
                'optimization_method': self.__class__.__name__,
                'error_precision': getattr(self, 'precision', None),
                'error_convergence': getattr(self, 'error_convergence', None),
                'results_per_n_layers': {}
            }
        stop_reason = None
        # At the moment only GradOptimizer has been deeply explored and stop_reasons may be extracted.
        # Nelder-Mead optimizer has been tested with standard parameters,
        # therefore further testing is required to define stop_reasons.
        if results.get('optimization_method', '') == 'GradOptimizer':
            if self.errors[-1] <= self.precision:
                stop_reason = 'Error precision'
            elif self.it >= self.max_iter:
                stop_reason = 'Max iterations'
            elif abs(self.errors[-2] - self.errors[-1]) <= self.error_convergence:
                stop_reason = 'Error convergence'
        base_results = {
            self.circuit_model.layers: {
                'errors': self.errors,
                'angles': self.angles,
                'stop_reason': stop_reason
            }
        }
        results['results_per_n_layers'].update(base_results.copy())
        return results

    def run(self) -> None:
        """
        This is the method that will run optimization until stop conditions are met.
        It must be implemented in the child class, since it depends on the specific optimization method.
        """
        raise Exception("This method must be implemented in the specific child class.")
