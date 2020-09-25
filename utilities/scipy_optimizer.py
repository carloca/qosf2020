import logging
import math
import numpy as np
import scipy.optimize

from functools import partial
from typing import Any, Dict, Optional

from utilities.optimizer_base import OptimizerBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ScipyOptimizer(OptimizerBase):
    """
    Optimization via scipy.
    --> N.B: Tested only with Nelder-Mead and DEPRECATED in favour of Tensorflow! <--
    """

    def __init__(self, method: str, circuit_config: Dict[str, Any], options: Optional[Dict[str, Any]] = None,
                 max_iter: Optional[int] = None) -> None:
        """
        :param method: Scipy optimization method.
        :param circuit_config: Configuration for the circuit model class.
        :param options: Options to pass to the scipy method (they may be method specific).
        :param max_iter: Max number of iteration before optimization is stopped
                            without reaching the wanted precision or convergence.
        """
        super(ScipyOptimizer, self).__init__(circuit_config, max_iter)

        self.latest_error: Optional[float] = None
        self.latest_state: Optional[np.ndarray] = None

        if max_iter is not None:
            if options is None:
                options = {}
            options['maxiter'] = self.max_iter
        self.optimizer = partial(scipy.optimize.minimize, method=method, options=options)

    def _calculate_error(self) -> float:
        """
        Calculates error between the state generated via current parameters and the target state.
        """
        diff = self.latest_state - self.circuit_model.target_state.data
        error = diff.dot(diff.conjugate())

        return math.sqrt(error.real)

    def error(self, params: np.ndarray) -> float:
        """
        Get the current state from the quantum circuit and updates the current error.
        """
        self.latest_state = self.circuit_model.get_state(params)
        self.latest_error = self._calculate_error()

        return self.latest_error

    def run(self) -> None:
        """
        Runs optimization step, updates and stores errors and parameters in the callback.
        """

        def step_callback(params, *args, **kwargs):
            self.errors.append(self.latest_error)
            self.angles.append(params)

            self.it += 1
            if self.it % 100 == 0:
                logger.info(f'Layers: {self.circuit_model.layers} - Iterations {self.it} '
                            f'-> loss: {self.latest_error}')

        starting_params = np.random.uniform(low=0, high=2 * np.pi,
                                            size=(self.circuit_model.layers * self.circuit_model.nqubit * 2,))

        self.optimizer(self.error, starting_params, callback=step_callback)
