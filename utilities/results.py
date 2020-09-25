import datetime
import json

from pathlib import Path
from qiskit.quantum_info import random_statevector, Statevector
from typing import Any, Dict, List, Optional, Tuple

from utilities.utils import ComplexEncoder


class Results:
    """
    Class to collect and work with simulations results.
    """

    # Dictionary containing results of all the runs
    result: Optional[Dict[str, Any]] = None
    # Layers previously done that can be skipped
    done_layers: List[str] = []

    def __init__(self, max_l: int = 31, result_filename: Optional[str] = None) -> None:
        """
        Initialize the result class.
        If a previous file is passed, optimization is continued from where it was stopped.

        :param max_l: The maximum value for layers.
        :param result_filename: The previous simulation filename (not the full path, ONLY the file name).
        """

        if result_filename is None:
            result_filename = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S') + '_results.json'

        self.result_filepath = Path('results/' + result_filename)

        if self.result_filepath.exists():
            self.result = json.loads(self.result_filepath.read_text(), object_hook=ComplexEncoder.as_complex)
            self.done_layers = list(self.result['results_per_n_layers'])

        # Decide which layer numbers to run based on previously loaded results
        self.layer_range: List[int] = [i for i in range(1, max_l) if str(i) not in self.done_layers]

    def update_configs(self, opt_config: Dict[str, Any], circ_config: Dict[str, Any]) -> None:
        """
        If continuing from previous run, it will update the configs with previous values and ignore current ones.
        Update is done in place.
        """
        if self.result is None:
            # If no previous run then generate random target state
            state = random_statevector(2 ** circ_config['nqubit'])
        else:
            state = Statevector(self.result['target_state'])
            opt_config['precision'] = self.result.get('error_precision', opt_config['precision'])
            opt_config['max_iter'] = self.result.get('max_iter', opt_config['max_iter'])
            opt_config['error_convergence'] = self.result.get('error_convergence', opt_config['error_convergence'])
            circ_config['nqubit'] = self.result.get('nqubit', circ_config['nqubit'])
        circ_config.update({
            'target_state': state
        })

    def save_results(self) -> None:
        """
        Store results as json.
        """
        with open(self.result_filepath, 'w') as out:
            json.dump(self.result, out, cls=ComplexEncoder)

    def add_layer_result(self, layer_result: Dict[str, Any]) -> None:
        """
        Populate result with results from a layer.
        If result is none then the full dictionary is used,
        i.e. taking also general values (like circuit config, optimizer options...).
        """
        if self.result is None:
            self.result = layer_result
        else:
            self.result['results_per_n_layers'].update(layer_result['results_per_n_layers'].copy())

    def get_latest_params(self, layer: str) -> List[float]:
        return self.result['results_per_n_layers'][layer]['angles'][-1]

    def get_latest_error(self, layer: str) -> float:
        return self.result['results_per_n_layers'][layer]['errors'][-1]
    
    def get_errors_per_layer(self, layer: str) -> List[float]:
        return self.result['results_per_n_layers'][layer]['errors']
    
    def get_lowest_set_up(self) -> Tuple[str, float, List[float]]:
        """
        Get the layer, the error and the respective params for which the error was the lowest.
        """
        min_err: float = 10000
        min_layer: Optional[str] = None
        for layer in self.done_layers:
            if self.get_latest_error(layer) < min_err:
                min_err = self.get_latest_error(layer)
                min_layer = layer
        return min_layer, min_err, self.get_latest_params(min_layer)

    def get_max_set_up(self) -> Tuple[str, float, List[float]]:
        """
        Get the layer, the error and the respective params for which the error was the maximum.
        """
        max_err: float = 1e-8
        max_layer: Optional[str] = None
        for layer in self.done_layers:
            if self.get_latest_error(layer) > max_err:
                max_err = self.get_latest_error(layer)
                max_layer = layer
        return max_layer, max_err, self.get_latest_params(max_layer)
            
    def get_last_errors(self) -> List[float]:
        """
        Collect last error per each layer
        """
        errors = []
        for layer in self.done_layers:
            errors.append(self.get_latest_error(layer))
        return errors
    
    def get_n_iterations_with_stop_reason(self) -> Tuple[List[int], List[int], List[str]]:
        """
        Collect number of iterations with stop reason per each layer.
        """
        layers: List[int] = []
        iterations: List[int] = []
        reasons: List[str] = []
        for layer in self.done_layers:
            layers.append(int(layer))
            iterations.append(len(self.result['results_per_n_layers'][layer]['errors']))
            reasons.append(self.result['results_per_n_layers'][layer]['stop_reason'])
            
        return layers, iterations, reasons
