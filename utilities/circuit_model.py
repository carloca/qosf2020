import numpy as np

from typing import Dict, List, Optional
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import Operator, Statevector


class CircuitModel:
    """
    Class to build and work with the quantum circuit, that depends on parameters set only at the optimization time.
    Independent from chosen the optimization method.
    """

    def __init__(self, target_state: Statevector, layers: int, nqubit: int, bonus_question: bool = False) -> None:
        """
        :param target_state: Statevector that should be obtained as output of the circuit.
        :param layers: Number of layers of specific gates that repeat in the circuit.
        :param nqubit: Number of qubits.
        :param bonus_question: Boolean to decide if to build the normal circuit or the bonus one.
        """
        self.target_state = target_state
        self.layers = layers
        self.nqubit = nqubit
        self.bonus_question = bonus_question
        if self.bonus_question:
            self.params: List[Optional[Parameter]] = [None for _ in range(3 * self.nqubit * self.layers)]
        else:
            self.params: List[Optional[Parameter]] = [None for _ in range(2 * self.nqubit * self.layers)]
        self.circuit: Optional[QuantumCircuit] = QuantumCircuit(self.nqubit)

        self._build_circuit()

    def _add_one_layer(self, layer: int) -> QuantumCircuit:
        """
        Create one layer of gates depending on parameters, to be set at the moment of calculation.
        """
        circ = QuantumCircuit(self.nqubit)

        for qb in range(self.nqubit):
            # Create list of parameter of the type {gate}_ij where i is the layer and j the qubit.
            self.params[2 * self.nqubit * layer + qb] = Parameter(f'x_{layer}{qb}')
            self.params[2 * self.nqubit * layer + qb + self.nqubit] = Parameter(f'z_{layer}{qb}')
            circ.rx(self.params[2 * self.nqubit * layer + qb], qb)
            circ.rz(self.params[2 * self.nqubit * layer + qb + self.nqubit], qb)

        # For each layer a chain of cz gates is also applied.
        for i in range(self.nqubit - 1):
            for j in range(i + 1, self.nqubit):
                circ.cz(i, j)

        return circ

    def _add_one_layer_bonus(self, layer: int) -> QuantumCircuit:
        """
        Create one layer of gates depending on parameters, to be set at the moment of calculation,
        for the bonus circuit.
        """
        circ = QuantumCircuit(self.nqubit)

        for qb in range(self.nqubit):
            start_value = 3 * self.nqubit * layer + 3 * qb
            # Create list of parameter of the type {angle}_ij where i is the layer and j the qubit.
            self.params[start_value] = Parameter(f'theta_{layer}{qb}')
            self.params[start_value + 1] = Parameter(f'phi_{layer}{qb}')
            self.params[start_value + 2] = Parameter(f'lambda_{layer}{qb}')
            circ.u3(self.params[start_value], self.params[start_value + 1],
                    self.params[start_value + 2], qb)

        # For each layer a chain of cx gates is also applied.
        for i in range(self.nqubit - 1):
            for j in range(i + 1, self.nqubit):
                circ.cx(j, i)

        return circ

    def _build_circuit(self) -> None:
        """
         Build the main circuit, repeating inner circuit structure for the number of layers.
        """
        for layer in range(self.layers):
            if self.bonus_question:
                self.circuit.extend(self._add_one_layer_bonus(layer))
            else:
                self.circuit.extend(self._add_one_layer(layer))
            self.circuit.barrier()

    def get_cz_operator(self) -> np.ndarray:
        """
        Get the operator for the cz gates for the optimizer.
        """
        circ = QuantumCircuit(self.nqubit)
        for i in range(self.nqubit - 1):
            for j in range(i + 1, self.nqubit):
                circ.cz(i, j)
        op = Operator(circ)
        return op.data

    def get_cnot_operator(self) -> np.ndarray:
        """
        Get the operator for the cx gates for the optimizer.
        """
        circ = QuantumCircuit(self.nqubit)
        for i in range(self.nqubit - 1):
            for j in range(i + 1, self.nqubit):
                circ.cx(j, i)
        op = Operator(circ)
        return op.data

    def _bind_circuit(self, input_params: np.ndarray) -> QuantumCircuit:
        """
        Parameters for the gates are passed as array with the following structure:
            - First all the x gates parameter for each qubit, then the y gates and all is repeated for each layer.
              Ex:  values = [x_00, x_01, ..., x_0nqubit, y_00, y_01, ..., y_0nqubit,...
                            ..., x_nlayer0, ..., x_nlayernqubit, y_nlayer0, ..., y_nlayernqubit]
                    where each element is {gate}_ij with i the layer and j the qubit.
        """
        input_dict: Dict[Parameter, float] = {}
        for layer in range(self.layers):
            for qb in range(self.nqubit):
                input_dict.update({
                    self.params[2 * layer * self.nqubit + qb]: input_params[2 * layer * self.nqubit + qb],
                    self.params[2 * layer * self.nqubit + qb + self.nqubit]: input_params[2 * layer * self.nqubit
                                                                                          + qb + self.nqubit]
                })
        return self.circuit.bind_parameters(input_dict)

    def _bind_circuit_bonus(self, input_params: np.ndarray) -> QuantumCircuit:
        """
        Parameters for the gates are passed as array with the following structure:
            - First all the 3 angles for the rotation gate parameter for the first qubit,
              then the other 3 angles for the second qubit gate and so on, finally all is repeated for each layer.
              Ex:  values = [theta_00, phi_00, lambda_00, ..., theta_0nqubit, phi_0nqubit, lambda_0nqubit,...
                             theta_nlayer0, phi_nlayer0, lambda_nlayer0, ...,
                             theta_nlayernqubit, phi_nlayernqubit, lambda_nlayernqubit]
                    where each element is {angle}_ij with i the layer and j the qubit.
        """
        input_dict: Dict[Parameter, float] = {}
        for layer in range(self.layers):
            for qb in range(self.nqubit):
                start_value = 3 * self.nqubit * layer + 3 * qb
                input_dict.update({
                    self.params[start_value]: input_params[start_value],
                    self.params[start_value + 1]: input_params[start_value + 1],
                    self.params[start_value + 2]: input_params[start_value + 2]
                })
        return self.circuit.bind_parameters(input_dict)

    def bind_circuit(self, input_params: np.ndarray) -> QuantumCircuit:
        """
        Assign values to the gates parameters.
        The method returns a new circuit so values are not assigned in place,
        leaving the class circuit untouched and independent from specific values.
        For parameters example see private specific bind method.
        """
        if self.bonus_question:
            return self._bind_circuit_bonus(input_params)
        else:
            return self._bind_circuit(input_params)

    def get_state(self, input_params: np.array) -> np.ndarray:
        """
        Given the parameters array with the values, it generate the Statevector that the circuit will output.
        """
        curr_circ = self.bind_circuit(input_params)
        simulator = Aer.get_backend('statevector_simulator')
        result = execute(curr_circ, backend=simulator).result()
        state = result.get_statevector(curr_circ)

        return state
