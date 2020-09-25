import logging
import numpy as np
import tensorflow as tf

from typing import Any, Dict

from utilities.optimizer_base import OptimizerBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GradOptimizer(OptimizerBase):
    """
    Optimization via gradient and Tensorflow with Adam method.
    """

    def __init__(self, precision: float, max_iter: int, error_convergence: float, circuit_config: Dict[str, Any],
                 learning_rate: float = 0.05, check_graph: bool = False) -> None:
        """
        :param precision: If the error/loss reach the wanted precision then the optimization is stopped.
        :param max_iter: Max number of iteration before optimization is stopped
                            without reaching the wanted precision or convergence.
        :param error_convergence: Convergence for the error/loss value.
        :param circuit_config: Configuration for the circuit model class.
        :param learning_rate: Learning rate for the optimizer.
        :param check_graph: If True, the state obtained via tensorflow graph will be compared to the one obtained
                            via quantum circuit, given the same parameters.
                            This is to check that the tf graph is correctly implemented.
        """
        super(GradOptimizer, self).__init__(circuit_config, max_iter)

        self.check_graph = check_graph
        self.precision = precision
        self.error_convergence = error_convergence
        self.learning_rate = learning_rate

        if self.circuit_model.bonus_question:
            self.var = tf.Variable(np.random.uniform(low=0, high=2 * np.pi,
                                                     size=(self.circuit_model.layers * self.circuit_model.nqubit * 3,)),
                                   constraint=lambda x: tf.clip_by_value(x, 0, 2 * np.pi))

            self.loss = lambda: self._generate_loss_bonus(self.var)
        else:
            self.var = tf.Variable(np.random.uniform(low=0, high=2 * np.pi,
                                                     size=(self.circuit_model.layers * self.circuit_model.nqubit * 2,)),
                                   constraint=lambda x: tf.clip_by_value(x, 0, 2 * np.pi))

            self.loss = lambda: self._generate_loss(self.var)

    @staticmethod
    def _r_x(angle: tf.Tensor) -> tf.Tensor:
        """
        Create the x-rotation operator for one qubit.
            rx (theta) = [[cos(theta/2), -i*sin(theta/2)],
                          [-i*sin(theta/2), cos(theta/2)]]
        """
        cos = tf.cos(tf.multiply(angle, tf.constant(0.5, dtype=tf.float64)))
        sin = tf.sin(tf.multiply(angle, tf.constant(0.5, dtype=tf.float64)))
        sin_complex = tf.complex(tf.constant(0, dtype=tf.float64), sin)
        cos_complex = tf.complex(cos, tf.constant(0, dtype=tf.float64))
        rx = tf.stack([[cos_complex, -sin_complex], [-sin_complex, cos_complex]])

        return rx

    @staticmethod
    def _r_z(angle: tf.Tensor) -> tf.Tensor:
        """
        Create the z-rotation operator for one qubit, without a general phase (replicating behaviour of qiskit).
            rz (theta) = [[1, 0],
                          [0, exp(i*theta)]]
        """
        zero = tf.constant(0, dtype=tf.float64)
        exponent = tf.complex(zero, angle)
        exp = tf.exp(exponent)
        zero_complex = tf.complex(zero, zero)
        one_complex = tf.complex(tf.constant(1, dtype=tf.float64), zero)
        rz = tf.stack([[one_complex, zero_complex], [zero_complex, exp]])

        return rz

    @staticmethod
    def _r_u3(angle: tf.Tensor) -> tf.Tensor:
        """
        Create the rotation u3 operator for one qubit.
            u3 (theta, phi, lambda) = [[cos(theta/2),           -exp(i*lambda)*sin(theta/2)],
                                       [exp(i*phi)*sin(theta/2), exp(i*(phi+lambda))*cos(theta/2)]]
        """
        zero = tf.constant(0, dtype=tf.float64)
        cos = tf.cos(tf.multiply(angle[0], tf.constant(0.5, dtype=tf.float64)))
        sin = tf.sin(tf.multiply(angle[0], tf.constant(0.5, dtype=tf.float64)))
        sin_complex = tf.complex(sin, zero)
        cos_complex = tf.complex(cos, zero)
        exponent_phi = tf.complex(zero, angle[1])
        exp_phi = tf.exp(exponent_phi)
        exponent_lambda = tf.complex(zero, angle[2])
        exp_lambda = tf.exp(exponent_lambda)
        exponent_lambda_phi = tf.complex(zero, angle[1] + angle[2])
        exp_lambda_phi = tf.exp(exponent_lambda_phi)
        ru3 = tf.stack([[cos_complex, -tf.multiply(exp_lambda, sin_complex)],
                        [tf.multiply(exp_phi, sin_complex), tf.multiply(exp_lambda_phi, cos_complex)]])

        return ru3

    def _fix_dimension(self, rot: tf.Tensor) -> tf.Tensor:
        """
        Fix dimension of rotation on n-qubit after tensor product of single qubit rotation.
        The new shape will be (2^nqubit, 2^nqubit).
        """
        even_n = [i for i in range(0, self.circuit_model.nqubit * 2, 2)]
        odd_n = [i for i in range(1, self.circuit_model.nqubit * 2, 2)]
        perm = even_n + odd_n
        rot = tf.transpose(rot, perm=perm)
        rot = tf.reshape(rot, shape=(2 ** self.circuit_model.nqubit, 2 ** self.circuit_model.nqubit))
        return rot

    def _get_norm(self, circ: tf.Tensor) -> tf.Tensor:
        """
        Given the circuit, calculate output state and compute loss with the target state.
        """
        final_state = self.circuit_model.target_state.data

        init = np.zeros(2 ** self.circuit_model.nqubit, dtype=np.float64)
        # Initially all the qubits are 0
        init[0] = 1

        out = tf.linalg.matvec(circ, tf.complex(init, tf.constant(0, dtype=tf.float64)))
        diff = out - final_state
        loss = tf.cast(tf.norm(diff), dtype=tf.float64)

        if self.check_graph:
            # Check if the obtained state is correct, comparing with the one obtained
            # from the quantum circuit with same parameters
            state_tmp = self.circuit_model.get_state(self.var.numpy())
            assert all(out.numpy().round(6) == state_tmp.round(6)) or \
                   all(out.numpy().round(6) == -state_tmp.round(6))

        return loss

    def _generate_loss(self, angles: tf.Variable) -> tf.Tensor:
        """
        Generate graph that simulate the circuit and calculate loss/error with respect the wanted state.
        Parameters for the gates are passed as array with the following structure:
            - First all the x gates parameter for each qubit, then the y gates and all is repeated for each layer.
              Ex:  values = [x_00, x_01, ..., x_0nqubit, y_00, y_01, ..., y_0nqubit,...
                            ..., x_nlayer0, ..., x_nlayernqubit, y_nlayer0, ..., y_nlayernqubit]
                    where each element is {gate}_ij with i the layer and j the qubit.
        """
        # Get the operator for all the cz gates
        cz = tf.constant(self.circuit_model.get_cz_operator())
        # Start from identity to iterate
        circ = tf.complex(tf.eye(2 ** self.circuit_model.nqubit, dtype=tf.float64), tf.constant(0, dtype=tf.float64))
        for layer in range(self.circuit_model.layers - 1, -1, -1):
            rot_x = self._r_x(angles[2 * layer * self.circuit_model.nqubit])
            rot_z = self._r_z(angles[2 * layer * self.circuit_model.nqubit
                                     + self.circuit_model.nqubit])
            for qb in range(1, self.circuit_model.nqubit):
                rot_x = tf.tensordot(self._r_x(angles[2 * layer * self.circuit_model.nqubit + qb]),
                                     rot_x, axes=0)
                rot_z = tf.tensordot(
                    self._r_z(angles[2 * layer * self.circuit_model.nqubit + qb + self.circuit_model.nqubit]),
                    rot_z, axes=0)
            rot_x = self._fix_dimension(rot_x)
            rot_z = self._fix_dimension(rot_z)
            circ = tf.matmul(circ, cz)
            circ = tf.matmul(circ, rot_z)
            circ = tf.matmul(circ, rot_x)

        loss = self._get_norm(circ)

        return loss

    def _generate_loss_bonus(self, angles: tf.Variable) -> tf.Tensor:
        """
        Generate graph that simulate the bonus circuit and calculate loss/error with respect the wanted state.
        Parameters for the gates are passed as array with the following structure:
            - First all the 3 angles for the rotation gate parameter for the first qubit,
              then the other 3 angles for the second qubit gate and so on, finally all is repeated for each layer.
              Ex:  values = [theta_00, phi_00, lambda_00, ..., theta_0nqubit, phi_0nqubit, lambda_0nqubit,...
                             theta_nlayer0, phi_nlayer0, lambda_nlayer0, ...,
                             theta_nlayernqubit, phi_nlayernqubit, lambda_nlayernqubit]
                    where each element is {angle}_ij with i the layer and j the qubit.
        """
        # Get the operator for all the cx gates
        cx = tf.constant(self.circuit_model.get_cnot_operator())
        # Start from identity to iterate
        circ = tf.complex(tf.eye(2 ** self.circuit_model.nqubit, dtype=tf.float64), tf.constant(0, dtype=tf.float64))
        for layer in range(self.circuit_model.layers - 1, -1, -1):
            start_angle = 3 * layer * self.circuit_model.nqubit
            rot_u3 = self._r_u3(angles[start_angle:start_angle + 3])
            for qb in range(1, self.circuit_model.nqubit):
                rot_u3 = tf.tensordot(self._r_u3(angles[start_angle + 3 * qb: start_angle + 3 * qb + 3]),
                                      rot_u3, axes=0)

            rot_u3 = self._fix_dimension(rot_u3)
            circ = tf.matmul(circ, cx)
            circ = tf.matmul(circ, rot_u3)

        loss = self._get_norm(circ)

        return loss

    def _get_learning_rate(self) -> float:
        """
        If loss starts growing, i.e. it will oscillate around a value, then the learning rate is reduced,
        to improve convergence.
        """
        if len(self.errors) > 1 and self.errors[-2] - self.errors[-1] < 0:
            self.learning_rate /= 2
        return self.learning_rate

    def run(self) -> None:
        """
        Run optimization step until the following conditions are met:
            - the error/loss is bigger than the wanted precision
            - the max iterations number has not been reached
            - the difference between two consecutive errors is bigger than the wanted convergence
        """
        opt = tf.keras.optimizers.Adam(learning_rate=self._get_learning_rate)
        while self.loss().numpy() > self.precision and self.it < self.max_iter \
                and (len(self.errors) == 0 or abs(self.errors[-1] - self.loss().numpy()) > self.error_convergence):
            self.errors.append(self.loss().numpy().copy())
            self.angles.append(self.var.numpy().tolist())
            opt.minimize(self.loss, var_list=[self.var])
            self.it += 1
            if self.it % 100 == 0:
                logger.info(f'Layers: {self.circuit_model.layers} - Iterations {self.it} '
                            f'-> loss: {self.loss().numpy()}, err_convergence: {self.errors[-1] - self.loss().numpy()}')
        self.errors.append(self.loss().numpy().copy())
        self.angles.append(self.var.numpy().tolist())
