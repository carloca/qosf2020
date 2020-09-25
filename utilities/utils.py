import json

from typing import Any, Dict, Union

from utilities.grad_optimizer import GradOptimizer


class ComplexEncoder(json.JSONEncoder):
    """
    Json encoder class for complex numbers
    """

    def default(self, obj: Any) -> Union[Dict[str, Any], Any]:
        if isinstance(obj, complex):
            return {"__complex__": True,
                    "real": obj.real,
                    "imag": obj.imag}
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)

    @staticmethod
    def as_complex(dct: Dict[str, Any]) -> Union[complex, Dict[str, Any]]:
        """
        Method to decode complex numbers from json dict.
        """
        if '__complex__' in dct:
            return complex(dct['real'], dct['imag'])
        return dct


def run_process(layer: int, opt_config: Dict[str, Any], circ_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run Optimization as process in multiprocessing for a single nlayer.
    """
    circ_config.update({
        'layers': layer
    })
    opt = GradOptimizer(**opt_config, circuit_config=circ_config)
    opt.run()
    result: Dict[str, Any] = opt.extract_result_to_json()

    return result
