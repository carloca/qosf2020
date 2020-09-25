import logging
import multiprocessing as mp

from utilities.results import Results
from utilities.utils import run_process

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CIRC_CONFIG = {
        'nqubit': 4,
        'bonus_question': False
    }

OPT_CONFIG = {
    'precision': 0.00000001,  # Error absolute precision
    'max_iter': 10000,  # Max optimizer iterations
    'error_convergence': 0.00000001,  # Error convergence respect the previous value
    'check_graph': True
}

PROCESSES = 4


if __name__ == '__main__':

    result = Results(max_l=51, result_filename=None)
    # If continuing from previous run, then update the configs with previous values and ignore current ones.
    result.update_configs(OPT_CONFIG, CIRC_CONFIG)

    with mp.Pool(processes=PROCESSES) as pool:
        runs = [pool.apply_async(run_process, args=(layer, OPT_CONFIG, CIRC_CONFIG))
                for layer in result.layer_range]

        for run in runs:
            run_result = run.get()
            layer = list(run_result['results_per_n_layers'])[0]
            errors = run_result['results_per_n_layers'][layer]['errors']
            logger.info(f"---> DONE LAYER: {layer} - Total Iterations: {len(errors)} "
                        f"-> last loss: {errors[-1]}, "
                        f"stop reason: {run_result['results_per_n_layers'][layer]['stop_reason']}")
            result.add_layer_result(run_result)
            # Save at each layer (order not important), so that if something happens
            # then it can restart and do only the missing ones.
            result.save_results()
