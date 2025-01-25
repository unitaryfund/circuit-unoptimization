import math
import numpy as np
from qiskit.quantum_info import Statevector
from qiskit.circuit import QuantumRegister
from qiskit.circuit.library import XGate
from qiskit.transpiler import PassManager, Layout

def get_ideal_probabilities(model_circuit):
    zero = Statevector.from_label('0' * model_circuit.num_qubits)
    sv = zero.evolve(model_circuit)
    return sv.probabilities_dict()

def get_heavy_strings(ideal_probs):
    prob_median = float(np.real(np.median(list(ideal_probs.values()))))
    heavy_strings = list(
        filter(
            lambda x: ideal_probs[x] > prob_median,
            list(ideal_probs.keys()),
        )
    )
    return prob_median, heavy_strings

def hop(counts, ideal_probs):
	_, heavy_strings = get_heavy_strings(ideal_probs)
	shots = sum(counts.values())
	heavy_output_probability = sum([counts.get(value, 0) for value in heavy_strings]) / shots
	return heavy_output_probability

def calc_z_value(mean, sigma):
        if sigma == 0:
            # assign a small value for sigma if sigma = 0
            sigma = 1e-10
            print('Standard deviation sigma should not be zero.')
        z_value = (mean - 2/3.0) / sigma
        return z_value

def calc_confidence_level(z_value):
        confidence_level = 0.5 * (1 + math.erf(z_value/2**0.5))
        return confidence_level
