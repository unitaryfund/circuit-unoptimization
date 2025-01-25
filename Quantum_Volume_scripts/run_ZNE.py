from scipy.stats import linregress
import copy
from collections import namedtuple
import functools
import random
import warnings
import numpy as np
from qiskit_aer import AerSimulator
from qiskit_aer.noise import amplitude_damping_error, depolarizing_error, NoiseModel
from qiskit import QuantumCircuit, transpile
from qiskit.providers import Backend
from qiskit.quantum_info import random_unitary, Operator
from qiskit.circuit.library import UnitaryGate
from mitiq import zne
from circuit_unoptimization import circuit_unoptimization
from qiskit.circuit.library import QuantumVolume
import copy
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit

from utils import get_ideal_probabilities, get_heavy_strings, hop, calc_z_value, calc_confidence_level

from mitiq import zne

def quadratic(x, a, b, c):
	return a*x**2 + b*x + c

def amplitude_damping_noise_model(prob_1: float = 0.04, prob_2: float = 0.08) -> NoiseModel:
    """Defines an amplitude damping noise model with one-qubit and two-qubit errors.
    
    Args:
        prob_1: One-qubit gate error rate (default 4%).
        prob_2: Two-qubit gate error rate (default 8%).
    Returns:
        Amplitude damping noise model.
    """
    # Quantum errors
    error_1 = amplitude_damping_error(param_amp=prob_1, excited_state_population=1)

    error_2 = amplitude_damping_error(param_amp=prob_2, excited_state_population=1)
    error_2 = error_2.tensor(error_2)

    # Add errors to noise model
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ["u1", "u2", "u3"])
    noise_model.add_all_qubit_quantum_error(error_2, ["cx"])
    
    return noise_model

def depolarizing_noise_model(error: float = 0.01) -> NoiseModel:
    """Defines an depolarizing noise model with one-qubit.

    Args:
        error: One-qubit gate error rate (default 1%).
    Returns:
        Depolarizing noise model.
    """
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(error, 1), ["u1", "u2", "u3"])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(error, 2), "cx")
    
    return noise_model
def get_exact_HOP(qc):
	qc.remove_final_measurements()
	ideal_probs = get_ideal_probabilities(qc)
	median, heavy_strings = get_heavy_strings(ideal_probs)
	theoretical_HOP = 0
	for s in heavy_strings:
		theoretical_HOP += ideal_probs[s]
	return theoretical_HOP, ideal_probs


strat = "P_c"
N = 10
rand_seed = 10
random.seed(rand_seed)
np.random.seed(rand_seed)

qc = QuantumVolume(N, N, seed=rand_seed)
qc.measure_all()
qc = transpile(qc, basis_gates=["u3", "cx"], optimization_level=3)
print(qc.count_ops())

theoretical_HOP, ideal_probs = get_exact_HOP(copy.deepcopy(qc))
print("ideal HOP", theoretical_HOP)

init = dict(qc.count_ops())["u3"]+dict(qc.count_ops())["cx"]

x = []
y = []
for idx in range(35):
	scaled = circuit_unoptimization(copy.deepcopy(qc), iterations=idx, strategy=strat)
	scaled = transpile(scaled, basis_gates=["u3", "cx"], optimization_level=3)
	scaled_count = dict(scaled.count_ops())["u3"]+dict(scaled.count_ops())["cx"]
	scale_factor = scaled_count/float(init)
	backend = AerSimulator()
	noise_model = depolarizing_noise_model(error=0.001)
	#noise_model = amplitude_damping_noise_model()
	shots = 1_000_000

	result = backend.run(scaled, noise_model=noise_model, shots=shots).result()
	counts = result.get_counts()
	#print(counts)

	experimental_prob = hop(counts, ideal_probs)
	print(experimental_prob)
	y.append(experimental_prob)
	x.append(scale_factor)
print(x)
print(y)

res = linregress(x, y)
print("Zero noise limit = ", res.intercept)
intercept = res.intercept

print(res.slope)
slope = res.slope

plt.scatter(x, y, marker=".", color="blue")
linx = np.linspace(0, max(x), 100)
plt.plot(linx, [slope*i + intercept for i in linx], "--", color="blue", alpha=0.7, label="Linear Best Fit")


popt, pcov = curve_fit(quadratic, x, y)
print("fitted quadratic coeffs:", popt)
plt.plot(linx, quadratic(np.array(linx), *popt), "--", color="green", alpha=0.7, label="Quadratic Best Fit")

#factory = zne.inference.RichardsonFactory(scale_factors=x)
#[factory.push({"scale_factor": s}, val) for s, val in zip(x, y)]
#richardson_extrapolation = factory.reduce()
#print("richardson_extrapolation", richardson_extrapolation)

plt.scatter([0], [theoretical_HOP], marker="o", color="red", label="Ideal HOP", zorder=5)
plt.scatter([0], [res.intercept], marker="*", color="gold", label="Zero Noise Linear Fit Intercept", zorder=10, alpha=1, s=60)
plt.scatter([0], [popt[2]], marker="+", color="gold", label="Zero Noise Quadratic Fit Intercept", zorder=10, alpha=1, s=60)
#plt.scatter([0], [richardson_extrapolation], marker="x", color="gold", label="Zero Noise Richardson Extrapolation", zorder=10, alpha=1, s=60)

plt.plot(linx, [0.5 for i in linx], "--", label="Decohered Noise HOP", color="black")

plt.ylabel("Heavy Output Probability")
plt.xlabel("$\\lambda$")
plt.grid()
fig = plt.gcf()
fig.set_size_inches(7, 3.2)
plt.legend(ncol=2, prop={'size': 9})
plt.tight_layout()
plt.savefig("figures/ZNE_QV_"+strat+"_"+str(N)+".pdf")
plt.close()
