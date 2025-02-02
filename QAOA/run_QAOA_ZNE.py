import networkx as nx
from qiskit import QuantumCircuit
import random
import math
from qiskit_aer import AerSimulator
import copy
import numpy as np
from circuit_unoptimization import circuit_unoptimization
from qiskit import QuantumCircuit, transpile
from qiskit_aer.noise import amplitude_damping_error, depolarizing_error, NoiseModel
from scipy.stats import linregress
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def quadratic(x, a, b, c):
	return a*x**2 + b*x + c
def calculate_cost(cut_vec, G):
    cut_weight = 0
    for edge in list(G.edges()):
        (i, j) = edge
        if cut_vec[i] != cut_vec[j]:
            cut_weight += 1
    return cut_weight
def create_QAOA_circuit(G, p, betas, gammas, n):
	qc = QuantumCircuit(n)
	for i in range(n):
		qc.h(i)
	qc.barrier()
	for p_idx in range(p):
		for e in list(G.edges()):
			qc.cx(e[0], e[1])
			qc.rz(-1*gammas[p_idx], e[1])
			qc.cx(e[0], e[1])
		qc.barrier()
		for i in range(n):
			qc.rx(betas[p_idx], i)
	qc.measure_all()
	return qc
def measure_sample_cuts(counts, G):
	cuts = []
	for bits in counts:
		counter = counts[bits]
		bitstring = bits[::-1]
		cut = calculate_cost(bitstring, G)
		for i in range(counter):
			cuts.append(cut)
	return cuts
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

strat = "P_c"
n = 12
G = nx.random_regular_graph(3, n, seed=1)
p = 2

get_binary = lambda x, n: format(x, "b").zfill(n)
all_possible_cuts = [bin(k)[2:].rjust(n, "0") for k in range(2**n)]
all_costs = [calculate_cost(cut, G) for cut in all_possible_cuts]
maxcost = max(all_costs)

print("maxcut=", maxcost)

#Random Angles, which do not work well:
#betas = [random.uniform(0, math.pi) for i in range(p)]
#gammas = [random.uniform(0, math.pi) for i in range(p)]

#Fixed angles from https://arxiv.org/abs/2107.00677
#gammas = [0.616]
#betas = [0.393]
gammas = [0.488, 0.898]
betas = [0.555, 0.293]

qc = create_QAOA_circuit(G, p, betas, gammas, n)
print(qc.count_ops())
qc = transpile(qc, basis_gates=["u3", "cx"], optimization_level=3)
print(qc.count_ops())
init = dict(qc.count_ops())["u3"]+dict(qc.count_ops())["cx"]

shots = 10_000_000
backend = AerSimulator()
result = backend.run(qc, shots=shots).result()
counts = result.get_counts()
cuts = measure_sample_cuts(counts, G)
print("no noise:", np.mean(cuts))
no_noise_cut_value = np.mean(cuts)

x = []
y = []
for idx in range(1, 35):
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
	cuts = measure_sample_cuts(counts, G)
	experimental_prob = np.mean(cuts)
	
	print(scale_factor, experimental_prob)
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


plt.scatter([0], [no_noise], marker="o", color="red", label="Noiseless Cut Value", zorder=5)
plt.scatter([0], [res.intercept], marker="*", color="gold", label="Zero Noise Linear Fit Intercept", zorder=10, alpha=1, s=60)
plt.scatter([0], [popt[2]], marker="+", color="gold", label="Zero Noise Quadratic Fit Intercept", zorder=10, alpha=1, s=60)



if strat == "P_c":
	plt.title("Concatenated Strategy")
if strat == "P_r":
	plt.title("Random Strategy")

plt.ylabel("Cut Value")
plt.xlabel("$\\lambda$")
plt.grid()
fig = plt.gcf()
fig.set_size_inches(7, 3.2)
plt.legend(ncol=2, prop={'size': 9})
plt.tight_layout()
plt.savefig("figures/ZNE_QAOA_"+strat+"_"+str(n)+"_"+str(p)+".pdf")
plt.close()
