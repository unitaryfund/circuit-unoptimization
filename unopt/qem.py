from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from qiskit import QuantumCircuit, transpile
from qiskit.providers import Backend

from mitiq import zne

from unopt.generator import generate_random_two_qubit_gate_circuit
from unopt.noise_models import depolarizing_noise_model


def execute(
    circuit: QuantumCircuit,
    backend: Backend,
    shots: float,
    noise_model: NoiseModel | None = None,
) -> float:
    """Executor for a Qiskit circuit."""
    circuit_with_measurement = circuit.copy()
    circuit_with_measurement.measure_all()

    # Transpile the circuit:
    compiled_circuit = transpile(
        circuit_with_measurement,
        backend,
        basis_gates=noise_model.basis_gates if noise_model is not None else None,
        optimization_level=0,
    )

    # Execute the circuit:
    result = backend.run(
        compiled_circuit, noise_model=noise_model, shots=shots
    ).result()
    counts = result.get_counts()

    # Calculate expectation value of Z on qubit 0
    total_counts = sum(counts.values())
    expectation = 0.0
    for outcome, count in counts.items():
        # Reverse the outcome string due to Qiskit's little-endian ordering
        bitstring = outcome[::-1]
        if bitstring[0] == "0":
            expectation += count / total_counts
        else:
            expectation -= count / total_counts
    return expectation


if __name__ == "__main__":
    # Hardware/simulator settings:
    backend = AerSimulator()
    noise_model = depolarizing_noise_model(error=0.01)
    # noise_model = amplitude_damping_noise_model()
    shots = 10_000

    # ZNE settings:
    # fold_method = zne.scaling.fold_gates_at_random
    fold_method = zne.scaling.fold_global
    extrapolation_method = zne.RichardsonFactory
    scale_factors_fold = [1, 3, 5]

    # Circuit unoptimization settings:
    scale_factors_unopt = [1, 3, 5]
    elementary_recipe_factors = [0, 3, 5]

    # Circuit properties:
    num_qubits = 5
    depth = 15
    qc = generate_random_two_qubit_gate_circuit(num_qubits=num_qubits, depth=depth)

    # Ideal value for random circuit consisting of two-qubits will always be 1.
    ideal_value = 1
