from qiskit_aer.noise import NoiseModel

from qiskit import QuantumCircuit, transpile
from qiskit.providers import Backend

from mitiq import zne


def execute(
    circuit: QuantumCircuit, 
    backend: Backend, 
    shots: float, 
    noise_model: NoiseModel | None = None
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
    result = backend.run(compiled_circuit, noise_model=noise_model, shots=shots).result()
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