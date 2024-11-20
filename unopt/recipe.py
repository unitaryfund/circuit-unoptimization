"""Recipe steps from arXiv:X"""
import random
import warnings

import numpy as np

from qiskit.circuit.library import UnitaryGate

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import random_unitary, Operator
from qiskit.transpiler.passes import Decompose
from qiskit.transpiler import PassManager


def elementary_recipe(qc: QuantumCircuit, iterations=1, strategy="P_c"):
    """Apply the elementary recipe to a quantum circuit multiple times.

    Args:
        qc: The input quantum circuit.
        iterations: The number of times to apply the recipe.
        strategy: The strategy used in gate insertion. Options are "P_c" or "P_r".

    Returns:
        new_qc: The quantum circuit after applying the recipe.
    """
    new_qc = qc.copy()
    for _ in range(iterations):
        # Step 1: Gate Insertion
        new_qc, B1_info = gate_insert(new_qc, strategy)

        # Step 2: Gate Swapping
        new_qc = gate_swap(new_qc, B1_info)

        # Step 3: Decomposition
        new_qc = decompose_circuit(new_qc)

        # Step 4: Synthesis
        new_qc = synthesize_circuit(new_qc)

    return new_qc


def gate_insert(qc: QuantumCircuit, strategy="P_c"):
    """Insert a two-qubit gate A and its Hermitian conjugate A† between two gates B1 and B2.
    Args:
        qc: The input quantum circuit.
        strategy: The strategy to select the pair of two-qubit gates. Options are "P_c" or "P_r".

    Returns:
        new_qc: The modified quantum circuit with A and A† inserted.
        B1_info: Information about gate B1 (index, qubits, gate).
    """
    # Collect all two-qubit gates with their indices and qubits
    two_qubit_gates = []

    for idx, instruction in enumerate(qc.data):
        instr = instruction.operation
        qargs = instruction.qubits
        cargs = instruction.clbits

        if len(qargs) == 2:
            qubit_indices = [qc.find_bit(qarg).index for qarg in qargs]
            two_qubit_gates.append({"index": idx, "qubits": qubit_indices, "gate": instr})

    found_pair = False
    B1_idx = B1_qubits = B1_gate = shared_qubit = None 

    if strategy == "P_c":
        # Strategy P_c: Find a pair of gates that share a common qubit
        for i in range(len(two_qubit_gates)):
            for j in range(i + 1, len(two_qubit_gates)):
                qubits_i = set(two_qubit_gates[i]["qubits"])
                qubits_j = set(two_qubit_gates[j]["qubits"])
                common_qubits = qubits_i & qubits_j
                
                if len(common_qubits) == 1:
                    B1_idx = two_qubit_gates[i]["index"]
                    B1_qubits = two_qubit_gates[i]["qubits"]
                    B1_gate = two_qubit_gates[i]["gate"]
                    shared_qubit = list(common_qubits)[0]
                    found_pair = True
                    break

            if found_pair:
                break

    elif strategy == "P_r":
        # Strategy P_r: Randomly select a two-qubit gate as B1
        if two_qubit_gates:
            gate_info = random.choice(two_qubit_gates)
            B1_idx = gate_info["index"]
            B1_qubits = gate_info["qubits"]
            B1_gate = gate_info["gate"]
            shared_qubit = B1_qubits[0]  # Choose the first qubit as shared
            found_pair = True
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Available strategies are 'P_c' and 'P_r'.")

    if not found_pair:
        warnings.warn("No suitable pair of two-qubit gates found. Skipping gate insertion.")
        return qc, None  # Return the original circuit unmodified

    # Generate a random two-qubit unitary A and its adjoint A†
    A = random_unitary(4)
    A_dag = A.adjoint()

    # Choose qubits for A and A† insertion
    all_qubits = set(range(qc.num_qubits))
    other_qubits = list(all_qubits - set(B1_qubits))

    if not other_qubits:
        warnings.warn("Not enough qubits to perform gate insertion. Skipping.")
        return qc, None  # Return the original circuit unmodified

    third_qubit = other_qubits[0]
    shared_qubit = shared_qubit

    # Map indices back to qubits
    qubit_map = {qc.find_bit(q).index: q for q in qc.qubits}

    # Create a new circuit and insert A and A†
    new_qc = QuantumCircuit(*qc.qregs, *qc.cregs)

    # Copy the gates up to and including B1
    for instruction in qc.data[: B1_idx + 1]:
        instr = instruction.operation
        qargs = instruction.qubits
        cargs = instruction.clbits
        new_qc.append(instr, qargs, cargs)

    # Insert A on qubits [shared_qubit, third_qubit]
    qubits_for_A = [qubit_map[shared_qubit], qubit_map[third_qubit]]

    # Insert A†, A on the same qubits
    new_qc.unitary(A_dag, qubits_for_A, label=r"$A^{\dagger}$")
    new_qc.unitary(A, qubits_for_A, label="A")

    # Copy the remaining gates
    for instruction in qc.data[B1_idx + 1 :]:
        instr = instruction.operation
        qargs = instruction.qubits
        cargs = instruction.clbits
        new_qc.append(instr, qargs, cargs)

    # Prepare B1_info for gate_swap function
    B1_info = {
        "index": B1_idx,
        "qubits": B1_qubits,
        "gate": B1_gate,
        "shared_qubit": shared_qubit,
        "third_qubit": third_qubit,
        "A": A,
        "A_dag": A_dag,
    }

    return new_qc, B1_info


def gate_swap(qc: QuantumCircuit, B1_info: dict) -> QuantumCircuit:
    r"""Swap the B1 gate with the A† gate in the circuit, replacing A† with \widetilde{A^\dagger}.
    Args:
        qc: The input quantum circuit.
        B1_info: Information about gate B1, including its index, qubits, and the A, A† gates.
    Returns:
        new_qc: The modified quantum circuit with B1 and A† swapped.
    """
    B1_idx = B1_info["index"]
    B1_qubits = B1_info["qubits"]
    B1_gate = B1_info["gate"]
    A = B1_info["A"]
    shared_qubit = B1_info["shared_qubit"]
    third_qubit = B1_info["third_qubit"]
    
    # Map qubit indices to qubit objects
    qubit_map = {qc.find_bit(q).index: q for q in qc.qubits}

    # Get the operators
    B1_operator = Operator(B1_gate)
    A_operator = Operator(A)
    A_dagger_operator = A_operator.adjoint()

    # Determine the qubits involved
    qubits_involved = sorted(set(B1_qubits + [shared_qubit, third_qubit]))
    qubits_involved_objs = [qubit_map[q] for q in qubits_involved]
    num_qubits_involved = len(qubits_involved)

    # Create mapping from qubit indices to positions
    qubit_positions = {q: idx for idx, q in enumerate(qubits_involved)}

    # Build B1_operator_full
    B1_operator_full = Operator(np.eye(2 ** num_qubits_involved))
    B1_qubit_positions = [qubit_positions[q] for q in B1_qubits]
    B1_operator_full = B1_operator_full.compose(B1_operator, qargs=B1_qubit_positions)

    # Build A_dagger_operator_full
    A_dagger_operator_full = Operator(np.eye(2 ** num_qubits_involved))
    A_dagger_qubits = [shared_qubit, third_qubit]
    A_dagger_qubit_positions = [qubit_positions[q] for q in A_dagger_qubits]
    A_dagger_operator_full = A_dagger_operator_full.compose(A_dagger_operator, qargs=A_dagger_qubit_positions)

    # Compute B1_operator_full_dagger
    B1_operator_full_dagger = B1_operator_full.adjoint()

    # Compute \widetilde{A^\dagger}
    widetilde_A_dagger_operator = B1_operator_full_dagger.dot(A_dagger_operator_full).dot(B1_operator_full)

    # Create UnitaryGate from \widetilde{A^\dagger}
    widetilde_A_dagger_gate = UnitaryGate(widetilde_A_dagger_operator.data, label=r"$\widetilde{A^{\dagger}}$")

    # Create a new quantum circuit
    new_qc = QuantumCircuit(*qc.qregs, *qc.cregs)

    # Copy the gates up to B1_idx
    for i in range(B1_idx):
        instruction = qc.data[i]
        new_qc.append(instruction.operation, instruction.qubits, instruction.clbits)

    # Insert \widetilde{A^\dagger} at position B1_idx
    new_qc.append(widetilde_A_dagger_gate, qubits_involved_objs)

    # Insert B1 gate at position B1_idx + 1
    new_qc.append(B1_gate, [qubit_map[q] for q in B1_qubits])

    # Copy the remaining gates, skipping the original A_dagger gate
    for i in range(B1_idx + 2, len(qc.data)):
        instruction = qc.data[i]
        new_qc.append(instruction.operation, instruction.qubits, instruction.clbits)

    return new_qc


def decompose_circuit(qc: QuantumCircuit) -> QuantumCircuit:
    """Decompose multi-qubit unitary gates into elementary gates.

    Args:
        qc: The quantum circuit to decompose.

    Returns:
        new_qc: The decomposed quantum circuit.
    """
    return PassManager(Decompose()).run(qc)


def synthesize_circuit(qc: QuantumCircuit, optimization_level: int = 0) -> QuantumCircuit:
    """Synthesize the circuit using a specified optimization level.

    Args:
        qc: The quantum circuit to synthesize.
        optimization_level: The optimization level for transpilation.

    Returns:
        new_qc: The synthesized quantum circuit.
    """
    return transpile(qc, optimization_level=optimization_level, basis_gates=["u1", "u2", "u3", "cx"])
