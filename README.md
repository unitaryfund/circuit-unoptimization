# circuit-unoptimization

Supplemental software for *"Digital Zero-Noise Extrapolation with Quantum Circuit Unoptimization"* ([arXiv:XXX]()).

Implements the quantum circuit unoptimization elementary recipe from *"Quantum Circuit Unoptimization"*
([arXiv:2311.03805](https://arxiv.org/pdf/2311.03805)). Unoptimizing a circuit increases its depth and gate count, which
can lead to higher noise due to increased opportunities for errors. By deliberately adding gates that do not change the
overall computation, we can amplify the noise without altering the circuit's functionality. This serves as an alternate
method of noise-scaling for quantum error mitigation techniques like zero-noise extrapolation (ZNE).

## Installing

You will require Python 3.12 and [`poetry`](https://python-poetry.org/).

Once you have `poetry` installed, run:

```sh
poetry install
poetry shell
```

## Example

Consider the following arbitrary quantum circuit:

```py
>>> from qiskit import QuantumCircuit
>>> from unopt.recipe import unoptimize_circuit
>>> 
>>> # Create a 4-qubit fully connected graph state.
>>> num_qubits = 4
>>> qc = QuantumCircuit(num_qubits)
>>> 
>>> for qubit in range(num_qubits):
>>>     qc.h(qubit)
>>> 
>>> for i in range(num_qubits):
>>>     for j in range(i + 1, num_qubits):
>>>         qc.cz(i, j)
>>> print(qc)
     ┌───┐                  
q_0: ┤ H ├─■──■──■──────────
     ├───┤ │  │  │          
q_1: ┤ H ├─■──┼──┼──■──■────
     ├───┤    │  │  │  │    
q_2: ┤ H ├────■──┼──■──┼──■─
     ├───┤       │     │  │ 
q_3: ┤ H ├───────■─────■──■─
     └───┘                  
```

We can perform comparative benchmarks for ZNE with unoptimization as a scaling technique against the ideal, unmitigated,
and ZNE with folding

```
**Averages Across All Trials:
  Ideal Value: 0.06250000000000006
  Unmitigated Value: 0.003799999999999956
  ZNE + Fold Value: 0.00037500000000005443 (Error: 0.062125)
  ZNE + Unopt Value: 0.008441716269841373 (Error: 0.05405828373015868)
  Percent Improvement (Unmit): 8.59%
  Percent Improvement (ZNE + Fold): 14.92%
  Original Circuit Depth: 6
  Avg Folded Depths: [6.0, 18.0, 30.0]
  Avg Unoptimized Depths: [37.0, 65.0, 101.0]


Trial Details:
Trial 1:
  Ideal Value: 0.06250000000000006
  Unmitigated Value: 0.003799999999999956
  ZNE + Fold Value: 0.00037500000000005443
  ZNE + Fold Depths: [6, 18, 30]
  ZNE + Unopt Value: 0.008441716269841373
  ZNE + Unopt Depths: [37, 65, 101]**
```

Note that for certain circuits and runs, you may obtain different results as there is a randomized component to the
circuit unoptimization and extrapolation. 

One can also run the unoptimization procedure of [arXiv:2311.03805](https://arxiv.org/pdf/2311.03805) in isolation as
such:

```py
>>> # Apply one rounds of unoptimization to the circuit.
>>> unopt_qc = unoptimize_circuit(qc, 1)
>>> print(unopt_qc)

global phase: 4.4801
     ┌─────────────────────┐           ┌────────────────┐           ┌──────────────────────┐                                  ┌─────────────────────┐                                ┌──────────────────┐         »
q_0: ┤ U3(0.62409,π/2,π/2) ├──■────────┤ U3(π,-π/4,π/4) ├───────■───┤ U3(0.36348,1.4779,0) ├───■───────────────────────────■──┤ U3(2.4621,0,1.6637) ├────────────────────────■───────┤ U3(2.5975,0,π/2) ├──────■──»
     └┬───────────────────┬┘┌─┴─┐┌─────┴────────────────┴────┐┌─┴─┐┌┴──────────────────────┴┐  │                           │  └─────────────────────┘┌────────────────────┐┌─┴─┐┌────┴──────────────────┴───┐┌─┴─┐»
q_1: ─┤ U3(2.1322,π/2,-π) ├─┤ X ├┤ U3(1.8034,2.0685,-2.4328) ├┤ X ├┤ U3(1.0094,1.3772,-π/2) ├──┼────■──────────────────────┼─────────────■───────────┤ U3(π/2,-π,-1.5964) ├┤ X ├┤ U3(2.0383,2.2097,-2.8185) ├┤ X ├»
     ┌┴───────────────────┴┐└───┘└───────────────────────────┘└───┘└────────────────────────┘┌─┴─┐┌─┴─┐┌────────────────┐┌─┴─┐         ┌─┴─┐         ├───────────────────┬┘└───┘└───────────────────────────┘└───┘»
q_2: ┤ U3(π/2,-0.21066,-π) ├─────────────────────────────────────────────────────────────────┤ X ├┤ X ├┤ U3(0,0,1.0189) ├┤ X ├─────────┤ X ├─────────┤ U3(0.048388,0,-π) ├────────────────────────────────────────»
     └─────────────────────┘                                                                 └───┘└───┘└────────────────┘└───┘         └───┘         └───────────────────┘                                        »
q_3: ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────»
                                                                                                                                                                                                                  »
«      ┌────────────────────────┐                                          ┌──────────────────┐             ┌──────────────────────┐                                  ┌───────────────────────┐        »
«q_0: ─┤ U3(π,-0.59091,-2.8735) ├──────────────────────────────────■───────┤ U3(2.5156,0,π/2) ├──────■──────┤ U3(π/2,2.986,0.6101) ├──────■──────────────────────■────┤ U3(π,-2.6303,-2.6303) ├─────■──»
«     ┌┴────────────────────────┤     ┌─────────────────────────┐┌─┴─┐┌────┴──────────────────┴───┐┌─┴─┐┌───┴──────────────────────┴───┐  │                      │    └───────────────────────┘   ┌─┴─┐»
«q_1: ┤ U3(2.5802,-2.6044,-π/2) ├──■──┤ U3(1.0094,-π/2,-1.0918) ├┤ X ├┤ U3(2.0383,2.2097,-2.8185) ├┤ X ├┤ U3(1.4839,-0.13902,-0.55533) ├──┼──────────────────────┼────────────────────────────────┤ X ├»
«     └─────────────────────────┘┌─┴─┐└┬────────────────────────┤└───┘└───────────────────────────┘└───┘└──────────────────────────────┘┌─┴─┐┌────────────────┐┌─┴─┐┌────────────────────────────┐└───┘»
«q_2: ───────────────────────────┤ X ├─┤ U3(0.99038,0.75265,-π) ├───────────────────────────────────────────────────────────────────────┤ X ├┤ U3(0,0,2.2352) ├┤ X ├┤ U3(1.633,-0.4035,0.080188) ├─────»
«                                └───┘ └────────────────────────┘                                                                       └───┘└────────────────┘└───┘└────────────────────────────┘     »
«q_3: ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────»
«                                                                                                                                                                                                      »
«           ┌────────────────┐          ┌──────────────────────────┐        ┌──────────────────────┐           ┌────────────────────┐         ┌──────────────────────────────┐                                   »
«q_0: ──────┤ U3(π,-π/4,π/4) ├───────■──┤ U3(2.5684,2.4992,2.8748) ├──■─────┤ U3(1.0824,-π/2,-π/2) ├────■──────┤ U3(0.15576,-π,π/2) ├──────■──┤ U3(0.80295,-1.7413,-0.98507) ├───────■───────────────────────────»
«     ┌─────┴────────────────┴────┐┌─┴─┐└─┬─────────────────────┬──┘  │     └──────────────────────┘    │      └────────────────────┘      │  └──────────────────────────────┘       │                           »
«q_1: ┤ U3(1.8527,2.1084,-2.5222) ├┤ X ├──┤ U3(2.1322,-π/8,π/2) ├─────┼─────────────────────────────────┼──────────────────────────────────┼────────────────────────────────────■────┼───────────────────■───────»
«     └───────────────────────────┘└───┘  └─────────────────────┘   ┌─┴─┐┌───────────────────────────┐┌─┴─┐┌────────────────────────────┐┌─┴─┐ ┌───────────────────────────┐  ┌─┴─┐  │  ┌─────────────┐  │       »
«q_2: ──────────────────────────────────────────────────────────────┤ X ├┤ U3(1.9504,2.7175,0.68731) ├┤ X ├┤ U3(1.1033,0.32306,-2.2097) ├┤ X ├─┤ U3(1.3489,1.3177,-2.6481) ├──┤ X ├──┼──┤ U3(π/2,0,π) ├──┼────■──»
«                                                                   └───┘└───────────────────────────┘└───┘└────────────────────────────┘└───┘ └───────────────────────────┘  └───┘┌─┴─┐└─────────────┘┌─┴─┐┌─┴─┐»
«q_3: ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤ X ├───────────────┤ X ├┤ X ├»
«                                                                                                                                                                                  └───┘               └───┘└───┘»
«                    
«q_0: ───────────────
«                    
«q_1: ───────────────
«                    
«q_2: ───────────────
«     ┌─────────────┐
«q_3: ┤ U3(π/2,0,π) ├
«     └─────────────┘

# 

```

## Plots

To generate the plots used in the paper, they can be run and generated directly by:

```sh
python unopt/plot.py
```

Note that generating these files from scratch can take several minutes. The progress of the computations used for the
plots are shown when the above is run.

## Testing

To run the tests:

```sh
poetry run pytest
```

## Linting/Formatting

To guarantee that both linter and formatter run before each commit, please install the pre-commit hook with:

```sh
poetry run pre-commit install
```
