# circuit-unoptimization

Implements the quantum circuit unoptimization elementary recipe from
[arXiv:2311.03805](https://arxiv.org/pdf/2311.03805). Unoptimizing a circuit increases its depth and gate count, which
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

## Testing

To run the tests:

```sh
poetry run pytest
```

## Contributing

To guarantee that both linter and formatter run before each commit, please install the pre-commit hook with:

```sh
poetry run pre-commit install
```