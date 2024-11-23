# circuit-unoptimization
Quantum circuit unoptimization with ZNE

## Setup

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

To guarantee that both linter and formatter run before each commit, please install the pre-commit hook with

```sh
poetry run pre-commit install
```