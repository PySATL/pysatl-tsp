# PySATL-TSP

[status-shield]: https://img.shields.io/github/actions/workflow/status/PySATL/pysatl-tsp/.github/workflows/check.yaml?branch=main&event=push&style=for-the-badge&label=Checks
[status-url]: https://github.com/PySATL/pysatl-tsp/blob/main/.github/workflows/check.yaml
[license-shield]: https://img.shields.io/github/license/PySATL/pysatl-tsp.svg?style=for-the-badge&color=blue
[license-url]: LICENSE

[![Checks][status-shield]][status-url]
[![MIT License][license-shield]][license-url]

PySATL **Time Series Processing** subproject (*abbreviated pysatl-tsp*) is a module designed for adaptive processing of time series data with a focus on streaming architecture. It implements a chain of responsibility pattern that enables building complex data processing pipelines with minimal boilerplate code, making it suitable for real-time applications and large dataset analysis.

---

## Requirements

- Python 3.10+
- Poetry 1.8.0+

## Installation

Clone the repository:

```bash
git clone https://github.com/PySATL/pysatl-tsp
```

Install dependencies:

```bash
poetry install
```

## Basic Pipeline Example:

```python
from pysatl_tsp.provider import SimpleDataProvider
from pysatl_tsp.processor import MappingHandler
from pysatl_tsp.scrubber import LinearScrubber

# Create a data source
data = [i for i in range(100)]
provider = SimpleDataProvider(data)

# Define a simple processing pipeline:
# 1. Create windows of 10 elements with 50% overlap
# 2. Calculate the average of each window
pipeline = (
    provider
    | LinearScrubber(window_length=10, shift_factor=0.5)
    | MappingHandler(map_func=lambda window: sum(window.values) / len(window))
)

# Process the data
results = []
for avg in pipeline:
    results.append(avg)
    
print(f"Number of windows: {len(results)}")
print(f"First 3 window averages: {results[:3]}")

# Visualize results
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(results)
plt.title('Window Averages')
plt.xlabel('Window Index')
plt.ylabel('Average Value')
plt.grid(True)
plt.show()

```

## Development

Install requirements

```bash
poetry install --with dev
```

## Pre-commit

Install pre-commit hooks:

```shell
poetry run pre-commit install
```

Starting manually:

```shell
poetry run pre-commit run --all-files --color always --verbose --show-diff-on-failure
```

## License

This project is licensed under the terms of the **MIT** license. See the [LICENSE](LICENSE) for more information.