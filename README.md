# ANDS

Implementation of our submitting paper Network Traffic Data Generation usingAutoregressive Neural Models.

ANDS is a autoregressive data synthesizer that can generate synthetic time-series multi-variable data with high fidelity.


# Install

## Requirements

**ANDS** has been developed and tested on [Python 3.5](https://www.python.org/downloads/)

# Data Format

**GANS** expects the input data to be a table given as either a `numpy.ndarray` or a
`pandas.DataFrame` object with two types of columns:

* **Continuous Columns**: Columns that contain numerical values and which can take any value.
* **Discrete columns**: Columns that only contain a finite number of possible values, wether
these are string values or not.
