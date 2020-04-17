# ANDS

## Overview

Implementation of our submitting paper Network Traffic Data Generation usingAutoregressive Neural Models.

ANDS is an autoregressive data synthesizer that can generate synthetic time-series multi-variable data.
A flexible architecture supports to generate multi-variable data with any combination of continuous & discrete attributes.

- **Dependency capturing**: ANDS learns dependency in a time-window context rectangular,
  including both temporal dependency and attribute dependency.
- **Network structure**: ANDS uses CNN to extract dependent context features, gaussian mixture layers to predict continuous attributes,
  and softmax layers to predict discrete attributes.
- **Application dataset**: UGR'16: A New Dataset for the Evaluation of Cyclostationarity-Based Network IDSs [[link]](https://nesg.ugr.es/nesg-ugr16/)


## Installment and settings

### Requirements

**ANDS** has been developed and tested on [Python 3.5](https://www.python.org/downloads/)

> git clone https://github.com/an-anonymous-repo/ANDS.git

### Data Format

**ANDS** expects the input data to be a table given as either a `numpy.ndarray` or a
`pandas.DataFrame` object with two types of columns:

* **Continuous Columns**: Columns that contain numerical values and which can take any value.
* **Discrete columns**: Columns that only contain a finite number of possible values, wether
these are string values or not.

## Play with our demo on UGR16 network traffic flow dataset

With attached trained models, just run the follow command to generate synthetic UGR'16 data:

> python3 runner.py -genfolder

## To train your own data

### Step 1. Prepare your data

Prepare the training data and test data in two folders under the input_data folders. (We already put example csv files)

> input_folder/train_set/*.csv

### Step 2. Make the nn-fitting data

By running the preprocessing tool, we're able to automatically read in a folder of csv raw data 
and transfer the raw data to time-window (X, y) pair set.

> python3 makedata.py

### Step 3. Define the model settings by config.ini

Following the example config.ini under the src folder, we can set the attributes index of the training data, as well as the discrete attribute category number. 


| settings | explanation |
| --- | --- |
| continuous_list | the index of the continuous attributes in your dataframe |
| discrete_list | the index of the discrete attributes in your dataframe | 
| discrete_category_num | the number of classification categories for each discrete attributes |
| memory_height | the time-window that wanted to be calucated as dependency |
| input_feature_num | total number of attribute in your dataframe (width)|


### Step 4. Fitting the ANDS model

> python3 runner.py -train
