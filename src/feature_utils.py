# Copyright 2021 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
"""Dataset features."""

import numpy as np

from dataclasses import dataclass
from dataclasses import field

def _get_criteo_col_dtypes(): 
    col_dtypes = {}
    col_dtypes["label"] = np.int32
    for x in ["I" + str(i) for i in range(1, 14)]:
        col_dtypes[x] = np.int32
    for x in ["C" + str(i) for i in range(1, 27)]:
        col_dtypes[x] = 'hex'

    return col_dtypes

def _categorical_columns():
    return ["C" + str(x) for x in range(1, 27)]

def _numerical_columns():
    return ["I" + str(x) for x in range(1, 14)]

def _label_columns():
    return ['label']


@dataclass
class FeatureSpecs:
    '''This class contains configurations specific to the Criteo dataset.'''
    
    num_slots: int  = 26
    max_nnz: int = 2
    num_outputs: int = 1
    dtypes: dict = field(default_factory=_get_criteo_col_dtypes)
    categorical_columns: list = field(default_factory=_categorical_columns)
    continuous_columns: list = field(default_factory=_numerical_columns)
    label_columns: list = field(default_factory=_label_columns)





