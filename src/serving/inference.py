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
"""Model Inference."""

import struct
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
import numpy as np
import json


def get_inference_input(data, binary_data):
    """An object of InferInput class is used to describe
    input tensor for an inference request.
    Parameters
    ----------
    "name" : str
        The name of input whose data will be described by this object
    "shape" : list
        The shape of the associated input.
    "datatype" : str
        The datatype of the associated input.
    ----------
    Based on the data format:
    Json:
        "data" : array
            The data vector
    
    Binary:
        "parameters": {
            "binary_data_size": int
         }
    """
    inputs = []

    col_names = list(data.columns)
    col_dtypes = [np.int32] * len(col_names)

    for i, col in enumerate(data.columns):
        d = data[col].astype(col_dtypes[i])
        d = d.values.reshape(len(d), 1)
        inputs.append(httpclient.InferInput(col_names[i], d.shape, np_to_triton_dtype(col_dtypes[i])))
        inputs[i].set_data_from_numpy(d, binary_data)
        
    return inputs

def get_inference_request(inputs, request_id):
    infer_request = {}
    if request_id != "":
        infer_request['id'] = request_id
    infer_request['inputs'] = [
        this_input._get_tensor() for this_input in inputs
    ]
    request_body = json.dumps(infer_request)
    json_size = len(request_body)
    binary_data = None
    for input_tensor in inputs:
        raw_data = input_tensor._get_binary_data()
        if raw_data is not None:
            if binary_data is not None:
                binary_data += raw_data
            else:
                binary_data = raw_data

    if binary_data is not None:
        request_body = struct.pack(
            '{}s{}s'.format(len(request_body), len(binary_data)),
            request_body.encode(), binary_data)
        return request_body, json_size

    return infer_request, request_body, None