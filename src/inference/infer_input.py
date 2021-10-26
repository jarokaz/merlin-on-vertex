import struct
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
import numpy as np
import json

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

def get_inference_input(data, binary_data):
    inputs = []

    col_names = list(data.columns)
    col_dtypes = [np.int32] * len(col_names)

    for i, col in enumerate(data.columns):
        d = data[col].astype(col_dtypes[i])
        d = d.values.reshape(len(d), 1)
        inputs.append(httpclient.InferInput(col_names[i], d.shape, np_to_triton_dtype(col_dtypes[i])))
        inputs[i].set_data_from_numpy(d, binary_data)
    return inputs