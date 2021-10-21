import os
import json 
import struct
import nvtabular.inference.triton as nvt_triton
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
import numpy as np
import pickle

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

def get_inference_input():
    from nvtabular.dispatch import get_lib
    df_lib = get_lib()
    BASE_DIR = os.environ.get("BASE_DIR", "/model/")

    # read in the workflow (to get input/output schema to call triton with)
    batch_path = os.path.join(BASE_DIR, "data")
    batch = df_lib.read_parquet(os.path.join(batch_path, "*.parquet"), num_rows=3)
    batch = batch[[x for x in batch.columns if x != "label"]]
    print(batch)

    inputs = []

    col_names = list(batch.columns)
    col_dtypes = [np.int32] * len(col_names)

    for i, col in enumerate(batch.columns):
        d = batch[col].values_host.astype(col_dtypes[i])
        d = d.reshape(len(d), 1)
        inputs.append(httpclient.InferInput(col_names[i], d.shape, np_to_triton_dtype(col_dtypes[i])))
        inputs[i].set_data_from_numpy(d)

    request_body, json_size = get_inference_request(inputs, '1')
    print("Add Header: Inference-Header-Content-Length: {}".format(json_size))
    print(request_body)
    with open('/model/data/criteo.dat', 'wb') as output_file:
        output_file.write(request_body)
        output_file.close() 