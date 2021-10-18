import tritonhttpclient
import tritonclient.http as httpclient

from tritonclient.utils import np_to_triton_dtype


import warnings
import os
import numpy as np

from nvtabular.dispatch import get_lib



BASE_DIR='/criteo_data'



def main():
    try:
        triton_client = tritonhttpclient.InferenceServerClient(url="localhost:8000", verbose=True)
        print("client created.")
    except Exception as e:
        print("channel creation failed: " + str(e))

    
    warnings.filterwarnings("ignore")

    triton_client.is_server_live()
    
    df_lib = get_lib()
    
    batch_path = os.path.join(BASE_DIR, 'criteo_parquet')
    batch = df_lib.read_parquet(os.path.join(batch_path, "day_0.parquet"), num_rows=3)
    batch = batch[[x for x in batch.columns if x != "label"]]

    inputs = []

    col_names = list(batch.columns)
    col_dtypes = [np.int32] * len(col_names)

    for i, col in enumerate(batch.columns):
        d = batch[col].values_host.astype(col_dtypes[i])
        d = d.reshape(len(d), 1)
        inputs.append(httpclient.InferInput(col_names[i], d.shape, np_to_triton_dtype(col_dtypes[i])))
        inputs[i].set_data_from_numpy(d)
        
    print('*** Invoking prediction')
    outputs = [httpclient.InferRequestedOutput("OUTPUT0")]

    response = triton_client.infer("deepfm_ens", inputs, request_id="1", outputs=outputs)

    print("predicted sigmoid result:\n", response.as_numpy("OUTPUT0"))
    

if __name__ == '__main__':
    
    main()