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
    
    triton_client.get_model_repository_index()
    
    triton_client.load_model(model_name="deepfm_nvt")
    
    triton_client.load_model(model_name="deepfm")
    
    triton_client.load_model(model_name="deepfm_ens")
    


if __name__ == '__main__':
    
    main()
