import numpy as np
import os
import warnings

import tritonhttpclient
import tritonclient.http as httpclient

from tritonclient.utils import np_to_triton_dtype


_examples = {
    'I1': [5, 32, 0], 
    'I2': [110, 3, 233], 
    'I3': [0, 5, 1], 
    'I4': [16, 0, 146], 
    'I5': [0, 1, 1], 
    'I6': [1, 0, 0], 
    'I7': [0, 0, 0], 
    'I8': [14, 61, 99], 
    'I9': [7, 5, 7], 
    'I10': [1, 0, 0], 
    'I11': [0, 1, 1], 
    'I12': [306, 3157, 3101], 
    'I13': [0, 5, 1], 
    'C1': [1651969401, -436994675, 1651969401], 
    'C2': [-501260968, -1599406170, -1382530557], 
    'C3': [-1343601617, 1873417685, 1656669709], 
    'C4': [-1805877297, -628476895, 946620910], 
    'C5': [951068488, 1020698403, -413858227], 
    'C6': [1875733963, 1875733963, 1875733963], 
    'C7': [897624609, -1424560767, -1242174622], 
    'C8': [679512323, 1128426537, -772617077], 
    'C9': [1189011366, 502653268, 776897055], 
    'C10': [771915201, 2112471209, 771915201], 
    'C11': [209470001, 1716706404, 209470001], 
    'C12': [-1785193185, -1712632281, 309420420], 
    'C13': [12976055, 12976055, 12976055], 
    'C14': [-1102125769, -1102125769, -1102125769], 
    'C15': [-1978960692, -205783399, -150008565], 
    'C16': [1289502458, 1289502458, 1289502458], 
    'C17': [-771205462, -771205462, -771205462], 
    'C18': [-1206449222, -1578429167, 1653545869], 
    'C19': [-1793932789, -1793932789, -1793932789], 
    'C20': [-1014091992, -20981661, -1014091992], 
    'C21': [351689309, -1556988767, 351689309], 
    'C22': [632402057, -924717482, 632402057], 
    'C23': [-675152885, 391309800, -675152885], 
    'C24': [2091868316, 1966410890, 883538181], 
    'C25': [809724924, -1726799382, -10139646], 
    'C26': [-317696227, -1218975401, -317696227]}


class VertexEndpointClient(object):
    """
    A convenience wrapper around Vertex AI Prediction REST API.
    """
    
    def __init__(self, service_endpoint):
        logging.info(
            "Setting the AI Platform Prediction service endpoint: {}".format(
                service_endpoint))
        credentials, _ = google.auth.default()
        self._authed_session = AuthorizedSession(credentials)
        self._service_endpoint = service_endpoint
    
    def predict(self, project_id, model, version, signature, instances):
        """
        Invokes the predict method on the specified signature.
        """
        
        url = '{}/v1/projects/{}/models/{}/versions/{}:predict'.format(
            self._service_endpoint, project_id, model, version)
            
        request_body = {
            'signature_name': signature,
            'instances': instances
        }
    
        response = self._authed_session.post(url, data=json.dumps(request_body))
        return response
    
    
def prepare_inference_request(
    examples: dict,
    binary_extension: bool=False):
    
    def _prepare_kfs_inference_request(inputs):
        
        inference_request = inputs
        
        return inference_request

    inputs = []
    for col_name, values in examples.items():
        d = np.array(values, dtype=np.int32)
        d = d.reshape(len(d), 1)
        inputs.append(httpclient.InferInput(col_name, d.shape, np_to_triton_dtype(np.int32)))
        inputs[len(inputs)-1].set_data_from_numpy(d)
    
    if binary_extension:
        inference_request = httpclient.InferenceServerClient.generate_request_body(inputs)
    else:
        inference_request = _prepare_kfs_inference_request(inputs)
        
    return inference_request




def main():
    try:
        triton_client = tritonhttpclient.InferenceServerClient(url="localhost:8000", verbose=True)
        print("client created.")
    except Exception as e:
        print("channel creation failed: " + str(e))

    warnings.filterwarnings("ignore")

    triton_client.is_server_live()
    
    #payload = prepare_inference_request(_examples, binary_extension=True)
    #inputs = payload
    
    
    #return
    
    inputs = []
    for col_name, values in _examples.items():
        d = np.array(values, dtype=np.int32)
        d = d.reshape(len(d), 1)
        inputs.append(httpclient.InferInput(col_name, d.shape, np_to_triton_dtype(np.int32)))
        
        inputs[len(inputs)-1].set_data_from_numpy(d)
    print('*** Invoking prediction')
    outputs = [httpclient.InferRequestedOutput("OUTPUT0")]

    response = triton_client.infer("deepfm_ens", inputs, request_id="1", outputs=outputs)

    print("predicted sigmoid result:\n", response.as_numpy("OUTPUT0"))
    

if __name__ == '__main__':
    main()
    
    