import argparse
import copy
import glob
import logging
import os
import json
import shutil

import nvtabular as nvt
from nvtabular.inference.triton import export_hugectr_ensemble


NUM_SLOTS = 26
MAX_NNZ = 2
NUM_OUTPUTS = 1

CATEGORICAL_COLUMNS=["C" + str(x) for x in range(1, 27)]
CONTINUOUS_COLUMNS=["I" + str(x) for x in range(1, 14)]
LABEL_COLUMNS = ['label']

HUGECTR_CONFIG_FILENAME = "ps.json"


def create_hugectr_backend_config(
    ensemble_root_path,
    model_registry_path,
    model_name):
    
    config_dict = dict()
    network_file = os.path.join(model_registry_path,
                                model_name,
                                "1",
                                f'{model_name}.json')

    pathname=os.path.join(ensemble_root_path,
                          model_name,
                          '1', 
                          f'{model_name}_dense_*.model')
    dense_file = glob.glob(pathname)[0]
    
    pathname=os.path.join(ensemble_root_path,
                          model_name,
                          '1', 
                          f'{model_name}[0-9]_sparse_*.model')
    sparse_files = glob.glob(pathname)
    
    config_dict['supportlonglong'] = True
    model_config = dict()
    model_config['model'] = model_name
    model_config['sparse_files'] = sparse_files
    model_config['dense_file'] = dense_file
    model_config['network_file'] = network_file
    config_dict['models'] = [model_config]
    
    return config_dict
    

def export_ensemble(
    workflow_path,
    saved_model_path,
    embedding_vector_size,
    output_path,
    model_name_prefix,
    model_registry_path,
    max_batch_size):
          
    workflow = nvt.Workflow.load(workflow_path)
    
    hugectr_params = dict()
    graph_filename = f'{model_name_prefix}.json'
    hugectr_params["config"] = os.path.join(model_registry_path,
                                            model_name_prefix,
                                            "1",
                                            graph_filename)
    hugectr_params["slots"] = NUM_SLOTS
    hugectr_params["max_nnz"] = MAX_NNZ
    hugectr_params["embedding_vector_size"] = embedding_vector_size
    hugectr_params["n_outputs"] = NUM_OUTPUTS
    
    export_hugectr_ensemble(
        workflow=workflow,
        hugectr_model_path=saved_model_path,
        hugectr_params=hugectr_params,
        name=model_name_prefix,
        output_path=output_path,
        label_columns=LABEL_COLUMNS,
        cats=CATEGORICAL_COLUMNS,
        conts=CONTINUOUS_COLUMNS,
        max_batch_size=max_batch_size,
    )
    
    hugectr_backend_config = create_hugectr_backend_config(
        ensemble_root_path=output_path,
        model_registry_path=model_registry_path,
        model_name=model_name_prefix)
    
    with open(os.path.join(output_path, HUGECTR_CONFIG_FILENAME), 'w') as f:
        json.dump(hugectr_backend_config, f)
    
 
    

def parse_args():
    """Parses command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--workflow_path',
                        type=str,
                        required=False,
                        default='/criteo_data/criteo_processed_parquet_0.6/workflow',
                        help='Path to preprocessing workflow')
    parser.add_argument('--saved_model_path',
                        type=str,
                        required=False,
                        default='/criteo_data/model_21.09',
                        help='Path to a saved model')
    #parser.add_argument('--model_graph_path',
    #                    type=str,
    #                    required=False,
    #                    default='/criteo_data/model_21.09/graph/deepfm.json',
    #                    help='Path to model graph')
    parser.add_argument('--model_registry_path',
                        type=str,
                        required=False,
                        default='/models',
                        help='A root path to registry embedd in hugectr model config.pbtxt and ps.json')
    #parser.add_argument('--model_params_path',
    #                    type=str,
    #                    required=False,
    #                    default='/criteo_data/model_21.09/parameters',
    #                    help='Path to model parameters')
    parser.add_argument('--output_path',
                        type=str,
                        required=False,
                        default='/criteo_data/models',
                        help='Base output dir')
    parser.add_argument('--model_name_prefix',
                        type=str,
                        required=False,
                        default='deepfm',
                        help='Path to output ensemble')
    parser.add_argument('--max_batch_size',
                        type=int,
                        required=False,
                        default=64,
                        help='Max batch size')
    parser.add_argument('--embedding_vector_size',
                        type=int,
                        required=False,
                        default=11,
                        help='embedding_vector_size')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%m-%y %H:%M:%S')
    
    args = parse_args()

    logging.info("Exporting ensemble to: {}".format(args.output_path))
    export_ensemble(
        workflow_path=args.workflow_path,
        saved_model_path=args.saved_model_path,
        embedding_vector_size=args.embedding_vector_size,
        output_path=args.output_path,
        model_name_prefix=args.model_name_prefix,
        model_registry_path=args.model_registry_path,
        max_batch_size=args.max_batch_size)
    
 
   