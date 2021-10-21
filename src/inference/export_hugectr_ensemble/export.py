import argparse
import logging

import nvtabular as nvt
from nvtabular.inference.triton import export_hugectr_ensemble


NUM_SLOTS = 26
MAX_NNZ = 2
NUM_OUTPUTS = 1
ENSEMBLE_NAME = 'deepfm'

CATEGORICAL_COLUMNS=["C" + str(x) for x in range(1, 27)]
CONTINUOUS_COLUMNS=["I" + str(x) for x in range(1, 14)]
LABEL_COLUMNS = ['label']


def export_ensemble(
    workflow_path,
    model_graph_path,
    model_params_path,
    embedding_vector_size,
    ensemble_path,
    batch_size):
    
    workflow = nvt.Workflow.load(workflow_path)
    
    hugectr_params = dict()
    hugectr_params["config"] = model_graph_path
    hugectr_params["slots"] = NUM_SLOTS
    hugectr_params["max_nnz"] = MAX_NNZ
    hugectr_params["embedding_vector_size"] = embedding_vector_size
    hugectr_params["n_outputs"] = NUM_OUTPUTS
    
    export_hugectr_ensemble(
        workflow=workflow,
        hugectr_model_path=model_params_path,
        hugectr_params=hugectr_params,
        name=ENSEMBLE_NAME,
        output_path=ensemble_path,
        label_columns=LABEL_COLUMNS,
        cats=CATEGORICAL_COLUMNS,
        conts=CONTINUOUS_COLUMNS,
        max_batch_size=batch_size,
    )
    

def parse_args():
    """Parses command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--workflow_path',
                        type=str,
                        required=False,
                        default='/criteo_data/criteo_processed_parquet_0.6_t1/workflow',
                        help='Path to preprocessing workflow')
    parser.add_argument('--model_graph_path',
                        type=str,
                        required=False,
                        default='/criteo_data/model_t1/graph/deepfm.json',
                        help='Path to model graph')
    parser.add_argument('--model_params_path',
                        type=str,
                        required=False,
                        default='/criteo_data/model_t1/parameters/',
                        help='Path to model parameters')
    parser.add_argument('--ensemble_path',
                        type=str,
                        required=False,
                        default='/criteo_data/model_ensemble_t1/',
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
    logging.info("Exporting ensemble to: {}".format(args.ensemble_path))
    export_ensemble(
        args.workflow_path,
        args.model_graph_path,
        args.model_params_path,
        args.embedding_vector_size,
        args.ensemble_path,
        args.max_batch_size)