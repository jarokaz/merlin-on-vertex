import nvtabular as nvt
from nvtabular.inference.triton import export_hugectr_ensemble


WORKFLOW_PATH = '/criteo_data/criteo_processed_parquet_0.6_t1/workflow'
MODEL_GRAPH_PATH = '/criteo_data/model_t1/graph/deepfm.json'
MODEL_PARAMS_PATH = '/criteo_data/model_t1/parameters/'
OUTPUT_PATH = '/criteo_data/model_ensemble_t1/'
NUM_SLOTS = 26
MAX_NNZ = 2
EMBEDDING_VECTOR_SIZE = 11
NUM_OUTPUTS = 1
ENSEMBLE_NAME = 'deepfm'


CATEGORICAL_COLUMNS=["C" + str(x) for x in range(1, 27)]
CONTINUOUS_COLUMNS=["I" + str(x) for x in range(1, 14)]
LABEL_COLUMNS = ['label']
MAX_BATCH_SIZE = 64

workflow = nvt.Workflow.load(WORKFLOW_PATH)

hugectr_params = dict()
hugectr_params["config"] = MODEL_GRAPH_PATH
hugectr_params["slots"] = NUM_SLOTS
hugectr_params["max_nnz"] = MAX_NNZ
hugectr_params["embedding_vector_size"] = EMBEDDING_VECTOR_SIZE
hugectr_params["n_outputs"] = NUM_OUTPUTS
export_hugectr_ensemble(
    workflow=workflow,
    hugectr_model_path=MODEL_PARAMS_PATH,
    hugectr_params=hugectr_params,
    name=ENSEMBLE_NAME,
    output_path=OUTPUT_PATH,
    label_columns=LABEL_COLUMNS,
    cats=CATEGORICAL_COLUMNS,
    conts=CONTINUOUS_COLUMNS,
    max_batch_size=MAX_BATCH_SIZE
)