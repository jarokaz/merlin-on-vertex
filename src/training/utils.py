from nvtabular.columns.schema import ColumnSchema, Schema
from nvtabular.tags import Tags
import json

def retrieve_cardinalities(schema_path):
    
    schema = Schema.load_protobuf(schema_path)
    cardinalities = {
        key: value.properties['embedding_sizes']['cardinality'] 
        for key, value in schema.column_schemas.items()
        if Tags.CATEGORICAL in value.tags}
    
    return cardinalities
    
    