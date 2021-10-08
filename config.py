BASE_IMAGE_NAME = 'us-east1-docker.pkg.dev/renatoleite-mldemos/docker-images/nvt-conda'

PROJECT_ID = 'renatoleite-mldemos'
REGION = 'us-central1'
BUCKET = 'renatoleite-criteo-partial'
LOCATION = 'us'

VERTEX_PARENT = f'projects/{PROJECT_ID}/locations/{REGION}'

SAVED_WORKFLOW_PATH = f'{BUCKET}/saved_workflow/' # Where to write the converted PARQUET files
OUTPUT_CONVERTED = f'{BUCKET}/converted/' # Location to write the transformed data
OUTPUT_TRANSFORMED = f'{BUCKET}/transformed_data/'

STAGING_BUCKET = f'gs://{BUCKET}/temp'