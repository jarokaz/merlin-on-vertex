def analyze_dataset_op(
    datasets,
    workflow_path,
    split_name = 'train',
    device_limit_frac = 0.4,
    device_pool_frac = 0.5,
    part_mem_frac = 0.5
):
    from preprocessing import etl
    import logging
    import os

    logging.basicConfig(level=logging.INFO)
    workflow = {}

    # Retrieve `split_name` from metadata
    data_path = datasets[split_name]

    # Create data transformation workflow. This step will only 
    # calculate statistics based on the transformations
    logging.info('Creating transformation workflow.')
    criteo_workflow = etl.create_criteo_nvt_workflow()

    # Create Dask cluster
    logging.info('Creating Dask cluster.')
    client = etl.create_transform_cluster(device_limit_frac, device_pool_frac)

    logging.info('Creating dataset.')
    # Create dataset to be fitted
    dataset = etl.create_parquet_dataset(
        data_path=data_path,
        part_mem_frac=part_mem_frac,
        client=client
    )

    logging.info('Starting workflow fitting')
    criteo_workflow = etl.analyze_dataset(criteo_workflow, dataset)
    logging.info('Finished generating statistics for dataset.')

    etl.save_workflow(criteo_workflow, os.path.join('/gcs', workflow_path))
    logging.info('Workflow saved to GCS')

    workflow['workflow'] = os.path.join('/gcs', workflow_path)
    workflow['datasets'] = datasets


if __name__ == '__main__':
    datasets = {}
    datasets['train'] = 'gs://renatoleite-criteo-partial/converted/train'
    datasets['valid'] = 'gs://renatoleite-criteo-partial/converted/valid'
    workflow_path = 'renatoleite-criteo-partial/workflow'
    analyze_dataset_op(datasets, workflow_path)