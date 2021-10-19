## Provisioning the NGC Merlin Training kernel on managed notebooks


```

BASE_ADDRESS=notebooks.googleapis.com
LOCATION=us-central1
PROJECT_ID=jk-mlops-dev
AUTH_TOKEN=$(gcloud auth application-default print-access-token)

RUNTIME_BODY="{ 
  'access_config': {
      'access_type': 'SINGLE_USER',
      'runtime_owner': <USER ID>
  }, 
  'virtual_machine': { 
    'virtual_machine_config': {
      'data_disk': {
         'initialize_params': {
            'disk_size_gb': 200,
            'disk_type': 'PD_SSD',
          }
      },
      'container_images': {
        'repository':'nvcr.io/nvidia/merlin/merlin-training',
        'tag':'21.09'
        },  
      'machine_type':'a2-highgpu-1g',
      'accelerator_config': {
        'type': 'NVIDIA_TESLA_A100',
        'core_count': '1'
      }    
    } 
  },
}"


RUNTIME_NAME=jk-merlin-dev

curl -X POST https://${BASE_ADDRESS}/v1/projects/$PROJECT_ID/locations/$LOCATION/runtimes?runtime_id=$RUNTIME_NAME -d "${RUNTIME_BODY}" \
 -H "Content-Type: application/json" \
 -H "Authorization: Bearer $AUTH_TOKEN" -v

```

### Check the provisioning status


```
curl -X GET https://${BASE_ADDRESS}/v1/projects/$PROJECT_ID/locations/$LOCATION/runtimes/${RUNTIME_NAME} \
-H "Content-Type: application/json"  -H "Authorization: Bearer $AUTH_TOKEN" -v
```

### Delete the instance

```
curl -X DELETE https://${BASE_ADDRESS}/v1/projects/$PROJECT_ID/locations/$LOCATION/runtimes/{RUNTIM_NAME}  \
-H "Content-Type: application/json"  -H "Authorization: Bearer $AUTH_TOKEN" -v
```
