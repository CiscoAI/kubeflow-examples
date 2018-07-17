#!/usr/bin/env bash

# Uncomment the following two lines to step through each command and to print
# the command being executed.
#set -x
#trap read debug

# Start TF serving on the trained results

# Note that `tfserving's modelPath` is set to `tfmnistjob's TF_EXPORT_DIR` so
# that tf serving pod automatically picks up the training results when training
# is completed.

# Read variables
source variables.bash

cd ${APP_NAME}
pwd

#2. Create namespace if not present

ks generate tf-serving tfserving --name=${APP_NAME}

# Set tf serving job specific environment params
ks param set tfserving modelPath ${NFS_MODEL_PATH}
ks param set tfserving modelStorageType nfs
ks param set tfserving nfsPVC ${NFS_PVC_NAME}

# Deploy and start serving
ks apply ${KF_ENV} -c tfserving

# Port forward to access the serving port locally
#SERVING_POD_NAME=`kubectl -n ${NAMESPACE} get pod -l=app=mnist -o jsonpath='{.items[0].metadata.name}'`
#kubectl -n ${NAMESPACE} port-forward ${SERVING_POD_NAME} 9000:9000 &
