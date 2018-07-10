#!/usr/bin/env bash
# Uncomment the following two lines to step through each command and to print
# the command being executed.
#set -x
#trap read debug

# Start the training job

# read common variables (between installation, training, and serving)
source variables.bash

cd ${APP_NAME}
pwd

ks generate kubebench-job ${JOB_NAME} --name=${JOB_NAME}

ks param set ${JOB_NAME} name ${JOB_NAME}
ks param set ${JOB_NAME} namespace ${NAMESPACE}
ks param set ${JOB_NAME} config_image gcr.io/xyhuang-kubeflow/kubebench-configurator:v20180522-1
ks param set ${JOB_NAME} report_image gcr.io/xyhuang-kubeflow/kubebench-tf-cnn-csv-reporter:v20180522-1
ks param set ${JOB_NAME} config_args -- --config-file=${PVC_MOUNT}/config/${CONFIG_NAME}.yaml
ks param set ${JOB_NAME} report_args -- --output-file=${PVC_MOUNT}/output/results.csv
ks param set ${JOB_NAME} pvc_name ${PVC_NAME}
ks param set ${JOB_NAME} pvc_mount ${PVC_MOUNT}

ks apply ${KF_ENV} -c ${JOB_NAME}

# Check that the container is up and running
kubectl get pods -n ${NAMESPACE}
