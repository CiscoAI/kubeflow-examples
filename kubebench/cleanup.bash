#!/usr/bin/env bash
source variables.bash

cd ${APP_NAME}
pwd

ks delete ${KF_ENV} -c ${JOB_NAME}
kubectl get pods -n ${NAMESPACE}

ks delete ${KF_ENV} -c kubebench-quickstarter-volume
ks delete ${KF_ENV} -c kubebench-quickstarter-service

kubectl get pv -n ${NAMESPACE}
kubectl get pvc -n ${NAMESPACE}

ks delete ${KF_ENV} -c kubeflow-argo
ks delete ${KF_ENV} -c centraldashboard
ks delete ${KF_ENV} -c tf-job-operator
kubectl get pods -n ${NAMESPACE}

ks env rm ${KF_ENV}
