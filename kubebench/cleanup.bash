#!/usr/bin/env bash
source variables.bash

cd ${APP_NAME}
pwd

ks delete ${KF_ENV} -c ${JOB_NAME}
kubectl get pods -n ${NAMESPACE}

ks delete ${KF_ENV} -c nfs-volume
kubectl get pv -n ${NAMESPACE}
kubectl get pvc -n ${NAMESPACE}

ks delete ${KF_ENV} -c nfs-server

ks delete ${KF_ENV} -c kubeflow-argo
ks delete ${KF_ENV} -c kubeflow-core
kubectl get pods -n ${NAMESPACE}

ks env rm ${KF_ENV}
