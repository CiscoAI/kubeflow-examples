#!/usr/local/env bash

## Namespace to be used in k8s cluster for your application
NAMESPACE=kubeflow

## Ksonnet app name
APP_NAME=mnist

## GITHUB version for official kubeflow components
KUBEFLOW_GITHUB_VERSION=v0.2.0-rc.0

## GITHUB version for ciscoai components
CISCOAI_GITHUB_VERSION=master

## Ksonnet environment name
KF_ENV=nativek8s

## Name of the NFS Persistent Volume
NFS_PVC_NAME=nfs

## Used in training.bash
# Enviroment variables for mnist training job (See mnist_model.py)
TF_DATA_DIR=/mnt/data
TF_MODEL_DIR=/mnt/model
NFS_MODEL_PATH=/mnt/export
TF_EXPORT_DIR=${NFS_MODEL_PATH}

# Make sure you have a dockerhub account
DOCKER_BASE_URL=docker.io/amsaha
IMAGE=${DOCKER_BASE_URL}/tfmodel
#docker build . --no-cache  -f Dockerfile -t ${IMAGE}
#docker push ${IMAGE}
