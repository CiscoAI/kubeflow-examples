#!/usr/local/env bash

## Namespace to be used in k8s cluster for your application
NAMESPACE=kubeflow

## Ksonnet app name
APP_NAME=mnist
VAGRANT_KS_APP=/home/vagrant/kf/ks_app

## GITHUB version for official kubeflow components
KUBEFLOW_GITHUB_VERSION=v0.4.1

## GITHUB version for ciscoai components
CISCOAI_GITHUB_VERSION=master

## Ksonnet environment name
KF_ENV=default

## Name of the NFS Persistent Volume
NFS_PVC_NAME=nfs

## Used in training.bash
# Enviroment variables for mnist training job (See mnist_model.py)
TF_DATA_DIR=/mnt/data
TF_MODEL_DIR=/mnt/model
NFS_MODEL_PATH=/mnt/export
TF_EXPORT_DIR=${NFS_MODEL_PATH}

# If you want to use your own image,
# make sure you have a dockerhub account and change
# DOCKER_BASE_URL and IMAGE below.
DOCKER_BASE_URL=gcr.io/cpsg-ai-demo
IMAGE=${DOCKER_BASE_URL}/tf-mnist-demo:v1
#docker build . --no-cache  -f Dockerfile -t ${IMAGE}
#docker push ${IMAGE}

# Used in portf.bash and webapp.bash
# If using without an application, source this file before using
PORT=9000
VAGRANT_IP=10.10.10.10
export TF_MODEL_SERVER_PORT=${PORT}

# Used in webapp.bash
DOCKER_HUB=gcr.io
DOCKER_USERNAME=cpsg-ai-demo
DOCKER_IMAGE=mnist-client-latest
WEBAPP_FOLDER=webapp
