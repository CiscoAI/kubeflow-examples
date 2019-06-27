#!/bin/bash -e

cd ~/KFLab/tf-mnist/mnist

ks registry add kubeflow https://github.com/kubeflow/kubeflow/tree/master/kubeflow
ks registry add ciscoai-global github.com/CiscoAI/ks-packages/tree/master/pkg

ks pkg install kubeflow/jupyter

ks generate centraldashboard centraldashboard
ks generate jupyter jupyter 
ks generate jupyter-web-app jupyter-web-app
ks generate notebook-controller notebook-controller

ks apply pipelines -c centraldashboard 
ks apply pipelines -c jupyter
ks apply pipelines -c jupyter-web-app
ks apply pipelines -c notebook-controller

