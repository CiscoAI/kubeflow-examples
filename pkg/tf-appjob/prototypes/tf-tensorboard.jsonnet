// @apiVersion 0.1
// @name io.ksonnet.pkg.tf-tensorboard
// @description A TensorFlow app client
// @shortDescription Run the TensorFlow app client
// @param name string Name for the app client.
// @param logdir string IP of the serving service
// @param image string Image of the app client
// @optionalParam app_serving_port string 6006 Port of the serving pod
// @optionalParam lbip string null client external loadbalancer ip
// @optionalParam replicas string 1 Number of client replica deployment
// @optionalParam namespace string null Namespace to use for the components. It is automatically inherited from the environment if not set.

local k = import "k.libsonnet";
local util = import "ciscoai/tf-appjob/util.libsonnet";

// updatedParams uses the environment namespace if
// the namespace parameter is not explicitly set
local updatedParams = params {
  namespace: if params.namespace == "null" then env.namespace else params.namespace,
};

local name = import "param://name";
local namespace = updatedParams.namespace;
local replicas = import "param://replicas";
local logdir = import "param://logdir";
local lbip = import "param://lbip";
local lb = 
  if lbip == "null" then
    ""
  else
    lbip;

local image = import "param://image";

local deployment = {
   "apiVersion": "apps/v1",
   "kind": "Deployment",
   "metadata": {
      "name": name,
      "namespace": namespace,
      "labels": {
         "app": "tensorboard",
      }
   },
   "spec": {
      "replicas" : std.parseInt(replicas),
      "selector": {
         "matchLabels": {
            "app": "tensorboard"
         }
      },
      "template": {
         "metadata": {
            "labels": {
               "app": "tensorboard",
            }
         },
         "spec": {
            "containers": [
               {
                  "name": "tensorboard",
                  "image": image,
                  "env": [
                     {
                        "name": "logdir",
                        "value": logdir
                     }   
                  ],
                  "ports": [
                     {
                        "containerPort": 6006
                     }
                  ],
                  "volumeMounts": [
                     {
                        "mountPath": /mnt,
			"name": nfs
                     }
                  ],
               }
            ],
	    "volumes": [
                    {
                        "name": "nfs",
                        "persistentVolumeClaim": {
                        "claimName": "nfs"
                    }
                }
            ]
         }
      }
   }
};

local service = {
   "apiVersion": "v1",
   "kind": "Service",
   "metadata": {
      "name": name,
      "namespace": namespace,
      "labels": {
         "app": "tensorboard"
      }
   },
   "spec": {
      "type": "NodePort",
      "ports": [
        {
          "port": 6006
        }
      ],
      "selector": {
         "app": "tensorboard"
      }
   }
};

std.prune(k.core.v1.list.new([deployment,service]))
