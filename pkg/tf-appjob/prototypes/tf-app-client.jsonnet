// @apiVersion 0.1
// @name io.ksonnet.pkg.tf-app-client
// @description A TensorFlow app client
// @shortDescription Run the TensorFlow app client
// @param name string Name for the app client.
// @param app_serving_ip string IP of the serving service
// @param image string Image of the app client
// @optionalParam app_serving_port string 9000 Port of the serving pod
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
local host = import "param://app_serving_ip";
local port = import "param://app_serving_port";
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
         "app": "app-client",
      }
   },
   "spec": {
      "replicas" : std.parseInt(replicas),
      "selector": {
         "matchLabels": {
            "app": "app-client"
         }
      },
      "template": {
         "metadata": {
            "labels": {
               "app": "app-client",
            }
         },
         "spec": {
            "containers": [
               {
                  "name": "app-client",
                  "image": image,
                  "env": [
                     {
                        "name": "TF_MODEL_SERVER_HOST",
                        "value": host
                     },
                     {
                        "name": "TF_MODEL_SERVER_PORT",
                        "value": port
                     }   
                  ],
                  "ports": [
                     {
                        "containerPort": 80
                     }
                  ],
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
         "app": "app-client"
      }
   },
   "spec": {
      "type": "LoadBalancer",
      "loadBalancerIP": lb,
      "ports": [
         {
            "port": 80,
            targetPort: 80
         }
      ],
      "selector": {
         "app": "app-client"
      }
   }
};

std.prune(k.core.v1.list.new([deployment,service]))
