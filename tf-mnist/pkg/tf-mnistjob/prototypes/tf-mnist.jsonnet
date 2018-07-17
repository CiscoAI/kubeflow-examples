// @apiVersion 0.1
// @name io.ksonnet.pkg.tf-mnistjob
// @description A TensorFlow Mnist job
// @shortDescription Run the TensorFlow Mnist job.
// @param name string Name for the job.
// @optionalParam namespace string null Namespace to use for the components. It is automatically inherited from the environment if not set.
// @optionalParam args string null Comma separated list of arguments to pass to the job
// @optionalParam envs string null Comma separated list of environment variables to pass to the job
// @optionalParam image string null The docker image to use for the job.
// @optionalParam image_gpu string null The docker image to use when using GPUs.
// @optionalParam image_pull_secrets string null Comma-delimited list of secret names to use credentials in pulling your docker images.
// @optionalParam num_masters number 1 The number of masters to use
// @optionalParam num_ps number 1 The number of ps to use
// @optionalParam num_workers number 1 The number of workers to use
// @optionalParam num_gpus number 0 The number of GPUs to attach to workers.
// @optionalParam volume_mount_path string null The volume mount point.

local k = import "k.libsonnet";
local deployment = k.extensions.v1beta1.deployment;
local container = deployment.mixin.spec.template.spec.containersType;
local podTemplate = k.extensions.v1beta1.podTemplate;
local util = import "ciscoai/tf-mnistjob/util.libsonnet";

// updatedParams uses the environment namespace if
// the namespace parameter is not explicitly set
local updatedParams = params {
  namespace: if params.namespace == "null" then env.namespace else params.namespace,
};

local tfJob = import "kubeflow/tf-job/tf-job.libsonnet";

local name = import "param://name";
local namespace = updatedParams.namespace;

local argsParam = import "param://args";
local args =
  if argsParam == "null" then
    []
  else
    std.split(argsParam, ",");

local envsParam = import "param://envs";
local envs =
  if envsParam == "null" then
    []
  else
    util.nameValuePair(std.split(envsParam,","));



local image = import "param://image";
local imageGpu = import "param://image_gpu";
local imagePullSecrets = import "param://image_pull_secrets";
local numPs = import "param://num_ps";
local numMasters = import "param://num_masters";
local numWorkers = import "param://num_workers";
local numGpus = import "param://num_gpus";
local path = import "param://volume_mount_path";

local mountPath = if path == "null" then "/mnt" else path;

local workerSpec = if numGpus > 0 then
  tfJob.parts.tfJobReplica("WORKER", numWorkers, args, imageGpu, imagePullSecrets, numGpus)
else
  tfJob.parts.tfJobReplica("WORKER", numWorkers, args, image, imagePullSecrets);

local terminationPolicy = if numMasters == 1 then
  tfJob.parts.tfJobTerminationPolicy("MASTER", 0)
else
  tfJob.parts.tfJobTerminationPolicy("WORKER", 0);

local replicas = std.map(function(s)
                           s {
                             template+: {
                               spec+: {
                                  volumes: [ { name: "nfsvolume", persistentVolumeClaim: { claimName: "nfs" } } ] ,
                                 
                                  containers: [
                                   s.template.spec.containers[0] {
                                           volumeMounts: [ { mountPath: mountPath , name: "nfsvolume" } ],
                                           env: envs 
                                
                                   },
                                 ],
                               },
                             },
                           },
                         std.prune([tfJob.parts.tfJobReplica("MASTER", numMasters, args, image, imagePullSecrets) ,workerSpec, tfJob.parts.tfJobReplica("PS", numPs, args, image, imagePullSecrets)]));


local job =
  if numWorkers < 1 then
    error "num_workers must be >= 1"
  else
    if numPs < 1 then
      error "num_ps must be >= 1"
    else
      tfJob.parts.tfJob(name, namespace, replicas, null) + {
        spec+: {
          tfImage: image,
          terminationPolicy: terminationPolicy,
        },
      };

std.prune(k.core.v1.list.new([job]))
