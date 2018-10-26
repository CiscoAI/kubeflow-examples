local params = std.extVar("__ksonnet/params").components.tf_mnist;

local k = import 'k.libsonnet';
local workflows = import 'tf_mnist.libsonnet';
local namespace = params.namespace;

// TODO(jlewi): Can we make name default so some random unique value?
// I didn't see any routines in the standard library for datetime or random.
local name = params.name;

local prowEnv = workflows.parseEnv(params.prow_env);
local prowDict = workflows.parseEnvToDict(params.prow_env);
local bucket = params.bucket;
std.prune(k.core.v1.list.new([workflows.parts(namespace, name, params).e2e(prowDict, prowEnv, bucket)]))
