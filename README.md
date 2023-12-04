# How to run

In this example, we seek to infer a model in a collaborative cloud-edge way. In this case, we infer the the shallow layers/parts of the model on the end side and the deep layers/parts on the cloud side.

## Step 1: Export the models refer to `README.md` in `prepare_models`

Please export the models and put them into s3 object storage.

## Step 2: Build the docker images refer to `dockerfile` in `codes/cloud/cloud.dockerfile` and `codes/edge/edge.dockerfile`

Please build the docker images and push them into our registry `cr.scut-smil.cn`

## Step 3: Run by `kustomize`

cd directory `k8s_resources` and `kubectl kustomize ./`

Tips for debugging: you can use `kustomize build ./` to render the actual K8S resources