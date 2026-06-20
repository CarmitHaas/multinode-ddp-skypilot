# GPU run runbook (personal Nebius tenant, eu-north1)

The image and code are built and pushed cold beforehand, so this session only provisions GPUs,
runs the job, captures the two run outputs (`mk8s-ng-config.json` and `training_log.txt`) plus
the recording, then deletes the cluster so billing stops.

## Before I start
- Nebius console open on my tenant, with billing active.
- GPU quota: a fresh tenant often has 0 GPU quota. This is the one thing that can block the
  whole run, so check it first and, if 0, request an increase early (approval is not always
  instant). The CLI enumerates the quota (existence and usage state, not the number):
  `nebius quotas quota-allowance get-by-name --parent-id project-e00mzjcgpr00yq3j88hbzg --region eu-north1 --name compute.instance.gpu.l40s --format json`
  The actual numeric limit and the request button are on the Console Quotas page (eu-north1).
- Image already public on Docker Hub: `docker.io/carmithaas/nebius-trainer:v1`.
- `train_job.yaml` has one placeholder left: `<CLUSTER_NAME>`.
- Recording tools installed: `asciinema` (plus `agg` for GIFs) for the terminal, `peek` for the console.

## Target config (cost-conscious)
Preset `gpu-l40s-1gpu`, 2 nodes, 100 GB SSD. Model `facebook/opt-1.3b`, world size 2 (one GPU
per node, two nodes), 500 steps on wikitext-2. The YAML asks for `L40S:1`.

## Cost discipline
The two L40S nodes bill for the whole time the cluster exists, not just while training. So the
order is create, run, capture, delete, with no long idle gaps. Realistic wall time is well under
an hour. Tear down in step 8 the moment `training_log.txt` is saved.

---

## 1. Create the cluster and GPU node group (console; record with Peek)
Managed Kubernetes, Create cluster: default VPC, latest stable Kubernetes.
Add a node group:
- Preset `gpu-l40s-1gpu`  (alternative `gpu-h100-b-1gpu`, then set `accelerators: "H100:1"` in the YAML)
- Nodes: 2
- Disk: 100 GB SSD

Public Docker Hub means the nodes pull anonymously, so I skip the service-account step. Wait for
the cluster to reach Running (about 5 min). Note the cluster name, cluster ID, and node-group ID.

> Capture: Peek-record the cluster and node-group creation as a short GIF, and screenshot the
> cluster overview once it is Running.

## 2. Get kubeconfig and confirm 2 GPU nodes Ready (asciinema starts here)
```bash
export PATH="$HOME/.nebius/bin:$PATH"
asciinema rec ddp-run.cast          # start recording the terminal portion
nebius mk8s cluster get-credentials --id <CLUSTER_ID> --external --kubeconfig ~/.kube/config
kubectl get nodes                   # expect 2 nodes, STATUS Ready
```

## 3. Capture the node-group spec (mk8s-ng-config.json)
```bash
cd /home/develeap/AI/ddp-skypilot
nebius mk8s node-group get --id <NODE_GROUP_ID> --format json | jq '{metadata, spec}' > mk8s-ng-config.json
jq '.spec' mk8s-ng-config.json | head -40     # sanity: 2 nodes, L40S preset
```

## 4. Connect SkyPilot (local, against my own cluster)
Because this is my own tenant, I do not need a shared SkyPilot API server. The local `sky` CLI
starts its own API server and reads my kubeconfig.
```bash
sky api start             # start the local API server (free; no shared/remote server needed)
sky check kubernetes      # should list my mk8s cluster context as an enabled infra
# (only if I later choose to self-host a managed API server would I instead run: sky api login -e https://<endpoint>)
```
If `sky check kubernetes` does not see the context, confirm `kubectl config current-context`
points at the new cluster.

## 5. Point the job at this cluster (fill the one remaining placeholder)
Edit `train_job.yaml`:
- `infra: k8s/<CLUSTER_NAME>`  set to the real cluster name
- `image_id` is already `docker:docker.io/carmithaas/nebius-trainer:v1`

## 6. Launch and watch (the main moment for the recording)
```bash
cd /home/develeap/AI/ddp-skypilot
sky launch -c ddp-run train_job.yaml          # answer yes to the launch prompt
sky logs ddp-run                               # live logs: image pull, torchrun, NCCL init, steps
```
What good looks like: 2 nodes provision, image pulls, torchrun starts on both, NCCL prints its
init section across the two nodes, 500 steps run, then `[Done] Training complete`.

## 7. Capture the training log (training_log.txt, must contain NCCL init)
```bash
sky logs ddp-run > training_log.txt
grep -nc "NCCL INFO" training_log.txt                         # must be > 0
grep -nE "Bootstrap|NET/|Ring|Channel|comm |nccl version" training_log.txt | head
grep -n "Training complete" training_log.txt
# stop the asciinema recording now (Ctrl-D), then make the GIF:
agg ddp-run.cast docs/diagrams/ddp-run.gif
```

## 8. Tear down to stop billing (immediately after capture)
```bash
sky down ddp-run                 # removes the SkyPilot job/pods
```
Then delete the node group and the cluster in the console. Confirm with `kubectl get nodes`
that they are gone, and that the console shows no running GPU compute.

## 9. Bundle the run outputs
```bash
cd /home/develeap/AI/ddp-skypilot
zip ddp-run-outputs.zip mk8s-ng-config.json Dockerfile train.py train_job.yaml training_log.txt
unzip -l ddp-run-outputs.zip      # the five core files, no docs or scratch
```

## Fast triage (do not redesign mid-run)
- Quota denied or nodes never appear: L40S quota not granted yet. This is why step 0 checks quota first.
- Image will not pull: confirm the Docker Hub repo is Public and the tag matches the YAML.
- Pods Pending: `kubectl describe pod <p>`, usually GPU not schedulable yet or a taint; wait a minute.
- NCCL hangs at init: the two pods must reach each other; the run block derives MASTER_ADDR from
  the first `SKYPILOT_NODE_IPS`. Re-launch fresh if it stalls.
- OOM (only if I switch to opt-2.7b on L40S): drop `PER_DEVICE_TRAIN_BATCH_SIZE` to 2.
