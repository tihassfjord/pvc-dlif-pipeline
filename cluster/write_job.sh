#!/usr/bin/env bash
# Write a Kubernetes job file for stage 05 from cluster/job_template.yaml.
#
#   bash cluster/write_job.sh --job-name <name> --bundle /storage/<bundle folder> --node <host> \
#        -- --network fcdlif --representation uncorrected --folds 1 --runs 1
#
# Everything after "--" is passed to scripts/05_train.py. Submit the written
# file yourself with your cluster's tool (e.g. "kubectl apply -f
# cluster/jobs/<name>.yaml"). Use a job name no other job has: some tools
# replace a running job of the same name.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
job_name=""; bundle=""; node=""
while [ $# -gt 0 ]; do
    case "$1" in
        --job-name) job_name="$2"; shift 2 ;;
        --bundle)   bundle="$2"; shift 2 ;;
        --node)     node="$2"; shift 2 ;;
        --)         shift; break ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done
if [ -z "$job_name" ] || [ -z "$bundle" ] || [ -z "$node" ]; then
    echo "need --job-name, --bundle and --node" >&2; exit 2
fi
mkdir -p "$here/jobs"
sed -e "s|__JOB_NAME__|$job_name|" -e "s|__BUNDLE__|$bundle|" -e "s|__NODE__|$node|" \
    -e "s|__TRAIN_ARGUMENTS__|$*|" "$here/job_template.yaml" > "$here/jobs/$job_name.yaml"
echo "written: $here/jobs/$job_name.yaml"
