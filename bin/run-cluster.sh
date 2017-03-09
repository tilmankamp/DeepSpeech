#!/bin/bash

PROGNAME=$(basename $0)
ps_count=1
worker_count=2

export ds_importer="ted"
export ds_train_batch_size=1
export ds_dev_batch_size=1
export ds_test_batch_size=1
export ds_limit_train=100
export ds_limit_dev=50
export ds_limit_test=50
export ds_epochs=5
export ds_display_step=2
export ds_validation_step=3

# Generating the parameter server addresses
index=0
while [ "$index" -lt "$ps_count" ]
do
  ps_hosts[$index]="localhost:$((index + 2000))"
  ((index++))
done
ps_hosts=$(printf ",%s" "${ps_hosts[@]}")
ps_hosts=${ps_hosts:1}

# Generating the worker addresses
index=0
while [ "$index" -lt "$worker_count" ]
do
  worker_hosts[$index]="localhost:$((index + 3000))"
  ((index++))
done
worker_hosts=$(printf ",%s" "${worker_hosts[@]}")
worker_hosts=${worker_hosts:1}

export ds_ps_hosts=$ps_hosts
export ds_worker_hosts=$worker_hosts


# Starting the parameter servers
index=0
while [ "$index" -lt "$ps_count" ]
do
  CUDA_VISIBLE_DEVICES="" ds_job_name=ps ds_task_index=$index python -u DeepSpeech.py 2>&1 | sed 's/^/[ps     '"$index"'] /' &
  echo "Started ps $index"
  ((index++))
done

# Starting the workers
index=1
while [ "$index" -lt "$worker_count" ]
do
  CUDA_VISIBLE_DEVICES="$index" ds_job_name=worker ds_task_index=$index python -u DeepSpeech.py 2>&1 | sed 's/^/[worker '"$index"'] /' &
  echo "Started worker $index"
  ((index++))
done

#JOBS=$(echo $(jobs -lp))
#trap "set -x; kill $JOBS" EXIT

CUDA_VISIBLE_DEVICES="0" ds_job_name=worker ds_task_index=0 python -u DeepSpeech.py 2>&1 | sed 's/^/[worker 0] /'


#while [ 1 ]; do sleep 1; test $? -gt 128 && break; done
