#!/bin/bash

PROGNAME=$(basename $0)
worker_count=2

export ds_importer="ldc93s1"
export ds_train_batch_size=1
export ds_dev_batch_size=1
export ds_test_batch_size=1
export ds_epochs=50


function join_by { local IFS="$1"; shift; echo "$*"; }

function usage {
	# Display usage message on standard error
	echo "Usage: $PROGNAME" 1>&2
}

function clean_up {
	jobs -pr | xargs kill -9
}

function error_exit {
	# Display error message and exit
	echo "${PROGNAME}: ${1:-"Unknown Error"}" 1>&2
	clean_up 1
}

trap clean_up SIGHUP SIGINT SIGTERM

# Do stuff

index=0
while [ "$index" -lt "$worker_count" ]
do
  worker_hosts[$index]="localhost:$((index + 2223))"
  ((index++))
done
worker_hosts=$(printf ",%s" "${worker_hosts[@]}")
worker_hosts=${worker_hosts:1}

export ds_ps_hosts='localhost:2222'
export ds_worker_hosts=$worker_hosts


CUDA_VISIBLE_DEVICES="" ds_job_name=ps ds_task_index=0 python -u DeepSpeech.py 2>&1 | sed 's/^/[server  ] /' &
param_server=`jobs -p`
echo "Started parameter server with PID $param_server"

sleep 4

index=0
while [ "$index" -lt "$worker_count" ]
do
  CUDA_VISIBLE_DEVICES=$index ds_job_name=worker ds_task_index=$index python -u DeepSpeech.py 2>&1 | sed 's/^/[worker '"$index"'] /' &
  echo "Started worker $index with PID ${workers[$index]}"
	sleep 4
  ((index++))
done

while ps -p $param_server > /dev/null
do
  sleep 1
done
clean_up
