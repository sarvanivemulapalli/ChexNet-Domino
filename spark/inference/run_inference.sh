#!/bin/bash
 
/opt/Intel/bin/spark-submit-with-zoo.sh --master yarn --deploy-mode cluster --num-executors 4 --executor-cores 32 --driver-memory 300g --executor-memory 300g /mnt/chestxray_inference.py
