#!/bin/bash

/usr/lib/zoo/bin/spark-submit-with-zoo.sh --master local[2] --num-executors 2 --executor-cores 8 --driver-memory 40g --executor-memory 40g /home/leelau/xray_flask_webApp/ver3//logic.py
