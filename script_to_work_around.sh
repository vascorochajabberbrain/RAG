#!/bin/bash

for i in $(seq 73 80); do
    echo "Running csv_ingestion.py with argument: $i"
    python csv_ingestion.py "$i"
done
