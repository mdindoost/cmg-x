#!/bin/bash

# List of lambda values to test
LAMBDAS=(0.0 1.0 10.0 100.0)
LOG_DIR="logs/lambda_sweep"
mkdir -p "$LOG_DIR"

for LAMBDA in "${LAMBDAS[@]}"; do
    LOG_FILE="${LOG_DIR}/phi_gamma_lambda_${LAMBDA}.csv"
    echo "▶ Running φγ model with λ=${LAMBDA} → ${LOG_FILE}"
    python train_phi_gamma_gcn.py --lambda_phi_gamma=$LAMBDA --log_file=$LOG_FILE
done
