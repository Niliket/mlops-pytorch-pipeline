#!/usr/bin/env bash
set -e

echo "=== 1. Checking Pod Status ==="
kubectl get pods -n ml-training

echo "=== 2. Checking Deployment & PVC ==="
kubectl get deployment,pvc,hpa -n ml-training

echo "=== 3. Testing Health Endpoint via Port-Forward ==="
kubectl port-forward svc/model-serving 8080:80 -n ml-training &
PF_PID=$!
sleep 3

curl -s http://localhost:8080/health | grep "healthy" && echo " Health Probe Passed"

kill $PF_PID
echo "=== Validation Complete ==="
