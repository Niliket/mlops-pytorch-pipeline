# MLOps PyTorch Pipeline

An end-to-end Machine Learning deployment pipeline featuring containerized PyTorch training, FastAPI model inference, CI/CD automated testing, and Kubernetes orchestration with Minikube.

---

## System Architecture

```text
               +-----------------------------------------------+
               |             Kubernetes Cluster                |
               |           (Namespace: ml-training)            |
               |                                               |
+------------+ |  +---------------------+                      |
| ConfigMap  |--->|  Training Job (Pod) |                      |
+------------+ |  +----------+----------+                      |
               |             | (Writes Checkpoint)             |
               |             v                                 |
               |  +---------------------+                      |
               |  |  ml-storage-pvc     |                      |
               |  |  (Persistent Volume)|                      |
               |  +----------+----------+                      |
               |             | (Reads Checkpoint)              |
               |             v                                 |
               |  +---------------------+                      |
               |  | Deployment (2 Pods) |<-------+             |
               |  | FastAPI Serving     |        | (Autoscale) |
               |  +----------+----------+        |             |
               |             ^             +-----+-----+       |
               |             |             |    HPA    |       |
               |             |             +-----------+       |
               |  +----------+----------+                      |
               |  | ClusterIP Service   |                      |
               |  | (Port 80 -> 8080)   |                      |
               |  +----------+----------+                      |
+------------+ |             ^                                 |
| Client /   | |             | (Port Forward)                  |
| cURL Test  |-+-------------+                                 |
+------------+ +-----------------------------------------------+

##Project Structure

mlops-pytorch-pipeline/
├── .github/workflows/
│   └── ci.yml                     # GitHub Actions CI workflow
├── configs/
│   └── training_config.yaml       # Hyperparameters & data config
├── docker/
│   ├── Dockerfile.train           # Multi-stage training image build
│   └── Dockerfile.serve           # Multi-stage FastAPI serving image build
├── k8s/
│   ├── namespace.yaml             # ml-training namespace definition
│   ├── configmap.yaml             # Mounts training configuration
│   ├── pvc.yaml                   # Persistent volume claim for storage
│   ├── training-job.yaml          # PyTorch batch training job
│   ├── serving-deployment.yaml    # Replicated serving pods with probes
│   ├── serving-service.yaml       # ClusterIP service
│   └── hpa.yaml                   # Horizontal pod autoscaler
├── requirements/
│   ├── train.txt                  # PyTorch, Torchvision, PyYAML
│   └── serve.txt                  # FastAPI, Uvicorn, Pillow
├── src/
│   ├── dataset.py                 # CIFAR-10 data loaders & transforms
│   ├── model.py                   # ResNet-18 model definition
│   ├── train.py                   # PyTorch training loop & early stopping
│   └── serve.py                   # FastAPI prediction service & health checks
├── tests/
│   └── test_model.py              # Unit tests for model forward pass
└── .gitignore
└── README.md
└── test_image.png


##Prerequisites
macOS / Linux environment
Docker Desktop (running)
Minikube & kubectl
Python 3.11 with pip

##Local Development & Docker Verification

###1. Build Multi-Stage Docker Images
```
bash
# Build training image
docker build -f docker/Dockerfile.train -t mlops-train:v1 .

# Build serving image
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
```

###2. Local Training Run
```
bash
docker run --rm \
  -v "$(pwd)/configs:/app/configs" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  mlops-train:v1
```

###3. Local Serving and Prediction test

```
bash
# Start container
docker run -d --name local-serve -p 8080:8080 \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  mlops-serve:v1

# Test health probe
curl http://localhost:8080/health

# Generate sample image and query inference endpoint
python3 -c "from PIL import Image; img = Image.new('RGB', (32, 32), color='blue'); img.save('test_image.png')"
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"

# Cleanup
docker stop local-serve && docker rm local-serve
```
