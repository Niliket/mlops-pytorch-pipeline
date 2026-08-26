# MLOps PyTorch Pipeline

End-to-end ML deployment pipeline featuring containerized PyTorch training and Kubernetes orchestration.

## Architecture
- **Training**: PyTorch CNN trained on CIFAR-10, packaged via multi-stage Dockerfile, executed as a Kubernetes Job.
- **Serving**: FastAPI/Flask model service with health probes deployed as a replicated Kubernetes Deployment behind a ClusterIP Service.
- **Configuration**: Managed via Kubernetes ConfigMaps and Persistent Volume Claims.
