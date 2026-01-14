#!/usr/bin/env bash
# Installer for Litigation OS Appliance (Air-gapped)
set -euo pipefail

# Variables
IMG_TARBALL="litigation-images.tar.gz"
HELM_CHART_TARBALL="litigation-charts.tar.gz"
CONFIG_BUNDLE="litigation-configs.tar.gz"
REGISTRY="localhost:5000"

echo "Loading Docker images into private registry..."
tar -xzf "$IMG_TARBALL" -C /tmp
for img in /tmp/*.tar; do
  echo "Importing $img" && ctr --namespace k8s.io images import "$img"
done

echo "Loading Helm charts..."
tar -xzf "$HELM_CHART_TARBALL" -C /tmp/charts

echo "Applying Kubernetes manifests via Helm..."
helm upgrade --install litigation-engine /tmp/charts/litigation-engine \
  --namespace litigation --create-namespace \
  --set image.registry=$REGISTRY \
  --values /app/configs/values.yaml

echo "Injector: post-install tasks"
kubectl apply -f /app/configs/post-install/

echo "Installation complete. Set license with 'litigation-cli license set <KEY>'"