"""Installer for Litigation OS Appliance (Air-gapped).

This script loads container images and Helm charts from tarball archives
into an air-gapped environment and applies the Kubernetes manifests needed
to run the litigation engine.
"""

from __future__ import annotations

import glob
import subprocess
import tarfile
from pathlib import Path


IMG_TARBALL = "litigation-images.tar.gz"
HELM_CHART_TARBALL = "litigation-charts.tar.gz"
CONFIG_BUNDLE = "litigation-configs.tar.gz"
REGISTRY = "localhost:5000"


def load_docker_images(img_tarball: str = IMG_TARBALL) -> None:
    """Extract image tarballs and import them into the container runtime."""

    print("Loading Docker images into private registry...")
    with tarfile.open(img_tarball, "r:gz") as tar:
        tar.extractall("/tmp")
    for img in glob.glob("/tmp/*.tar"):
        print(f"Importing {img}")
        subprocess.run(
            ["ctr", "--namespace", "k8s.io", "images", "import", img],
            check=True,
        )


def load_helm_charts(chart_tarball: str = HELM_CHART_TARBALL) -> Path:
    """Extract Helm charts to a temporary directory and return its path."""

    print("Loading Helm charts...")
    target_dir = Path("/tmp/charts")
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(chart_tarball, "r:gz") as tar:
        tar.extractall(target_dir)
    return target_dir


def apply_kubernetes_manifests(chart_dir: Path) -> None:
    """Install Kubernetes manifests via Helm and apply post-install tasks."""

    print("Applying Kubernetes manifests via Helm...")
    subprocess.run(
        [
            "helm",
            "upgrade",
            "--install",
            "litigation-engine",
            str(chart_dir / "litigation-engine"),
            "--namespace",
            "litigation",
            "--create-namespace",
            "--set",
            f"image.registry={REGISTRY}",
            "--values",
            "/app/configs/values.yaml",
        ],
        check=True,
    )
    print("Injector: post-install tasks")
    subprocess.run(
        ["kubectl", "apply", "-f", "/app/configs/post-install/"],
        check=True,
    )
    print("Installation complete. Set license with 'litigation-cli license set <KEY>'")


def main() -> None:
    load_docker_images()
    chart_dir = load_helm_charts()
    apply_kubernetes_manifests(chart_dir)


if __name__ == "__main__":
    main()
