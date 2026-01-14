"""Utility for loading Litigation OS appliance assets from tarballs."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Iterable, Sequence


LOGGER = logging.getLogger("tarball_loader")


class TarballError(RuntimeError):
    """Raised when a tarball operation fails."""


def configure_logging(verbose: bool = False) -> None:
    """Configure a simple console logger."""
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.DEBUG if verbose else logging.INFO)


def extract_tarball(tarball: Path, destination: Path) -> None:
    """Extract a tarball to a destination directory."""
    if not tarball.exists():
        raise TarballError(f"Tarball not found: {tarball}")

    destination.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Extracting %s to %s", tarball, destination)
    try:
        with tarfile.open(tarball) as archive:
            archive.extractall(destination)
    except (tarfile.TarError, OSError) as exc:
        raise TarballError(f"Failed to extract {tarball}: {exc}") from exc


def run_command(command: Sequence[str]) -> None:
    """Execute an external command with logging."""
    LOGGER.info("Running command: %s", " ".join(command))
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        LOGGER.debug("stdout:\n%s", result.stdout.strip())
    if result.stderr:
        LOGGER.debug("stderr:\n%s", result.stderr.strip())

    if result.returncode != 0:
        raise TarballError(f"Command '{' '.join(command)}' failed with exit code {result.returncode}")


def import_images(image_dir: Path, ctr_namespace: str) -> None:
    """Import OCI images found in a directory using containerd's ctr tool."""
    importer = shutil.which("ctr")
    if importer is None:
        raise TarballError("The 'ctr' command is required but not found in PATH")

    image_files = sorted(image_dir.glob("*.tar"))
    if not image_files:
        LOGGER.warning("No image archives found in %s", image_dir)
        return

    for image in image_files:
        run_command([importer, "--namespace", ctr_namespace, "images", "import", str(image)])


def load_helm_chart(chart_dir: Path, release: str, namespace: str, registry: str, values: Path) -> None:
    """Install or upgrade a Helm chart extracted from a tarball."""
    helm = shutil.which("helm")
    if helm is None:
        raise TarballError("The 'helm' command is required but not found in PATH")

    if not chart_dir.exists():
        raise TarballError(f"Chart directory {chart_dir} does not exist")

    chart_path = chart_dir / release
    if not chart_path.exists():
        available = ", ".join(sorted(item.name for item in chart_dir.iterdir()))
        raise TarballError(f"Unable to locate chart directory '{chart_path}'. Available entries: {available or 'none'}")

    run_command(
        [
            helm,
            "upgrade",
            "--install",
            release,
            str(chart_path),
            "--namespace",
            namespace,
            "--create-namespace",
            "--set",
            f"image.registry={registry}",
            "--values",
            str(values),
        ]
    )


def apply_kubectl(manifest_dir: Path) -> None:
    """Apply Kubernetes manifests from a directory."""
    kubectl = shutil.which("kubectl")
    if kubectl is None:
        raise TarballError("The 'kubectl' command is required but not found in PATH")

    run_command([kubectl, "apply", "-f", str(manifest_dir)])


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Deploy Litigation OS assets from tarballs")
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("litigation-images.tar.gz"),
        help="OCI image tarball",
    )
    parser.add_argument(
        "--charts",
        type=Path,
        default=Path("litigation-charts.tar.gz"),
        help="Helm chart tarball",
    )
    parser.add_argument(
        "--configs",
        type=Path,
        default=Path("litigation-configs.tar.gz"),
        help="Configuration bundle tarball",
    )
    parser.add_argument(
        "--registry",
        default="localhost:5000",
        help="Container registry host used by the Helm chart",
    )
    parser.add_argument(
        "--values",
        type=Path,
        default=Path("/app/configs/values.yaml"),
        help="Path to the Helm values file",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("/app/configs/post-install"),
        help="Directory containing kubectl manifests",
    )
    parser.add_argument(
        "--ctr-namespace",
        default="k8s.io",
        help="Containerd namespace for image imports",
    )
    parser.add_argument(
        "--chart-name",
        default="litigation-engine",
        help="Helm release and directory name",
    )
    parser.add_argument(
        "--namespace",
        default="litigation",
        help="Kubernetes namespace for the Helm release",
    )
    parser.add_argument(
        "--chart-extract-dir",
        type=Path,
        default=Path("/tmp/charts"),
        help="Directory where Helm charts are extracted",
    )
    parser.add_argument(
        "--image-extract-dir",
        type=Path,
        default=Path("/tmp/images"),
        help="Directory where OCI images are extracted",
    )
    parser.add_argument(
        "--config-extract-dir",
        type=Path,
        default=Path("/app/configs"),
        help="Destination directory for configuration bundle",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging output")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """Execute the tarball deployment workflow."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    configure_logging(verbose=args.verbose)

    try:
        extract_tarball(args.images, args.image_extract_dir)
        import_images(args.image_extract_dir, args.ctr_namespace)

        extract_tarball(args.charts, args.chart_extract_dir)
        load_helm_chart(
            args.chart_extract_dir,
            args.chart_name,
            args.namespace,
            args.registry,
            args.values,
        )

        extract_tarball(args.configs, args.config_extract_dir)
        apply_kubectl(args.manifest_dir)
    except TarballError as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info("Installation complete. Set license with 'litigation-cli license set <KEY>'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
