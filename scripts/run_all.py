import shutil

from build_system import run as build_run
from codex_brain import main as brain_main
from codex_patch_manager import apply as patch_apply


def main() -> None:
    """Run the build, brain update, and patch application sequence."""
    build_run()
    brain_main()
    patch_apply()
    shutil.make_archive("output/archive", "zip", "output")


if __name__ == "__main__":
    main()
