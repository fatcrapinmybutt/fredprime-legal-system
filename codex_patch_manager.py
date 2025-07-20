import importlib.util
import os
from types import ModuleType

PATCH_DIR = "patches"
TARGET_EXE = "MBP_LITIGATION_OS_FINAL.exe"


def _apply_patch(module: ModuleType) -> None:
    if hasattr(module, "apply_patch"):
        module.apply_patch(TARGET_EXE)


def apply_patches() -> None:
    if not os.path.isdir(PATCH_DIR):
        print("\u26a0\ufe0f No patches found.")
        return
    for patch_file in os.listdir(PATCH_DIR):
        if patch_file.endswith(".py"):
            print(f"\ud83d\udd28 Applying patch: {patch_file}")
            path = os.path.join(PATCH_DIR, patch_file)
            spec = importlib.util.spec_from_file_location("patch_module", path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                _apply_patch(module)
    print("\u2705 All patches applied successfully.")


if __name__ == "__main__":
    apply_patches()
