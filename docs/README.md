# Documentation

Project documentation lives here.

## Run All

Execute the complete build and patch sequence:

```bash
python -m scripts.run_all
```

The command runs `build_system.run()`, `codex_brain.main()`, and `codex_patch_manager.apply()` before creating the `output/archive.zip` file.
