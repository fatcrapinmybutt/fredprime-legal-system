"""
Example: Custom Configuration Usage

This example shows how to load and use custom bucket configurations.
"""

from pathlib import Path
from drive_organizer_pro.config.config_manager import ConfigManager
from drive_organizer_pro.core.organizer_engine import OrganizerEngine


def main():
    """Load custom configuration and organize files."""

    # Create config manager
    config = ConfigManager()

    # Load a preset configuration
    preset_path = Path(__file__).parent.parent / "config" / "presets" / "legal_preset.json"

    if preset_path.exists():
        config.load_buckets(preset_path)
        print(f"✓ Loaded preset: {preset_path}")
        print(f"  Buckets: {list(config.buckets.keys())}")
    else:
        print("✗ Preset not found, using defaults")

    # Create engine with custom config
    engine = OrganizerEngine(config_dir=config.config_dir)
    engine.config_manager = config

    # Organize with custom configuration
    source = Path("C:/MyFiles")  # Change this to your path

    if source.exists():
        print(f"\nOrganizing: {source}")
        print("Mode: DRY RUN (preview only)\n")

        stats = engine.organize_drive(
            source_path=source,
            dry_run=True,  # Safe preview mode
            create_sub_buckets=True
        )

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Files processed: {stats['files_processed']}")
        print(f"Files moved: {stats['files_moved']}")
        print(f"Errors: {stats['errors']}")
        print("\nBucket distribution:")
        for bucket, count in sorted(stats['buckets_used'].items()):
            print(f"  {bucket}: {count} files")
    else:
        print(f"✗ Source path not found: {source}")


if __name__ == "__main__":
    main()
