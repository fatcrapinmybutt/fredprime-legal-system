"""
Example: Advanced Usage

This example demonstrates advanced features of DriveOrganizerPro.
"""

from pathlib import Path
from drive_organizer_pro.core.organizer_engine import OrganizerEngine
from drive_organizer_pro.core.duplicate_handler import DuplicateHandler
from drive_organizer_pro.core.backup_manager import BackupManager


def progress_callback(current: int, total: int, status: str):
    """Custom progress callback with percentage."""
    if total > 0:
        percentage = (current / total) * 100
        print(f"\r[{percentage:5.1f}%] {current}/{total} - {status}", end='', flush=True)


def organize_with_monitoring():
    """Organize files with detailed progress monitoring."""
    print("=" * 60)
    print("ADVANCED ORGANIZATION WITH MONITORING")
    print("=" * 60)

    engine = OrganizerEngine()
    source = Path("E:/Documents")  # Change this

    if not source.exists():
        print(f"✗ Source not found: {source}")
        return

    print(f"\n📂 Source: {source}")
    print(f"🎯 Mode: LIVE (files will be moved!)")
    print(f"⚙️  Workers: 8 threads")
    print(f"🔍 Duplicate detection: Enabled")
    print(f"📁 Sub-buckets: Enabled")
    print()

    stats = engine.organize_drive(
        source_path=source,
        dry_run=False,  # LIVE mode
        remove_empty=True,
        handle_duplicates=True,
        create_sub_buckets=True,
        progress_callback=progress_callback,
        max_workers=8
    )

    print("\n\n" + "=" * 60)
    print("ORGANIZATION COMPLETE")
    print("=" * 60)
    print(f"✓ Files processed: {stats['files_processed']}")
    print(f"✓ Files moved: {stats['files_moved']}")
    print(f"⚠ Files skipped: {stats['files_skipped']}")
    print(f"🔄 Duplicates found: {stats['duplicates_found']}")
    print(f"❌ Errors: {stats['errors']}")


def scan_for_duplicates():
    """Scan directory for duplicates without organizing."""
    print("\n" + "=" * 60)
    print("DUPLICATE SCAN")
    print("=" * 60)

    handler = DuplicateHandler(algorithm='md5')
    source = Path("E:/Documents")

    if not source.exists():
        print(f"✗ Source not found: {source}")
        return

    print(f"\n📂 Scanning: {source}")
    print(f"🔍 Algorithm: MD5\n")

    duplicates = handler.scan_directory(source, recursive=True)

    print("\n" + "=" * 60)
    print("DUPLICATE SCAN RESULTS")
    print("=" * 60)
    print(f"Found {len(duplicates)} duplicate files\n")

    if duplicates:
        print("Duplicates:")
        for i, (duplicate, original) in enumerate(duplicates[:10], 1):
            print(f"\n{i}. {duplicate.name}")
            print(f"   Original: {original}")

        if len(duplicates) > 10:
            print(f"\n... and {len(duplicates) - 10} more")


def manage_backup_sessions():
    """List and manage backup sessions."""
    print("\n" + "=" * 60)
    print("BACKUP SESSION MANAGEMENT")
    print("=" * 60)

    manager = BackupManager()
    sessions = manager.list_sessions()

    print(f"\nFound {len(sessions)} backup sessions:\n")

    for session in sessions:
        print(f"📋 {session['session_id']}")
        print(f"   Created: {session['created']}")
        print(f"   Files: {session['total_moves']}")
        print()

    if sessions:
        # Revert most recent session
        latest = sessions[0]['session_id']
        print(f"To revert latest session, run:")
        print(f"  manager.revert_session('{latest}')")


def batch_organize_drives():
    """Organize multiple drives in sequence."""
    print("\n" + "=" * 60)
    print("BATCH DRIVE ORGANIZATION")
    print("=" * 60)

    engine = OrganizerEngine()
    drives = [Path("E:/"), Path("F:/"), Path("H:/")]

    for drive in drives:
        if not drive.exists():
            print(f"⚠ Skipping non-existent drive: {drive}")
            continue

        print(f"\n📂 Processing: {drive}")

        stats = engine.organize_drive(
            source_path=drive,
            dry_run=True,  # Safety first!
            progress_callback=lambda c, t, s: print(f"\r  [{c}/{t}]", end='', flush=True)
        )

        print(f"\n  ✓ Complete: {stats['files_moved']} files moved")


if __name__ == "__main__":
    # Uncomment to run different examples:

    # organize_with_monitoring()
    # scan_for_duplicates()
    # manage_backup_sessions()
    # batch_organize_drives()

    print("\nUncomment function calls in __main__ to run examples")
