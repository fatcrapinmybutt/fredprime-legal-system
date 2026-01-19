# Frequently Asked Questions (FAQ)

## General Questions

### What is DriveOrganizerPro?

DriveOrganizerPro is a professional file organization system that automatically categorizes files into intelligent buckets based on file type, with support for duplicate detection, sub-bucket organization, and full revert capability.

### Who is MBP LLC?

MBP LLC (Maximum Business Performance, LLC) is the company behind DriveOrganizerPro. "Powered by Pork™" is our commitment to quality and performance.

### Is it free?

Yes! DriveOrganizerPro is open source and licensed under MIT License.

### What operating systems are supported?

- Windows 10/11
- macOS 10.14+
- Linux (most distributions)

## Usage Questions

### Should I use DRY RUN first?

**YES!** Always run in DRY RUN mode first to preview what changes will be made. This lets you verify the organization before actually moving files.

### Will it delete my files?

No! DriveOrganizerPro only moves files, never deletes them. You can always revert changes using the built-in backup system.

### What if I don't like the results?

Use the "REVERT CHANGES" button to undo the last organization. Files will be moved back to their original locations.

### How long does it take?

Processing time depends on:
- Number of files (typically 100-500 files/second)
- Duplicate detection (adds hashing time)
- Drive speed (SSD vs HDD)
- System performance

Example: 10,000 files usually takes 1-5 minutes.

### Can I organize multiple drives at once?

The GUI shows drive selection, but processes selected directories. For multiple independent drives, run organization separately for each.

## Technical Questions

### Does it require internet?

No! DriveOrganizerPro works completely offline with no internet connection required.

### Does it use AI?

No, it uses deterministic rule-based organization. File classification is based on:
- File extensions
- Keyword detection
- Hash comparison for duplicates

### How does duplicate detection work?

Files are hashed using MD5 or SHA256 algorithms. Files with matching hashes are considered duplicates. The oldest file is kept as the original.

### Can I customize the buckets?

Yes! You can create custom bucket configurations by modifying the JSON config files or creating your own.

### What happens to files with unknown extensions?

They go to the "Miscellaneous" bucket.

## Safety & Security Questions

### Is my data safe?

Yes! DriveOrganizerPro:
- Never deletes files
- Creates backup logs for every operation
- Supports full revert
- Works entirely on your local system

### Can I trust it with important files?

- Always backup important data first
- Use DRY RUN mode to preview
- Start with test directories
- The backup/revert system provides safety net

### Does it modify file contents?

No! Only file locations are changed. File contents remain exactly the same.

### Does it collect any data?

No! DriveOrganizerPro doesn't collect, transmit, or store any data outside your local system.

## Feature Questions

### Can it handle nested folders?

Yes! The de-nesting algorithm flattens all subdirectories and organizes all files into buckets.

### What about empty folders?

Enable "Remove empty directories" option to automatically clean up empty folders after organization.

### Can I keep my folder structure?

DriveOrganizerPro is designed to flatten and reorganize. If you want to preserve structure, it's not the right tool for that use case.

### Does it work with network drives?

Yes, but performance may be slower depending on network speed.

### Can I pause and resume?

The process runs continuously once started, but operations are stateful - you can run multiple times to process additional files.

## Error & Problem Questions

### What if it gets stuck?

- Check log output for errors
- Wait - large operations take time
- For GUI freeze, operation still runs in background
- Restart application if truly frozen

### Files weren't organized?

- Check if DRY RUN is enabled
- Verify source path is correct
- Check log for errors
- Ensure files have recognized extensions

### Got permission errors?

- Run as administrator (Windows)
- Check file permissions
- Close programs using the files

### Can't revert?

Revert requires:
- Valid backup session
- Files still at organized location
- Backup JSON file intact

## Performance Questions

### How can I make it faster?

- Disable duplicate detection (most CPU-intensive)
- Reduce worker threads on slower systems
- Use SSD instead of HDD
- Process smaller batches

### It's using too much memory?

- Process directories in smaller batches
- Close other applications
- Reduce worker threads

## Customization Questions

### Can I add custom file types?

Yes! Edit the bucket configuration JSON files to add new extensions.

### Can I change bucket names?

Yes, but you'll need to edit the configuration files.

### Can I add more sub-buckets?

Yes! Edit `sub_buckets.json` to add your own sub-buckets and keyword mappings.

## Support Questions

### Where do I get help?

1. Check this FAQ
2. Read the [Troubleshooting Guide](TROUBLESHOOTING.md)
3. Review [User Guide](USER_GUIDE.md)
4. Open an issue on GitHub

### How do I report bugs?

Open an issue on [GitHub Issues](https://github.com/fatcrapinmybutt/fredprime-legal-system/issues) with:
- Error message
- Log output
- System info
- Steps to reproduce

### Can I contribute?

Yes! See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## About the Project

### Why the pig logo?

It represents our company mascot and commitment to maximum business performance. Plus, pigs are intelligent animals - perfect for a smart organization system!

### What does "Powered by Pork™" mean?

It's our company tagline representing quality, performance, and a bit of fun personality.

### Is there a roadmap?

Feature requests and improvements are tracked in GitHub Issues. The core functionality is complete and production-ready.

---

**Still have questions?** Open an issue on GitHub or check the documentation!

© 2026 MBP LLC. All rights reserved. Powered by Pork™
