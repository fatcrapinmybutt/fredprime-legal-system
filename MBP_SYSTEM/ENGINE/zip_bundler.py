import zipfile
import os
import datetime


def bundle_zip(files):
    out_dir = "MBP_SYSTEM/VFS/GENERATED_ZIPS"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = os.path.join(out_dir, f"Motion_Bundle_{timestamp}.zip")

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in files:
            zipf.write(file, arcname=os.path.basename(file))
    print(f"ZIP Created: {zip_path}")
    return zip_path


if __name__ == "__main__":
    files = []
    bundle_zip(files)
