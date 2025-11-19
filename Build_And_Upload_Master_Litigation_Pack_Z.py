from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
import zipfile
import shutil


# === GOOGLE DRIVE SETUP ===
def setup_drive():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()  # Opens local web browser for OAuth2
    return GoogleDrive(gauth)


# === FILE PREPARATION ===
def prepare_files():
    files = {
        "Z:/PLAINTIFF_REBUTTAL_vOmegaPlus_FINAL_FIXED.docx": "MASTER_MOTION.docx",
        "Z:/Unified_Affirmative_Defense_and_Rebuttal_ShowCause5_6_4d8033f2.docx": "MASTER_AFFIDAVIT.docx",
        "Z:/STRIKEBACK_LINE_BY_LINE_REBUTTAL_vOMEGA.docx": "MASTER_SANCTIONS_REBUTTAL.docx",
        "Z:/STRIKEBACK_PARAGRAPH_6_TO_30_REBUTTAL.docx": "MASTER_PARAGRAPH_REBUTTAL.docx",
    }

    output_dir = r"Z:/F_LITIGATION_OS_STAGING"
    zip_path = r"Z:/MASTER_LITIGATION_PACK_F.zip"
    os.makedirs(output_dir, exist_ok=True)

    for src, renamed in files.items():
        if os.path.exists(src):
            dst = os.path.join(output_dir, renamed)
            shutil.copyfile(src, dst)
            print(f"[LLM-LOG] Prepared {dst}")
        else:
            print(f"[WARNING] Missing: {src}")

    # Compress files
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in os.listdir(output_dir):
            abs_path = os.path.join(output_dir, file)
            zf.write(abs_path, arcname=file)
    print(f"[✅] ZIP created at {zip_path}")

    return zip_path


# === GOOGLE DRIVE UPLOAD ===
def upload_to_drive(drive, file_path):
    file_name = os.path.basename(file_path)
    f = drive.CreateFile({"title": file_name})
    f.SetContentFile(file_path)
    f.Upload()
    print(f"[📤] Uploaded to Google Drive: {file_name}")


if __name__ == "__main__":
    drive = setup_drive()
    zip_file = prepare_files()
    upload_to_drive(drive, zip_file)
