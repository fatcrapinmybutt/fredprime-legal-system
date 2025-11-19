"""Minimal helper for uploading files to Google Drive."""

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os


def upload_to_drive(path: str) -> None:
    """Upload ``path`` to Google Drive if credentials are available."""

    gauth = GoogleAuth()
    gauth.LoadCredentialsFile("token.json")
    if gauth.credentials is None:
        try:
            gauth.LocalWebserverAuth()
            gauth.SaveCredentialsFile("token.json")
        except Exception as exc:
            print(f"Google Drive auth failed: {exc}")
            return
    else:
        gauth.Authorize()

    drive = GoogleDrive(gauth)
    file = drive.CreateFile({"title": os.path.basename(path)})
    file.SetContentFile(path)
    file.Upload()
    print(f'Uploaded {path} to Google Drive as {file["title"]}')
