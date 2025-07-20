from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os


def upload_to_drive(path: str):
    """Upload file to Google Drive using saved OAuth token."""
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile('token.json')
    if gauth.credentials is None:
        gauth.LocalWebserverAuth()
        gauth.SaveCredentialsFile('token.json')
    else:
        gauth.Authorize()

    drive = GoogleDrive(gauth)
    file = drive.CreateFile({'title': os.path.basename(path)})
    file.SetContentFile(path)
    file.Upload()
    print(f"Uploaded {path} to Google Drive as {file['title']}")
