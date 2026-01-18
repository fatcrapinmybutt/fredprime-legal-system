import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


class LocalScanner:
    def __init__(self, root: Path):
        self.root = root

    def scan(self) -> Dict[str, str]:
        index = {}
        for path in self.root.rglob("*"):
            if path.is_file():
                h = hashlib.sha256(path.read_bytes()).hexdigest()
                index[str(path)] = h
        return index


class GoogleDriveSync:
    def __init__(self, credentials_file: Path, token_file: Path):
        try:
            from google.auth.transport.requests import Request  # type: ignore
            from google.oauth2.credentials import Credentials  # type: ignore
            from google_auth_oauthlib.flow import (  # type: ignore
                InstalledAppFlow,
            )
            from googleapiclient.discovery import build  # type: ignore
        except ImportError:
            raise RuntimeError(
                "google-api-python-client is not installed.\n"
                "Run `pip install google-api-python-client "
                "google-auth-httplib2 google-auth-oauthlib`"
            )

        self.creds = None
        if token_file.exists():
            self.creds = Credentials.from_authorized_user_file(token_file)
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    credentials_file,
                    SCOPES,
                )
                self.creds = flow.run_local_server(port=0)
            with open(token_file, "w") as f:
                f.write(self.creds.to_json())
        self.service = build("drive", "v3", credentials=self.creds)

    def upload_file(self, local_path: Path, drive_folder: Optional[str] = None) -> str:
        file_metadata = {"name": local_path.name}
        if drive_folder:
            file_metadata["parents"] = [drive_folder]
        with local_path.open("rb") as fh:
            file = self.service.files().create(body=file_metadata, media_body=fh, fields="id").execute()
        return file.get("id")

    def list_files(self, folder_id: Optional[str] = None) -> Dict[str, str]:
        query = f"'{folder_id}' in parents" if folder_id else None
        response = self.service.files().list(q=query).execute()
        return {item["name"]: item["id"] for item in response.get("files", [])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync local files with Google Drive")
    parser.add_argument("--scan", help="Directory to scan")
    parser.add_argument("--upload", help="File to upload")
    parser.add_argument(
        "--credentials",
        default="credentials.json",
        help="OAuth credentials JSON",
    )
    parser.add_argument(
        "--token",
        default="token.json",
        help="Token storage path",
    )
    args = parser.parse_args()

    if args.scan:
        scanner = LocalScanner(Path(args.scan))
        index = scanner.scan()
        print(json.dumps(index, indent=2))
        return

    if args.upload:
        try:
            gdrive = GoogleDriveSync(Path(args.credentials), Path(args.token))
            file_id = gdrive.upload_file(Path(args.upload))
        except RuntimeError as exc:
            print(f"Cannot upload: {exc}")
            return
        print(f"Uploaded {args.upload} to Google Drive with id {file_id}")


if __name__ == "__main__":
    main()
