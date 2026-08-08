"""Module 14 (Optional) — Google Drive: upload, search, store generated reports.

Falls back to a local `data/drive_storage/` folder when no Google credentials
are configured, so uploading/searching "reports" always works in a demo.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.tools import BaseTool, tool

import config


def _local_dir() -> Path:
    config.DRIVE_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    return config.DRIVE_STORAGE_DIR


@tool("upload_report_to_drive")
def _stub_upload(filename: str, content: str) -> str:
    """Save a generated report/document under the given filename. In stub mode
    this writes to a local folder instead of real Google Drive."""
    path = _local_dir() / filename
    path.write_text(content, encoding="utf-8")
    return f"[STUB MODE — no Google Drive credentials configured] Saved to {path}"


@tool("search_drive_reports")
def _stub_search(query: str = "") -> str:
    """Search previously stored reports by filename substring. In stub mode this
    searches the local drive_storage folder."""
    matches = [p.name for p in _local_dir().glob("*") if query.lower() in p.name.lower()]
    if not matches:
        return f"[STUB MODE] No stored reports match '{query}'."
    return "\n".join(matches)


def _stub_tools() -> List[BaseTool]:
    return [_stub_upload, _stub_search]


def _real_drive_tools() -> List[BaseTool]:
    """Minimal real Google Drive tools using google-api-python-client directly
    (LangChain has no first-party Drive toolkit as of this writing)."""
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaInMemoryUpload

    token_path = config.BASE_DIR / "drive_token.json"
    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), config.DRIVE_SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(config.GOOGLE_CREDENTIALS_PATH), config.DRIVE_SCOPES
            )
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json(), encoding="utf-8")

    service = build("drive", "v3", credentials=creds)

    @tool("upload_report_to_drive")
    def upload_report_to_drive(filename: str, content: str) -> str:
        """Upload a generated report/document to Google Drive."""
        metadata = {"name": filename}
        if config.GOOGLE_DRIVE_FOLDER_ID:
            metadata["parents"] = [config.GOOGLE_DRIVE_FOLDER_ID]
        media = MediaInMemoryUpload(content.encode("utf-8"), mimetype="text/plain")
        file = service.files().create(body=metadata, media_body=media, fields="id, webViewLink").execute()
        return f"Uploaded to Google Drive: {file.get('webViewLink', file.get('id'))}"

    @tool("search_drive_reports")
    def search_drive_reports(query: str = "") -> str:
        """Search previously stored reports in Google Drive by name."""
        q = f"name contains '{query}'" if query else None
        results = service.files().list(q=q, pageSize=10, fields="files(id, name, webViewLink)").execute()
        files = results.get("files", [])
        if not files:
            return f"No Drive files match '{query}'."
        return "\n".join(f"{f['name']} -> {f.get('webViewLink', f['id'])}" for f in files)

    return [upload_report_to_drive, search_drive_reports]


def get_drive_tools() -> List[BaseTool]:
    if config.DRIVE_CONFIGURED:
        try:
            return _real_drive_tools()
        except Exception:
            return _stub_tools()
    return _stub_tools()
