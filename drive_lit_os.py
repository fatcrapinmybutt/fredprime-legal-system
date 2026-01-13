"""Google Drive helper utilities for the litigation OS pipeline."""

from __future__ import annotations

import io
import json
import random
import time
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence, cast

if TYPE_CHECKING:  # pragma: no cover - typing only
    from googleapiclient.errors import HttpError  # type: ignore
    from googleapiclient.http import MediaIoBaseUpload  # type: ignore
else:  # pragma: no cover - runtime fallback

    try:
        from googleapiclient.errors import HttpError  # type: ignore
    except ImportError:

        class HttpError(Exception):
            """Fallback ``HttpError`` used for tests when googleapiclient is unavailable."""

            def __init__(
                self, resp: Any | None = None, content: bytes | None = None
            ) -> None:
                super().__init__("HttpError")
                self.resp = resp
                self.content = content

    try:
        from googleapiclient.http import MediaIoBaseUpload  # type: ignore
    except ImportError:

        class MediaIoBaseUpload:  # noqa: D401 - minimal stand in
            """Minimal stand-in for :class:`MediaIoBaseUpload` used in tests."""

            def __init__(
                self, fh: io.IOBase, mimetype: str, resumable: bool = False
            ) -> None:
                self.fh = fh
                self.mimetype = mimetype
                self.resumable = resumable


RetryFunction = Callable[..., Any]
RETRYABLE_HTTP_STATUS = {500, 502, 503, 504}
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 1.0
BACKOFF_MULTIPLIER = 2.0
MAX_BACKOFF_SECONDS = 16.0


def _sleep(backoff: float) -> None:
    """Wrapper around :func:`time.sleep` for easier testing."""

    time.sleep(backoff)


def _next_backoff(attempt: int) -> float:
    backoff = min(
        INITIAL_BACKOFF_SECONDS * (BACKOFF_MULTIPLIER**attempt), MAX_BACKOFF_SECONDS
    )
    # Add a little jitter so we do not stampede retries.
    return backoff + random.uniform(0, 0.5)


def _join_parents(parents: Iterable[str]) -> str:
    return ",".join(list(parents))


def gcall(fn: RetryFunction, *args: Any, **kwargs: Any) -> Any:
    """Execute a Google API call with retries."""

    last_error: HttpError | None = None
    for attempt in range(MAX_RETRIES):
        try:
            request = fn(*args, **kwargs)
            return request.execute()
        except HttpError as exc:  # pragma: no cover - error path tested separately
            status = getattr(getattr(exc, "resp", None), "status", None)
            if status not in RETRYABLE_HTTP_STATUS or attempt == MAX_RETRIES - 1:
                raise
            last_error = exc
            _sleep(_next_backoff(attempt))
    if last_error is not None:  # pragma: no cover - defensive guard
        raise last_error
    raise RuntimeError("gcall failed without executing a request")  # pragma: no cover


def list_files(service: Any, **kwargs: Any) -> Mapping[str, Any]:
    """Return the JSON payload for ``files().list``."""

    return cast(Mapping[str, Any], gcall(service.files().list, **kwargs))


def create_folder_if_absent(
    service: Any, name: str, parent_id: str | None = None
) -> Mapping[str, Any]:
    """Ensure a folder with ``name`` exists under ``parent_id``."""

    query = [
        "name = '{}'".format(name.replace("'", "\\'")),
        "mimeType = 'application/vnd.google-apps.folder'",
        "trashed = false",
    ]
    if parent_id:
        query.append("'{}' in parents".format(parent_id))
    response = list_files(
        service,
        q=" and ".join(query),
        spaces="drive",
        fields="files(id, name, parents)",
    )
    files = cast(Sequence[Mapping[str, Any]], response.get("files", []))
    if files:
        return files[0]

    metadata: dict[str, Any] = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
    }
    if parent_id:
        metadata["parents"] = [parent_id]
    return cast(
        Mapping[str, Any],
        gcall(service.files().create, body=metadata, fields="id, name, parents"),
    )


def _build_media(data: bytes, mimetype: str) -> MediaIoBaseUpload:
    buffer = io.BytesIO(data)
    buffer.seek(0)
    return MediaIoBaseUpload(buffer, mimetype=mimetype, resumable=False)


def upload_json(
    service: Any, parent_id: str, name: str, payload: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Upload ``payload`` as a JSON file to Drive."""

    body: dict[str, Any] = {"name": name, "mimeType": "application/json"}
    if parent_id:
        body["parents"] = [parent_id]
    media = _build_media(
        json.dumps(payload, separators=(",", ":")).encode("utf-8"), "application/json"
    )
    return cast(
        Mapping[str, Any],
        gcall(
            service.files().create,
            body=body,
            media_body=media,
            fields="id, name, parents",
        ),
    )


def upload_text(
    service: Any, parent_id: str, name: str, text: str, mimetype: str = "text/plain"
) -> Mapping[str, Any]:
    """Upload ``text`` content as a simple Drive file."""

    body: dict[str, Any] = {"name": name, "mimeType": mimetype}
    if parent_id:
        body["parents"] = [parent_id]
    media = _build_media(text.encode("utf-8"), mimetype)
    return cast(
        Mapping[str, Any],
        gcall(
            service.files().create,
            body=body,
            media_body=media,
            fields="id, name, parents",
        ),
    )


def move_file(
    service: Any,
    file_id: str,
    new_parents: Iterable[str],
    remove_parents: Iterable[str] | None = None,
) -> Mapping[str, Any]:
    """Move ``file_id`` to ``new_parents`` removing ``remove_parents`` if provided."""

    kwargs: dict[str, Any] = {
        "fileId": file_id,
        "addParents": _join_parents(new_parents),
        "fields": "id, name, parents",
    }
    if remove_parents:
        kwargs["removeParents"] = _join_parents(remove_parents)
    return cast(Mapping[str, Any], gcall(service.files().update, **kwargs))


def delete_file(service: Any, file_id: str) -> Any:
    """Delete a file from Drive."""

    return gcall(service.files().delete, fileId=file_id)
