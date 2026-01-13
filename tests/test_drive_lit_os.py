import types
from unittest import mock

import pytest

from drive_lit_os import (
    HttpError,
    delete_file,
    gcall,
    list_files,
    move_file,
    upload_json,
    upload_text,
)


class _Resp:
    def __init__(self, status: int | None) -> None:
        self.status = status
        self.reason = str(status) if status is not None else ""


class _Request:
    def __init__(self, responses):
        self._responses = list(responses)

    def execute(self):
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _fake_gcall(fn, *a, **kw):
    request = fn(*a, **kw)
    assert isinstance(request, _Request)
    return request.execute()


def _make_service():
    files_resource = mock.MagicMock()
    service = types.SimpleNamespace(files=mock.MagicMock(return_value=files_resource))
    files_resource.list = mock.MagicMock()
    files_resource.create = mock.MagicMock()
    files_resource.update = mock.MagicMock()
    files_resource.delete = mock.MagicMock()
    return service


def _http_error(status: int):
    return HttpError(resp=_Resp(status), content=b"")


def test_gcall_executes_and_returns_response():
    request = mock.Mock()
    request.execute.return_value = {"ok": True}
    fn = mock.Mock(return_value=request)
    assert gcall(fn) == {"ok": True}
    request.execute.assert_called_once_with()


def test_gcall_retries_and_eventually_succeeds(monkeypatch):
    request = mock.Mock()
    request.execute.side_effect = [_http_error(503), {"ok": True}]
    fn = mock.Mock(return_value=request)
    sleep_calls = []
    monkeypatch.setattr(
        "drive_lit_os._sleep", lambda duration: sleep_calls.append(duration)
    )

    assert gcall(fn) == {"ok": True}
    assert len(sleep_calls) == 1
    assert request.execute.call_count == 2


def test_gcall_raises_for_non_retryable_error(monkeypatch):
    request = mock.Mock()
    request.execute.side_effect = _http_error(404)
    fn = mock.Mock(return_value=request)
    monkeypatch.setattr("drive_lit_os._sleep", mock.Mock())

    with pytest.raises(HttpError):
        gcall(fn)


def test_list_files_returns_payload():
    service = _make_service()
    service.files.return_value.list.return_value = _Request([{"files": []}])
    assert list_files(service, q="foo") == {"files": []}


def test_upload_json_uses_gcall(monkeypatch):
    service = _make_service()
    request = _Request([{"id": "123"}])
    service.files.return_value.create.return_value = request
    monkeypatch.setattr("drive_lit_os.gcall", _fake_gcall)

    response = upload_json(service, "parent", "test.json", {"a": 1})
    assert response == {"id": "123"}


def test_upload_text_uses_gcall(monkeypatch):
    service = _make_service()
    request = _Request([{"id": "t"}])
    service.files.return_value.create.return_value = request
    monkeypatch.setattr("drive_lit_os.gcall", _fake_gcall)

    response = upload_text(service, "parent", "note.txt", "hello")
    assert response == {"id": "t"}


def test_move_and_delete_calls_gcall(monkeypatch):
    service = _make_service()
    service.files.return_value.update.return_value = _Request([{"id": "f"}])
    service.files.return_value.delete.return_value = _Request([None])
    monkeypatch.setattr("drive_lit_os.gcall", _fake_gcall)

    assert move_file(service, "id", ["parent"], None) == {"id": "f"}
    assert delete_file(service, "id") is None
