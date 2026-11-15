"""La ingesta recupera los mismos bytes o se detiene antes de procesar."""

import hashlib
import json
import subprocess

import pytest

from mibici import dataset

CSV = b"id,name,latitude,longitude,dpcapacity\n1,Centro,20.6,-103.3,15\n"
PRIMARY = "https://example.org/catalogo.csv"
ARCHIVE = "https://archive.example.org/catalogo.csv"


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    name = dataset.NOMENCLATURA_FILENAME
    expected = {
        "filename": name,
        "sha256": hashlib.sha256(CSV).hexdigest(),
        "size_bytes": len(CSV),
        "records": 1,
        "source_url": PRIMARY,
    }
    (manifests / dataset.DOWNLOADS_MANIFEST).write_text(json.dumps([expected]))
    monkeypatch.setattr(dataset, "RAW_DATA_DIR", raw)
    monkeypatch.setattr(dataset, "MANIFESTS_DIR", manifests)
    monkeypatch.setattr(dataset, "DOWNLOAD_FALLBACKS", {name: [ARCHIVE]})
    monkeypatch.setattr(dataset, "expected_files", lambda: [name])
    return raw, manifests, expected


def test_primary_success_does_not_contact_archive(frozen, monkeypatch):
    raw, _, expected = frozen
    calls = []

    def fetch(url, output):
        calls.append(url)
        output.write_bytes(CSV)

    monkeypatch.setattr(dataset, "run_curl", fetch)
    path, source = dataset.download(expected["filename"], PRIMARY, expected)
    assert calls == [PRIMARY]
    assert source == PRIMARY
    assert path.read_bytes() == CSV
    assert not list(raw.glob("*.part"))


@pytest.mark.parametrize("failure", ["404", "html", "changed_csv"])
def test_invalid_primary_uses_verified_archive(frozen, monkeypatch, failure):
    raw, _, expected = frozen
    calls = []

    def fetch(url, output):
        calls.append(url)
        if url == PRIMARY:
            if failure == "404":
                output.write_bytes(b"partial response")
                raise subprocess.CalledProcessError(22, ["curl", url])
            output.write_bytes(
                b"<!DOCTYPE html><title>Not found</title>"
                if failure == "html"
                else CSV.replace(b"Centro", b"Cambio")
            )
        else:
            output.write_bytes(CSV)

    monkeypatch.setattr(dataset, "run_curl", fetch)
    path, source = dataset.download(expected["filename"], PRIMARY, expected)
    assert calls == [PRIMARY, ARCHIVE]
    assert source == ARCHIVE
    assert path.read_bytes() == CSV
    assert not list(raw.glob("*.part"))


def test_both_sources_invalid_leave_no_final_file(frozen, monkeypatch):
    raw, _, expected = frozen
    monkeypatch.setattr(dataset, "run_curl", lambda url, output: output.write_bytes(b"error"))
    with pytest.raises(RuntimeError, match="hash esperado"):
        dataset.download(expected["filename"], PRIMARY, expected)
    assert list(raw.iterdir()) == []


def test_main_works_without_portal_and_preserves_provenance(frozen, monkeypatch):
    _, manifests, expected = frozen
    calls = []
    processed = []

    def fetch(url, output):
        calls.append(url)
        if url == PRIMARY:
            raise subprocess.CalledProcessError(22, ["curl", url])
        output.write_bytes(CSV)

    def no_portal():
        pytest.fail("Con URLs congeladas no debe consultar el portal")

    monkeypatch.setattr(dataset, "run_curl", fetch)
    monkeypatch.setattr(dataset, "discover_files", no_portal)
    monkeypatch.setattr(dataset, "generate_monthly_report", lambda manifest: None)
    monkeypatch.setattr(dataset, "procesar", processed.append)
    dataset.main()
    first = json.loads((manifests / dataset.DOWNLOADS_MANIFEST).read_text())
    assert first[0]["source_url"] == PRIMARY
    assert first[0]["downloaded_from"] == ARCHIVE
    assert first[0]["sha256"] == expected["sha256"]
    dataset.main()
    assert calls == [PRIMARY, ARCHIVE]  # La segunda ejecución valida la copia local.
    assert json.loads((manifests / dataset.DOWNLOADS_MANIFEST).read_text()) == first
    assert len(processed) == 2


def test_altered_cached_csv_does_not_rewrite_manifest_or_process(frozen, monkeypatch):
    raw, manifests, expected = frozen
    (raw / expected["filename"]).write_bytes(CSV.replace(b"Centro", b"Cambio"))
    path = manifests / dataset.DOWNLOADS_MANIFEST
    before = path.read_bytes()
    monkeypatch.setattr(dataset, "procesar", lambda _: pytest.fail("No debe procesar"))
    with pytest.raises(ValueError, match="SHA-256"):
        dataset.main()
    assert path.read_bytes() == before


def test_missing_hash_stops_before_download(frozen, monkeypatch):
    _, manifests, expected = frozen
    del expected["sha256"]
    (manifests / dataset.DOWNLOADS_MANIFEST).write_text(json.dumps([expected]))
    monkeypatch.setattr(dataset, "run_curl", lambda *args: pytest.fail("No debe descargar"))
    with pytest.raises(ValueError, match="Faltan hashes"):
        dataset.main()
