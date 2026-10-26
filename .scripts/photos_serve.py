#!/usr/bin/env python3
"""
Local drag-and-drop photo uploader (127.0.0.1 only — no Worker, keys stay local).

  python3 .scripts/photos_serve.py

Opens http://127.0.0.1:8787 — drop images, EXIF auto-fills date/geo/camera/place,
upload → R2 + _photos/*.md
"""

from __future__ import annotations

import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from photos_lib import (
        PhotosError,
        add_photo_bytes,
        inspect_photo_bytes,
        load_env,
        parse_date,
        title_from_filename,
    )
except ImportError:
    print(
        "Missing deps. Run: pip install -r .scripts/requirements-photos.txt",
        file=sys.stderr,
    )
    sys.exit(1)

HOST = "127.0.0.1"
PORT = 8787

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Photo upload — zlu.me</title>
  <style>
    :root {
      --bg: #f7f6f2;
      --ink: #1c1b19;
      --muted: #6b6560;
      --line: #d4d0c8;
      --accent: #7c0000;
      --ok: #1f6b3a;
      --err: #9b1c1c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font: 15px/1.45 "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
      color: var(--ink);
      background:
        radial-gradient(ellipse at 20% 0%, #efe8dc 0%, transparent 55%),
        radial-gradient(ellipse at 100% 80%, #e4ebe8 0%, transparent 45%),
        var(--bg);
      min-height: 100vh;
    }
    main {
      max-width: 780px;
      margin: 0 auto;
      padding: 2.5rem 1.25rem 4rem;
    }
    h1 {
      font-size: 1.75rem;
      font-weight: 600;
      letter-spacing: -0.02em;
      margin: 0 0 0.25rem;
    }
    .sub {
      color: var(--muted);
      margin: 0 0 1.75rem;
      font-size: 0.95rem;
    }
    label.field {
      display: block;
      margin-bottom: 0.85rem;
    }
    label.field span {
      display: block;
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--muted);
      margin-bottom: 0.25rem;
    }
    input[type=text], input[type=date] {
      width: 100%;
      padding: 0.55rem 0.65rem;
      border: 1px solid var(--line);
      border-radius: 4px;
      background: #fff;
      font: inherit;
      color: var(--ink);
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.85rem;
    }
    @media (max-width: 560px) {
      .row { grid-template-columns: 1fr; }
    }
    #drop {
      margin: 1.25rem 0 1rem;
      padding: 2.5rem 1.25rem;
      border: 1.5px dashed var(--line);
      border-radius: 8px;
      background: rgba(255,255,255,0.55);
      text-align: center;
      color: var(--muted);
      cursor: pointer;
      transition: border-color 0.15s, background 0.15s;
    }
    #drop.is-over {
      border-color: var(--accent);
      background: #fff;
      color: var(--ink);
    }
    #drop strong { color: var(--ink); }
    #file { display: none; }
    #queue {
      list-style: none;
      margin: 0 0 1.25rem;
      padding: 0;
      display: flex;
      flex-direction: column;
      gap: 0.65rem;
    }
    #queue:empty { display: none; }
    .item {
      display: grid;
      grid-template-columns: 72px 1fr auto;
      gap: 0.75rem;
      align-items: start;
      padding: 0.65rem;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 6px;
    }
    .item.is-loading { opacity: 0.65; }
    .item img {
      width: 72px;
      height: 72px;
      object-fit: cover;
      border-radius: 3px;
      background: #eee;
    }
    .item .fields {
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
      min-width: 0;
    }
    .item input {
      width: 100%;
      padding: 0.35rem 0.5rem;
      border: 1px solid var(--line);
      border-radius: 3px;
      font: inherit;
    }
    .item .meta {
      font-size: 0.8rem;
      color: var(--muted);
      line-height: 1.35;
    }
    .item .rm {
      border: 0;
      background: transparent;
      color: var(--muted);
      cursor: pointer;
      font-size: 1.25rem;
      line-height: 1;
      padding: 0.25rem;
    }
    .item .rm:hover { color: var(--err); }
    .actions {
      display: flex;
      gap: 0.75rem;
      align-items: center;
      flex-wrap: wrap;
    }
    button.primary {
      appearance: none;
      border: 0;
      background: var(--accent);
      color: #fff;
      font: inherit;
      font-size: 0.95rem;
      padding: 0.65rem 1.15rem;
      border-radius: 4px;
      cursor: pointer;
    }
    button.primary:disabled {
      opacity: 0.45;
      cursor: not-allowed;
    }
    button.primary:hover:not(:disabled) {
      filter: brightness(1.05);
    }
    #status {
      margin-top: 1.25rem;
      font-size: 0.92rem;
    }
    #status .ok { color: var(--ok); }
    #status .err { color: var(--err); }
    #status li { margin: 0.35rem 0; }
    .hint {
      margin-top: 2rem;
      font-size: 0.85rem;
      color: var(--muted);
    }
  </style>
</head>
<body>
  <main>
    <h1>Photo upload</h1>
    <p class="sub">Local only · EXIF auto-fills date, timezone, GPS → place, camera · then R2 + <code>_photos/*.md</code></p>

    <div class="row">
      <label class="field"><span>Place fallback</span>
        <input type="text" id="place" placeholder="Filled from GPS when possible" autocomplete="off">
      </label>
      <label class="field"><span>Date fallback</span>
        <input type="date" id="date">
      </label>
    </div>
    <div class="row">
      <label class="field"><span>Camera fallback</span>
        <input type="text" id="camera" placeholder="From EXIF Make/Model/Lens" autocomplete="off">
      </label>
      <label class="field"><span>Baidu original folder</span>
        <input type="text" id="original_prefix" placeholder="/Photos/2024/Kyoto" autocomplete="off">
      </label>
    </div>

    <div id="drop" tabindex="0" role="button" aria-label="Drop photos here">
      <strong>Drop photos here</strong> or click to choose<br>
      <span style="font-size:0.85rem">Reads EXIF on drop — JPEG/HEIC/PNG</span>
    </div>
    <input type="file" id="file" accept="image/*" multiple>

    <ul id="queue"></ul>

    <div class="actions">
      <button type="button" class="primary" id="upload" disabled>Upload 0 photos</button>
      <span id="progress" style="color:var(--muted);font-size:0.9rem"></span>
    </div>

    <div id="status"></div>
    <p class="hint">Per-photo fields win over the fallbacks above. Decap → Photos is still fine for caption tweaks later.</p>
  </main>
  <script>
    const queueEl = document.getElementById('queue');
    const drop = document.getElementById('drop');
    const fileInput = document.getElementById('file');
    const uploadBtn = document.getElementById('upload');
    const progressEl = document.getElementById('progress');
    const statusEl = document.getElementById('status');
    const placeInput = document.getElementById('place');
    const dateInput = document.getElementById('date');
    const cameraInput = document.getElementById('camera');
    dateInput.value = new Date().toISOString().slice(0, 10);

    /** @type {Array<any>} */
    let items = [];

    function metaLine(item) {
      const bits = [];
      if (item.taken_at) bits.push(item.taken_at.replace('T', ' '));
      else if (item.date) bits.push(item.date);
      if (item.timezone) bits.push(item.timezone);
      if (item.lat != null && item.lon != null) {
        bits.push(item.lat.toFixed(5) + ', ' + item.lon.toFixed(5));
      }
      if (item.place) bits.push(item.place);
      if (item.camera) bits.push(item.camera);
      if (item.inspecting) return 'Reading EXIF…';
      if (!bits.length) return 'No EXIF found — using fallbacks';
      return bits.join(' · ');
    }

    function refresh() {
      const ready = items.length > 0 && items.every((it) => !it.inspecting);
      uploadBtn.disabled = !ready;
      uploadBtn.textContent = 'Upload ' + items.length + ' photo' + (items.length === 1 ? '' : 's');
      queueEl.innerHTML = '';
      items.forEach((item, i) => {
        const li = document.createElement('li');
        li.className = 'item' + (item.inspecting ? ' is-loading' : '');
        li.innerHTML =
          '<img alt="">' +
          '<div class="fields">' +
            '<input type="text" data-k="title" placeholder="Title">' +
            '<div class="row" style="gap:0.35rem">' +
              '<input type="date" data-k="date">' +
              '<input type="text" data-k="place" placeholder="Place">' +
            '</div>' +
            '<input type="text" data-k="camera" placeholder="Camera">' +
            '<div class="meta"></div>' +
          '</div>' +
          '<button type="button" class="rm" aria-label="Remove">&times;</button>';
        li.querySelector('img').src = item.url;
        li.querySelector('.meta').textContent = metaLine(item);
        li.querySelectorAll('input[data-k]').forEach((input) => {
          const k = input.getAttribute('data-k');
          input.value = item[k] || '';
          input.disabled = !!item.inspecting;
          input.addEventListener('input', () => { items[i][k] = input.value; });
        });
        li.querySelector('.rm').addEventListener('click', () => {
          URL.revokeObjectURL(item.url);
          items.splice(i, 1);
          refresh();
        });
        queueEl.appendChild(li);
      });
    }

    function applyBatchFallbacks() {
      // Fill empty global fallbacks from first photo that has the value
      for (const item of items) {
        if (!placeInput.value && item.place) placeInput.value = item.place;
        if (!cameraInput.value && item.camera) cameraInput.value = item.camera;
        if (item.date) dateInput.value = item.date;
        break;
      }
      // Prefer consensus place if several share one
      const places = items.map((it) => it.place).filter(Boolean);
      if (places.length) {
        const counts = {};
        places.forEach((p) => { counts[p] = (counts[p] || 0) + 1; });
        const top = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
        if (top && top[1] >= 1 && !placeInput.dataset.userEdited) {
          placeInput.value = top[0];
        }
      }
    }

    placeInput.addEventListener('input', () => { placeInput.dataset.userEdited = '1'; });

    async function inspectItem(item) {
      const fd = new FormData();
      fd.append('file', item.file, item.file.name);
      const res = await fetch('/api/inspect', { method: 'POST', body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || res.statusText);
      item.title = item.title || data.title_guess || item.file.name;
      item.date = data.date || '';
      item.taken_at = data.taken_at || '';
      item.timezone = data.timezone || '';
      item.lat = data.lat;
      item.lon = data.lon;
      item.place = data.place || '';
      item.camera = data.camera || '';
      item.inspecting = false;
    }

    async function addFiles(fileList) {
      const incoming = Array.from(fileList || []).filter((f) => f.type.startsWith('image/') || /\.(jpe?g|png|webp|heic|tif?f)$/i.test(f.name));
      for (const file of incoming) {
        const title = file.name.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' ').trim() || 'Untitled';
        const item = {
          file, title, url: URL.createObjectURL(file),
          date: '', taken_at: '', timezone: '', place: '', camera: '',
          lat: null, lon: null, inspecting: true,
        };
        items.push(item);
      }
      refresh();
      progressEl.textContent = 'Reading EXIF…';
      for (const item of items.filter((it) => it.inspecting)) {
        try {
          await inspectItem(item);
        } catch (err) {
          item.inspecting = false;
          item.place = item.place || '';
          console.warn(err);
        }
        refresh();
      }
      applyBatchFallbacks();
      progressEl.textContent = '';
      refresh();
    }

    drop.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', () => {
      addFiles(fileInput.files);
      fileInput.value = '';
    });
    ;['dragenter','dragover'].forEach((ev) => {
      drop.addEventListener(ev, (e) => {
        e.preventDefault();
        drop.classList.add('is-over');
      });
    });
    ;['dragleave','drop'].forEach((ev) => {
      drop.addEventListener(ev, (e) => {
        e.preventDefault();
        drop.classList.remove('is-over');
      });
    });
    drop.addEventListener('drop', (e) => addFiles(e.dataTransfer.files));

    uploadBtn.addEventListener('click', async () => {
      if (!items.length || items.some((it) => it.inspecting)) return;
      uploadBtn.disabled = true;
      statusEl.innerHTML = '';
      const placeFallback = placeInput.value.trim();
      const cameraFallback = cameraInput.value.trim();
      const dateFallback = dateInput.value;
      const originalPrefix = document.getElementById('original_prefix').value.trim().replace(/\/+$/, '');
      const results = [];
      const batch = items.slice();

      for (let i = 0; i < batch.length; i++) {
        const item = batch[i];
        progressEl.textContent = 'Uploading ' + (i + 1) + ' / ' + batch.length + ' — ' + item.file.name;
        const fd = new FormData();
        fd.append('file', item.file, item.file.name);
        fd.append('title', item.title || item.file.name);
        fd.append('place', (item.place || placeFallback));
        fd.append('camera', (item.camera || cameraFallback));
        fd.append('date', item.date || dateFallback);
        if (item.taken_at) fd.append('taken_at', item.taken_at);
        if (item.timezone) fd.append('timezone', item.timezone);
        if (item.lat != null) fd.append('lat', String(item.lat));
        if (item.lon != null) fd.append('lon', String(item.lon));
        if (originalPrefix) {
          fd.append('original', originalPrefix + '/' + item.file.name);
        }
        try {
          const res = await fetch('/api/upload', { method: 'POST', body: fd });
          const data = await res.json();
          if (!res.ok) throw new Error(data.error || res.statusText);
          results.push({ ok: true, ...data });
        } catch (err) {
          results.push({ ok: false, title: item.title, error: String(err.message || err) });
        }
      }

      progressEl.textContent = 'Done.';
      const ul = document.createElement('ul');
      results.forEach((r) => {
        const li = document.createElement('li');
        if (r.ok) {
          li.className = 'ok';
          const extra = [r.place, r.taken_at || r.date, r.camera].filter(Boolean).join(' · ');
          li.innerHTML = '✓ <strong>' + escapeHtml(r.title) + '</strong>' +
            (extra ? ' <span style="color:var(--muted)">' + escapeHtml(extra) + '</span>' : '') +
            ' → <a href="' + escapeHtml(r.image_url) + '" target="_blank" rel="noopener">' +
            escapeHtml(r.md) + '</a>';
        } else {
          li.className = 'err';
          li.textContent = '✗ ' + r.title + ' — ' + r.error;
        }
        ul.appendChild(li);
      });
      statusEl.appendChild(ul);

      const okCount = results.filter((r) => r.ok).length;
      if (okCount) {
        items = [];
        refresh();
      } else {
        uploadBtn.disabled = false;
      }
    });

    function escapeHtml(s) {
      return String(s).replace(/[&<>"']/g, (c) => ({
        '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
      })[c]);
    }
  </script>
</body>
</html>
"""


def _parse_multipart(handler: BaseHTTPRequestHandler) -> dict:
    """Parse multipart form into {field: str|tuple(filename, bytes)}."""
    content_type = handler.headers.get("Content-Type", "")
    if "multipart/form-data" not in content_type:
        raise PhotosError("Expected multipart/form-data")

    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0 or length > 80 * 1024 * 1024:
        raise PhotosError("Invalid or too-large upload (max 80MB)")

    body = handler.rfile.read(length)
    boundary = None
    for part in content_type.split(";"):
        part = part.strip()
        if part.startswith("boundary="):
            boundary = part.split("=", 1)[1].strip().strip('"')
            break
    if not boundary:
        raise PhotosError("Missing multipart boundary")

    marker = b"--" + boundary.encode("ascii", errors="ignore")
    chunks = body.split(marker)
    fields: dict = {}

    for chunk in chunks:
        if not chunk or chunk in (b"--\r\n", b"--", b"--\n"):
            continue
        if chunk.startswith(b"\r\n"):
            chunk = chunk[2:]
        elif chunk.startswith(b"\n"):
            chunk = chunk[1:]
        if chunk.endswith(b"\r\n"):
            chunk = chunk[:-2]
        elif chunk.endswith(b"\n"):
            chunk = chunk[:-1]
        if chunk == b"--":
            continue

        header_blob, _, data = chunk.partition(b"\r\n\r\n")
        if not _:
            header_blob, _, data = chunk.partition(b"\n\n")
        if data.endswith(b"\r\n"):
            data = data[:-2]
        elif data.endswith(b"\n"):
            data = data[:-1]

        headers = header_blob.decode("utf-8", errors="replace")
        name = None
        filename = None
        for line in headers.splitlines():
            if line.lower().startswith("content-disposition:"):
                for bit in line.split(";"):
                    bit = bit.strip()
                    if bit.startswith("name="):
                        name = bit.split("=", 1)[1].strip().strip('"')
                    elif bit.startswith("filename="):
                        filename = bit.split("=", 1)[1].strip().strip('"')
        if not name:
            continue
        if filename is not None:
            fields[name] = (filename, data)
        else:
            fields[name] = data.decode("utf-8", errors="replace")
    return fields


def _optional_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class Handler(BaseHTTPRequestHandler):
    server_version = "PhotosUpload/1.0"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/health":
            try:
                env = load_env()
                payload = {
                    "ok": True,
                    "bucket": env["R2_BUCKET"],
                    "public_base": env["R2_PUBLIC_BASE"],
                }
            except PhotosError as err:
                payload = {"ok": False, "error": str(err)}
            raw = json.dumps(payload).encode("utf-8")
            self._send(200 if payload["ok"] else 500, raw, "application/json")
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in ("/api/upload", "/api/inspect"):
            self._send(404, b'{"error":"not found"}', "application/json")
            return
        try:
            fields = _parse_multipart(self)
            file_field = fields.get("file")
            if not isinstance(file_field, tuple):
                raise PhotosError("Missing file")
            filename, data = file_field
            if not data:
                raise PhotosError("Empty file")

            if path == "/api/inspect":
                meta = inspect_photo_bytes(data, filename, geocode=True)
                raw = json.dumps(meta).encode("utf-8")
                self._send(200, raw, "application/json")
                return

            title = (fields.get("title") or title_from_filename(filename)).strip()
            place = str(fields.get("place") or "").strip()
            camera = str(fields.get("camera") or "").strip()
            original = str(fields.get("original") or "").strip()
            taken_at = str(fields.get("taken_at") or "").strip()
            timezone_offset = str(fields.get("timezone") or "").strip()
            lat = _optional_float(fields.get("lat"))
            lon = _optional_float(fields.get("lon"))
            when = parse_date(str(fields.get("date") or "") or None)
            result = add_photo_bytes(
                data,
                filename,
                title=title,
                place=place,
                when=when,
                camera=camera,
                original=original,
                taken_at=taken_at,
                timezone_offset=timezone_offset,
                lat=lat,
                lon=lon,
                geocode=not place,
            )
            raw = json.dumps(result).encode("utf-8")
            self._send(200, raw, "application/json")
        except PhotosError as err:
            raw = json.dumps({"error": str(err)}).encode("utf-8")
            self._send(400, raw, "application/json")
        except Exception as err:  # noqa: BLE001
            raw = json.dumps({"error": str(err)}).encode("utf-8")
            self._send(500, raw, "application/json")


def main() -> None:
    try:
        env = load_env()
    except PhotosError as err:
        print(err, file=sys.stderr)
        sys.exit(1)

    url = f"http://{HOST}:{PORT}/"
    print(f"Photo upload UI → {url}")
    print(f"R2 bucket: {env['R2_BUCKET']}  public: {env['R2_PUBLIC_BASE']}")
    print("Bound to localhost only. Ctrl+C to stop.")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
