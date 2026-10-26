"""Shared helpers for portfolio photo upload (CLI + local web UI)."""

from __future__ import annotations

import json
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone, tzinfo
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from PIL import Image, ImageOps
from PIL.ExifTags import GPSTAGS, IFD

ROOT = Path(__file__).resolve().parents[1]
PHOTOS_DIR = ROOT / "_photos"
ALBUMS_DIR = ROOT / "_albums"
WORK_DIR = Path(__file__).resolve().parent / ".photos-work"
ENV_PATH = Path(__file__).resolve().parent / "r2.env"

WEB_MAX = 2400
THUMB_MAX = 600
JPEG_QUALITY_WEB = 85
JPEG_QUALITY_THUMB = 80

# EXIF tags
_DT_ORIGINAL = 0x9003  # DateTimeOriginal
_DT_DIGITIZED = 0x9004
_DT = 0x0132  # DateTime
_OFFSET_ORIGINAL = 0x9011  # OffsetTimeOriginal
_OFFSET_DIGITIZED = 0x9012
_OFFSET = 0x9010  # OffsetTime
_MAKE = 0x010F
_MODEL = 0x0110
_LENS_MODEL = 0xA434
_LENS_MAKE = 0xA433
_SUBSEC_ORIGINAL = 0x9291


class PhotosError(Exception):
    pass


def load_env(path: Path = ENV_PATH) -> dict[str, str]:
    if not path.is_file():
        raise PhotosError(
            f"Missing {path}. Copy r2.env.example → r2.env and fill credentials."
        )
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        env[key.strip()] = val.strip().strip('"').strip("'")
    required = [
        "R2_ACCOUNT_ID",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_BUCKET",
        "R2_PUBLIC_BASE",
    ]
    missing = [k for k in required if not env.get(k)]
    if missing:
        raise PhotosError(f"r2.env missing: {', '.join(missing)}")
    return env


def slugify(text: str) -> str:
    s = text.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"[-\s]+", "-", s).strip("-")
    return s or "photo"


_UUIDish = re.compile(
    r"^[0-9a-f]{8}[-_ ]?[0-9a-f]{4}[-_ ]?[0-9a-f]{4}[-_ ]?[0-9a-f]{4}[-_ ]?[0-9a-f]{12}",
    re.I,
)
# IMG_1234, DSC00001, P1234567, export hashes, etc.
_BORING_NAME = re.compile(
    r"^(img[_\- ]?\d+|dsc[_\- ]?\d+|p\d{4,}|\d{8,}[_-]?\d*|untitled|photo|image)([_\- ]+\d+)*$",
    re.I,
)


def title_from_filename(name: str) -> str:
    """Human title from filename; empty string if the name is a UUID/export hash."""
    stem = Path(name).stem
    stem = re.sub(r"[_\-]+", " ", stem).strip()
    stem = re.sub(r"\s+", " ", stem)
    if not stem:
        return ""
    compact = stem.replace(" ", "").replace("-", "")
    if _UUIDish.match(stem.replace(" ", "-")) or _UUIDish.match(compact):
        return ""
    # UUID with trailing junk: "4268d494-…-1-105-c"
    if re.match(r"^[0-9a-f]{8}[- ].+", stem, re.I) and len(re.findall(r"[0-9a-f]{4,}", stem, re.I)) >= 3:
        return ""
    if _BORING_NAME.match(stem.replace(" ", "")) or _BORING_NAME.match(stem):
        return ""
    return stem


def _month_year(when: date) -> str:
    return when.strftime("%B %Y")  # e.g. March 2018


def default_title(*, filename: str, place: str = "", when: date | None = None) -> str:
    """
    Human gallery title. Prefer a meaningful filename; otherwise
    "Austin, March 2018" / "March 2018" / "Untitled".
    """
    from_name = title_from_filename(filename)
    if from_name:
        return from_name
    locality = (place.split(",")[0].strip() if place else "") or ""
    if locality and when:
        return f"{locality}, {_month_year(when)}"
    if locality:
        return locality
    if when:
        return _month_year(when)
    return "Untitled"


def _parse_front_matter(path: Path) -> dict | None:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    fm: dict = {"slug": path.stem, "path": str(path.relative_to(ROOT))}
    for line in parts[1].splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key in ("lat", "lon"):
            try:
                fm[key] = float(val)
            except ValueError:
                fm[key] = val
        elif key == "published":
            fm[key] = val.lower() in ("true", "yes", "1")
        else:
            fm[key] = val
    body = parts[2].strip()
    if body:
        fm["description"] = body
        fm["body"] = body
    return fm


def list_photo_entries() -> list[dict]:
    """Load published photo front matter for the local editor."""
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    for path in sorted(PHOTOS_DIR.glob("*.md"), reverse=True):
        fm = _parse_front_matter(path)
        if fm and fm.get("image"):
            entries.append(fm)
    return entries


def list_albums() -> list[dict]:
    ALBUMS_DIR.mkdir(parents=True, exist_ok=True)
    albums: list[dict] = []
    for path in sorted(ALBUMS_DIR.glob("*.md"), reverse=True):
        fm = _parse_front_matter(path)
        if fm and fm.get("published") is not False:
            albums.append(fm)
    return albums


def ensure_album(
    slug: str,
    *,
    title: str = "",
    place: str = "",
    when: date | None = None,
) -> dict:
    """Create _albums/<slug>.md if missing; return album dict."""
    slug = slugify(slug)
    if not slug:
        raise PhotosError("Album slug is empty")
    ALBUMS_DIR.mkdir(parents=True, exist_ok=True)
    path = ALBUMS_DIR / f"{slug}.md"
    if path.is_file():
        existing = _parse_front_matter(path) or {"slug": slug}
        return existing

    when = when or date.today()
    title = (title or slug.replace("-", " ").title()).strip()
    lines = [
        "---",
        f"title: {yaml_escape(title)}",
        f"key: {yaml_escape(slug)}",
        f"date: {when.isoformat()}",
    ]
    if place:
        lines.append(f"place: {yaml_escape(place)}")
    lines.append("published: true")
    lines.append("---")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "slug": slug,
        "key": slug,
        "title": title,
        "date": when.isoformat(),
        "place": place,
        "published": True,
        "path": str(path.relative_to(ROOT)),
    }


def update_photo_fields(slug: str, **fields) -> dict:
    """Update selected front-matter fields (and optional description body)."""
    path = PHOTOS_DIR / f"{slug}.md"
    if not path.is_file():
        raise PhotosError(f"No photo entry: {slug}")
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise PhotosError(f"Invalid front matter: {slug}")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise PhotosError(f"Invalid front matter: {slug}")
    fm_lines = parts[1].splitlines()
    body = parts[2]
    if "description" in fields and fields["description"] is not None:
        body = "\n" + str(fields["description"]).strip() + "\n"
    allowed = {
        "title",
        "place",
        "camera",
        "date",
        "taken_at",
        "timezone",
        "published",
        "album",
    }
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}

    def format_value(key: str, value) -> str:
        if key == "published":
            flag = str(value).lower() in ("true", "yes", "1")
            return "true" if flag else "false"
        return yaml_escape(str(value).strip())

    def set_line(lines: list[str], key: str, value) -> list[str]:
        rendered = f"{key}: {format_value(key, value)}"
        out: list[str] = []
        found = False
        for line in lines:
            if line.strip().startswith(f"{key}:"):
                out.append(rendered)
                found = True
            else:
                out.append(line)
        if not found and str(value).strip() != "":
            insert_at = len(out)
            for i, line in enumerate(out):
                if line.strip().startswith("image:"):
                    insert_at = i
                    break
            out.insert(insert_at, rendered)
        return out

    for key, value in updates.items():
        fm_lines = set_line(fm_lines, key, value)

    fm_block = "\n".join(fm_lines).strip("\n")
    new_text = f"---\n{fm_block}\n---\n{body.lstrip(chr(10))}"
    if not new_text.endswith("\n"):
        new_text += "\n"
    path.write_text(new_text, encoding="utf-8")
    updated = next((e for e in list_photo_entries() if e["slug"] == slug), {"slug": slug})
    return updated


def parse_date(value: str | None) -> date | None:
    """Parse YYYY-MM-DD (etc). Empty → None (caller may fall back to EXIF)."""
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%Y"):
        try:
            dt = datetime.strptime(value, fmt)
            if fmt == "%Y":
                return date(dt.year, 1, 1)
            return dt.date()
        except ValueError:
            continue
    raise PhotosError(f"Could not parse date: {value!r} (use YYYY-MM-DD)")


def unique_slug(base: str) -> str:
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
    slug = base
    n = 2
    while (PHOTOS_DIR / f"{slug}.md").exists():
        slug = f"{base}-{n}"
        n += 1
    return slug


def _as_float(value) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, tuple) and len(value) == 2:
            return float(value[0]) / float(value[1]) if value[1] else None
        if isinstance(value, Fraction):
            return float(value)
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _dms_to_decimal(values, ref: str | None) -> float | None:
    if not values or len(values) < 3:
        return None
    deg = _as_float(values[0])
    minutes = _as_float(values[1])
    seconds = _as_float(values[2])
    if deg is None or minutes is None or seconds is None:
        return None
    decimal = deg + minutes / 60.0 + seconds / 3600.0
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal


def _parse_exif_datetime(raw: str) -> datetime | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y:%m:%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _parse_offset(raw: str | None) -> tzinfo | None:
    if not raw:
        return None
    raw = raw.strip()
    m = re.fullmatch(r"([+-])(\d{2}):?(\d{2})", raw)
    if not m:
        return None
    sign = 1 if m.group(1) == "+" else -1
    hours = int(m.group(2))
    minutes = int(m.group(3))
    return timezone(sign * timedelta(hours=hours, minutes=minutes))


def _format_offset(tz: tzinfo | None) -> str:
    if tz is None:
        return ""
    off = datetime.now(tz).utcoffset()
    if off is None:
        return ""
    total = int(off.total_seconds())
    sign = "+" if total >= 0 else "-"
    total = abs(total)
    return f"{sign}{total // 3600:02d}:{(total % 3600) // 60:02d}"


def _camera_label(make: str, model: str, lens: str) -> str:
    make = (make or "").strip()
    model = (model or "").strip()
    lens = (lens or "").strip()
    # Avoid "Canon Canon EOS …"
    if make and model.upper().startswith(make.upper()):
        body = model
    elif make and model:
        body = f"{make} {model}"
    else:
        body = model or make
    if lens and lens.lower() not in body.lower():
        return f"{body} · {lens}" if body else lens
    return body


def extract_photo_meta(src: Path) -> dict:
    """
    Read EXIF (and GPS) from an image file.

    Returns keys: date, taken_at, timezone, lat, lon, camera, make, model, lens,
    and source flags for what was found.
    """
    src = Path(src)
    meta: dict = {
        "date": None,
        "taken_at": None,
        "timezone": "",
        "lat": None,
        "lon": None,
        "camera": "",
        "make": "",
        "model": "",
        "lens": "",
        "has_exif": False,
        "has_gps": False,
    }
    try:
        with Image.open(src) as im:
            exif = im.getexif()
            if not exif:
                return meta
            meta["has_exif"] = True

            # Merge Exif IFD (where DateTimeOriginal usually lives)
            try:
                exif_ifd = exif.get_ifd(IFD.Exif)
            except Exception:
                exif_ifd = {}

            def get_tag(*tags):
                for tag in tags:
                    if tag in exif_ifd and exif_ifd[tag] not in (None, ""):
                        return exif_ifd[tag]
                    if tag in exif and exif[tag] not in (None, ""):
                        return exif[tag]
                return None

            dt_raw = get_tag(_DT_ORIGINAL, _DT_DIGITIZED, _DT)
            offset_raw = get_tag(_OFFSET_ORIGINAL, _OFFSET_DIGITIZED, _OFFSET)
            make = str(get_tag(_MAKE) or "").strip()
            model = str(get_tag(_MODEL) or "").strip()
            lens = str(get_tag(_LENS_MODEL) or get_tag(_LENS_MAKE) or "").strip()

            meta["make"] = make
            meta["model"] = model
            meta["lens"] = lens
            meta["camera"] = _camera_label(make, model, lens)

            naive = _parse_exif_datetime(str(dt_raw) if dt_raw else "")
            tz = _parse_offset(str(offset_raw) if offset_raw else None)
            meta["timezone"] = _format_offset(tz) if tz else (
                str(offset_raw).strip() if offset_raw else ""
            )

            if naive is not None:
                aware = naive.replace(tzinfo=tz) if tz else naive
                meta["date"] = aware.date().isoformat()
                if tz:
                    meta["taken_at"] = aware.isoformat(timespec="seconds")
                else:
                    meta["taken_at"] = naive.strftime("%Y-%m-%dT%H:%M:%S")

            # GPS
            try:
                gps_ifd = exif.get_ifd(IFD.GPSInfo)
            except Exception:
                gps_ifd = {}
            if gps_ifd:
                gps = {GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}
                lat = _dms_to_decimal(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef"))
                lon = _dms_to_decimal(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef"))
                if lat is not None and lon is not None:
                    # (0,0) is almost always a bad/missing GPS fix
                    if abs(lat) > 0.01 or abs(lon) > 0.01:
                        meta["lat"] = round(lat, 6)
                        meta["lon"] = round(lon, 6)
                        meta["has_gps"] = True
    except OSError:
        return meta
    return meta


def reverse_geocode(lat: float, lon: float) -> str:
    """Best-effort place label via Nominatim (cached to ~100m)."""
    return _reverse_geocode_cached(round(float(lat), 3), round(float(lon), 3))


@lru_cache(maxsize=256)
def _reverse_geocode_cached(lat: float, lon: float) -> str:
    # Nominatim usage policy: max ~1 req/s
    time.sleep(1.05)
    params = urllib.parse.urlencode(
        {
            "lat": f"{lat:.5f}",
            "lon": f"{lon:.5f}",
            "format": "json",
            "zoom": "12",
            "addressdetails": "1",
        }
    )
    url = f"https://nominatim.openstreetmap.org/reverse?{params}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "zlu.me-photos/1.0 (personal portfolio; hello@zlu.me)",
            "Accept": "application/json",
            "Accept-Language": "en",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return ""

    addr = data.get("address") or {}
    locality = (
        addr.get("city")
        or addr.get("town")
        or addr.get("village")
        or addr.get("municipality")
        or addr.get("county")
        or addr.get("state_district")
    )
    region = addr.get("state") or addr.get("region")
    country = addr.get("country")
    parts = []
    if locality:
        parts.append(locality)
    if region and region not in parts:
        parts.append(region)
    if country and country not in parts:
        parts.append(country)
    if parts:
        return ", ".join(parts)
    return (data.get("display_name") or "").split(",")[0].strip()


def enrich_meta(meta: dict, *, geocode: bool = True) -> dict:
    """Add place from GPS when possible. Mutates and returns meta."""
    out = dict(meta)
    out.setdefault("place", "")
    if geocode and out.get("lat") is not None and out.get("lon") is not None and not out.get("place"):
        place = reverse_geocode(float(out["lat"]), float(out["lon"]))
        if place:
            out["place"] = place
    return out


def inspect_photo(src: Path, *, geocode: bool = True) -> dict:
    meta = extract_photo_meta(src)
    meta = enrich_meta(meta, geocode=geocode)
    when = date.fromisoformat(meta["date"]) if meta.get("date") else None
    meta["title_guess"] = default_title(
        filename=src.name, place=meta.get("place") or "", when=when
    )
    return meta


def inspect_photo_bytes(data: bytes, filename: str, *, geocode: bool = True) -> dict:
    suffix = Path(filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        meta = inspect_photo(tmp_path, geocode=geocode)
        meta["filename"] = filename
        return meta
    finally:
        tmp_path.unlink(missing_ok=True)


def resize_to_jpeg(src: Path, dest: Path, max_edge: int, quality: int) -> tuple[int, int]:
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        elif im.mode == "L":
            im = im.convert("RGB")
        im.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        dest.parent.mkdir(parents=True, exist_ok=True)
        im.save(dest, format="JPEG", quality=quality, optimize=True, progressive=True)
        return im.size


def r2_client(env: dict[str, str]):
    endpoint = f"https://{env['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=env["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=env["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def upload_file(client, bucket: str, key: str, path: Path) -> None:
    client.upload_file(
        str(path),
        bucket,
        key,
        ExtraArgs={
            "ContentType": "image/jpeg",
            "CacheControl": "public, max-age=31536000",
        },
    )


def yaml_escape(value: str) -> str:
    if value == "":
        return '""'
    if re.search(r'[:#\[\]{},&*?|!>\'" %@`]', value) or value.strip() != value:
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return value


def write_photo_md(
    slug: str,
    *,
    title: str,
    when: date,
    place: str,
    camera: str,
    image_url: str,
    thumb_url: str,
    original: str,
    tags: list[str],
    taken_at: str = "",
    timezone_offset: str = "",
    lat: float | None = None,
    lon: float | None = None,
    album: str = "",
) -> Path:
    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
    path = PHOTOS_DIR / f"{slug}.md"
    if path.exists():
        raise PhotosError(f"Refusing to overwrite existing {path}")

    lines = [
        "---",
        f"title: {yaml_escape(title)}",
        f"date: {when.isoformat()}",
    ]
    if taken_at:
        lines.append(f"taken_at: {yaml_escape(taken_at)}")
    if timezone_offset:
        lines.append(f"timezone: {yaml_escape(timezone_offset)}")
    if place:
        lines.append(f"place: {yaml_escape(place)}")
    if album:
        lines.append(f"album: {yaml_escape(album)}")
    if lat is not None and lon is not None:
        lines.append(f"lat: {lat}")
        lines.append(f"lon: {lon}")
    if camera:
        lines.append(f"camera: {yaml_escape(camera)}")
    lines.append(f"image: {yaml_escape(image_url)}")
    lines.append(f"thumb: {yaml_escape(thumb_url)}")
    if original:
        lines.append(f"original: {yaml_escape(original)}")
    lines.append("published: true")
    if tags:
        lines.append("tags:")
        for tag in tags:
            lines.append(f"  - {yaml_escape(tag)}")
    lines.append("---")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def add_photo(
    src: Path,
    *,
    title: str = "",
    place: str = "",
    when: date | None = None,
    camera: str = "",
    original: str = "",
    slug: str = "",
    tags: list[str] | None = None,
    taken_at: str = "",
    timezone_offset: str = "",
    lat: float | None = None,
    lon: float | None = None,
    album: str = "",
    album_title: str = "",
    dry_run: bool = False,
    geocode: bool = True,
    env: dict[str, str] | None = None,
) -> dict:
    """Resize, upload to R2, write _photos/<slug>.md. EXIF fills any blank fields."""
    src = Path(src)
    if not src.is_file():
        raise PhotosError(f"File not found: {src}")

    exif = enrich_meta(extract_photo_meta(src), geocode=geocode and not place)

    place = (place or exif.get("place") or "").strip()
    camera = (camera or exif.get("camera") or "").strip()
    taken_at = (taken_at or exif.get("taken_at") or "").strip()
    timezone_offset = (timezone_offset or exif.get("timezone") or "").strip()
    if lat is None:
        lat = exif.get("lat")
    if lon is None:
        lon = exif.get("lon")

    if when is None:
        if exif.get("date"):
            when = date.fromisoformat(exif["date"])
        else:
            when = date.today()

    title = (title or "").strip() or default_title(
        filename=src.name, place=place, when=when
    )

    album = slugify(album) if album else ""
    if album:
        ensure_album(
            album,
            title=album_title or album.replace("-", " ").title(),
            place=place,
            when=when,
        )

    tags = tags or []
    base_slug = slugify(slug or title)
    if when.year and not base_slug.startswith(str(when.year)):
        base_slug = f"{when.year}-{base_slug}"
    final_slug = unique_slug(base_slug)

    work = WORK_DIR / final_slug
    web_path = work / "web.jpg"
    thumb_path = work / "thumb.jpg"
    resize_to_jpeg(src, web_path, WEB_MAX, JPEG_QUALITY_WEB)
    resize_to_jpeg(src, thumb_path, THUMB_MAX, JPEG_QUALITY_THUMB)

    if dry_run:
        return {
            "slug": final_slug,
            "title": title,
            "place": place,
            "camera": camera,
            "album": album,
            "date": when.isoformat(),
            "taken_at": taken_at,
            "timezone": timezone_offset,
            "lat": lat,
            "lon": lon,
            "dry_run": True,
            "web_path": str(web_path),
            "thumb_path": str(thumb_path),
        }

    env = env or load_env()
    base = env["R2_PUBLIC_BASE"].rstrip("/")
    web_key = f"web/{final_slug}.jpg"
    thumb_key = f"thumbs/{final_slug}.jpg"
    image_url = f"{base}/{web_key}"
    thumb_url = f"{base}/{thumb_key}"

    client = r2_client(env)
    bucket = env["R2_BUCKET"]
    try:
        upload_file(client, bucket, web_key, web_path)
        upload_file(client, bucket, thumb_key, thumb_path)
    except (BotoCoreError, ClientError) as err:
        raise PhotosError(f"R2 upload failed: {err}") from err

    md_path = write_photo_md(
        final_slug,
        title=title,
        when=when,
        place=place,
        camera=camera,
        image_url=image_url,
        thumb_url=thumb_url,
        original=original,
        tags=tags,
        taken_at=taken_at,
        timezone_offset=timezone_offset,
        lat=lat,
        lon=lon,
        album=album,
    )
    return {
        "slug": final_slug,
        "title": title,
        "place": place,
        "camera": camera,
        "album": album,
        "date": when.isoformat(),
        "taken_at": taken_at,
        "timezone": timezone_offset,
        "lat": lat,
        "lon": lon,
        "image_url": image_url,
        "thumb_url": thumb_url,
        "md": str(md_path.relative_to(ROOT)),
        "dry_run": False,
    }


def add_photo_bytes(
    data: bytes,
    filename: str,
    **kwargs,
) -> dict:
    suffix = Path(filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        return add_photo(tmp_path, **kwargs)
    finally:
        tmp_path.unlink(missing_ok=True)
