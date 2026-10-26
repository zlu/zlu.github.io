#!/usr/bin/env python3
"""
CLI: resize one photo, upload to R2, write _photos/<slug>.md.
EXIF fills date / timezone / GPS→place / camera when flags are omitted.

Prefer the drag-and-drop UI:
  python3 .scripts/photos_serve.py

CLI usage:
  python3 .scripts/photos_add.py ./IMG_1234.jpg --title "Fushimi Inari"
  python3 .scripts/photos_add.py ./IMG_1234.jpg   # title from filename; rest from EXIF
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from photos_lib import PhotosError, add_photo, parse_date
except ImportError:
    print(
        "Missing deps. Run: pip install -r .scripts/requirements-photos.txt",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add a photo to the zlu.me portfolio")
    parser.add_argument("file", type=Path, help="Source image")
    parser.add_argument("--title", default="", help="Gallery title (default: filename)")
    parser.add_argument("--place", default="", help="Place / city (default: GPS reverse-geocode)")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (default: EXIF DateTimeOriginal)")
    parser.add_argument("--camera", default="", help="Camera / lens (default: EXIF Make/Model)")
    parser.add_argument("--original", default="", help="Baidu path to the master file")
    parser.add_argument("--album", default="", help="Album/trip slug, e.g. austin-2018")
    parser.add_argument("--slug", default="", help="URL slug (default: from title)")
    parser.add_argument("--tag", action="append", default=[], dest="tags", help="Tag (repeatable)")
    parser.add_argument("--no-geocode", action="store_true", help="Skip Nominatim place lookup")
    parser.add_argument("--dry-run", action="store_true", help="Resize + EXIF only; skip R2 + markdown")
    args = parser.parse_args()

    try:
        result = add_photo(
            args.file.expanduser().resolve(),
            title=args.title,
            place=args.place,
            when=parse_date(args.date),
            camera=args.camera,
            original=args.original,
            album=args.album,
            slug=args.slug,
            tags=args.tags,
            dry_run=args.dry_run,
            geocode=not args.no_geocode,
        )
    except PhotosError as err:
        print(err, file=sys.stderr)
        sys.exit(1)

    if result.get("dry_run"):
        print(f"Dry run OK — {result['slug']}")
        print(f"  date     {result.get('date')}")
        print(f"  taken_at {result.get('taken_at')}")
        print(f"  timezone {result.get('timezone')}")
        print(f"  place    {result.get('place')}")
        print(f"  camera   {result.get('camera')}")
        print(f"  lat/lon  {result.get('lat')}, {result.get('lon')}")
        print(f"  web      {result['web_path']}")
        return

    print(f"Wrote {result['md']}")
    print(f"  {result.get('taken_at') or result.get('date')}  {result.get('place')}  {result.get('camera')}")
    print(f"Live image: {result['image_url']}")
    print("Next: commit _photos/*.md and push.")


if __name__ == "__main__":
    main()
