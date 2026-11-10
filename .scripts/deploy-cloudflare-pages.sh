#!/usr/bin/env bash
# Build Jekyll locally and upload _site to Cloudflare Pages (no GitHub Actions).
#
# Prereqs:
#   - `wrangler login` (or CLOUDFLARE_API_TOKEN with Pages Edit)
#   - CLOUDFLARE_ACCOUNT_ID, or R2_ACCOUNT_ID in .scripts/r2.env
#
# Usage:
#   .scripts/deploy-cloudflare-pages.sh              # build (CF overlay) + deploy
#   .scripts/deploy-cloudflare-pages.sh --skip-build # upload existing _site
#   .scripts/deploy-cloudflare-pages.sh --full        # include teach_retired_redirects (needs >20k file plan)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Prefer Homebrew Ruby (mise.toml) over macOS system Ruby.
if [[ -d /opt/homebrew/opt/ruby/bin ]]; then
  export PATH="/opt/homebrew/opt/ruby/bin:$PATH"
fi

# Existing Cloudflare Pages project (CF Git builds fail on custom _plugins — use this script instead).
PROJECT_NAME="${CLOUDFLARE_PAGES_PROJECT:-zlu-github-io}"
SKIP_BUILD=0
FULL_BUILD=0
for arg in "$@"; do
  case "$arg" in
    --skip-build) SKIP_BUILD=1 ;;
    --full) FULL_BUILD=1 ;;
    -h|--help)
      sed -n '2,14p' "$0"
      exit 0
      ;;
  esac
done

if [[ -z "${CLOUDFLARE_ACCOUNT_ID:-}" && -f .scripts/r2.env ]]; then
  CLOUDFLARE_ACCOUNT_ID="$(grep -E '^R2_ACCOUNT_ID=' .scripts/r2.env | cut -d= -f2-)"
  export CLOUDFLARE_ACCOUNT_ID
fi

if [[ -z "${CLOUDFLARE_ACCOUNT_ID:-}" ]]; then
  echo "Set CLOUDFLARE_ACCOUNT_ID (or put R2_ACCOUNT_ID in .scripts/r2.env)." >&2
  exit 1
fi

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  if [[ "$FULL_BUILD" -eq 1 ]]; then
    echo "→ jekyll build (production, full teach redirects)"
    JEKYLL_ENV=production bundle exec jekyll build
  else
    echo "→ jekyll build (production + _config.cloudflare-pages.yml)"
    JEKYLL_ENV=production bundle exec jekyll build \
      --config _config.yml,_config.cloudflare-pages.yml
  fi
else
  echo "→ skipping build (using existing _site)"
  [[ -d _site ]] || { echo "_site missing; run without --skip-build" >&2; exit 1; }
fi

FILE_COUNT="$(find _site -type f | wc -l | tr -d ' ')"
echo "→ _site file count: ${FILE_COUNT} (Cloudflare free plan limit: 20000)"
if [[ "$FILE_COUNT" -gt 20000 ]]; then
  echo "Too many files for free Pages. Re-run without --full, or raise the plan / move redirects to a Function." >&2
  exit 1
fi

echo "→ wrangler pages deploy → project=${PROJECT_NAME}"
wrangler pages deploy _site \
  --project-name="$PROJECT_NAME" \
  --commit-dirty=true

echo
echo "Done. Open the *.pages.dev URL above (zlu.me DNS unchanged)."
echo "Tip: in the CF dashboard, disable automatic Git builds for ${PROJECT_NAME} so failed CF builds stop firing on every push."
