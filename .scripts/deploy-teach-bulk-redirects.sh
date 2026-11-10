#!/usr/bin/env bash
# Upload GSC-hit retired teach uni shells to Cloudflare Bulk Redirects (Free ≤10k).
#
# Prereq token permissions:
#   - Account Filter Lists Edit
#   - Account Rulesets Write  (or Mass URL Redirects Write)
#
#   export CLOUDFLARE_API_TOKEN=...
#   export CLOUDFLARE_ACCOUNT_ID=...   # or R2_ACCOUNT_ID in .scripts/r2.env
#
# No token? Import the CSV in the dashboard instead:
#   https://dash.cloudflare.com/?to=/:account/bulk-redirects
#   Create list → import .scripts/teach-uni-bulk-redirects.csv → create rule → Deploy
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CSV="${1:-.scripts/teach-uni-bulk-redirects.csv}"
LIST_NAME="${BULK_REDIRECT_LIST_NAME:-teach_retired_uni_shells}"
ACCOUNT_ID="${CLOUDFLARE_ACCOUNT_ID:-}"

if [[ -z "$ACCOUNT_ID" && -f .scripts/r2.env ]]; then
  ACCOUNT_ID="$(grep -E '^R2_ACCOUNT_ID=' .scripts/r2.env | cut -d= -f2-)"
fi

: "${ACCOUNT_ID:?Set CLOUDFLARE_ACCOUNT_ID (or R2_ACCOUNT_ID in .scripts/r2.env)}"
: "${CLOUDFLARE_API_TOKEN:?Set CLOUDFLARE_API_TOKEN}"

[[ -f "$CSV" ]] || {
  echo "Missing $CSV" >&2
  echo "Generate: python3 .scripts/generate_teach_bulk_redirects.py --gsc-pages Pages.csv" >&2
  exit 1
}

API="https://api.cloudflare.com/client/v4/accounts/${ACCOUNT_ID}"
AUTH_HDR=(-H "Authorization: Bearer ${CLOUDFLARE_API_TOKEN}" -H "Content-Type: application/json")

api() {
  local method="$1" path="$2"; shift 2
  curl -sS "${AUTH_HDR[@]}" -X "$method" "${API}${path}" "$@"
}

echo "→ list '${LIST_NAME}'"
LISTS="$(api GET "/rules/lists?kind=redirect")"
LIST_ID="$(LIST_NAME="$LIST_NAME" python3 -c '
import json,os,sys
d=json.load(sys.stdin)
assert d.get("success"), d
name=os.environ["LIST_NAME"]
for it in d.get("result") or []:
    if it.get("name")==name:
        print(it["id"]); break
' <<<"$LISTS")"

if [[ -z "$LIST_ID" ]]; then
  CREATE="$(api POST "/rules/lists" --data "{\"name\":\"${LIST_NAME}\",\"description\":\"GSC-hit retired /teach/universities shells\",\"kind\":\"redirect\"}")"
  LIST_ID="$(python3 -c 'import json,sys; d=json.load(sys.stdin); assert d.get("success"), d; print(d["result"]["id"])' <<<"$CREATE")"
  echo "  created $LIST_ID"
else
  echo "  exists $LIST_ID"
fi

wait_op() {
  local op="$1"
  for _ in $(seq 1 60); do
    st="$(api GET "/rules/lists/bulk_operations/${op}")"
    status="$(python3 -c 'import json,sys; d=json.load(sys.stdin); print((d.get("result") or {}).get("status",""))' <<<"$st")"
    case "$status" in
      completed) return 0 ;;
      failed|canceled) echo "$st" >&2; return 1 ;;
    esac
    sleep 2
  done
  echo "timeout waiting for op $op" >&2
  return 1
}

echo "→ clear existing items (if any)"
# Fetch ids then delete in batches
python3 - "$ACCOUNT_ID" "$LIST_ID" "$CLOUDFLARE_API_TOKEN" <<'PY' || true
import json, sys, urllib.request
account, list_id, token = sys.argv[1:4]
base = f"https://api.cloudflare.com/client/v4/accounts/{account}/rules/lists/{list_id}/items"
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

def get(url):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)

def delete(ids):
    if not ids: return
    body = json.dumps({"items": [{"id": i} for i in ids]}).encode()
    req = urllib.request.Request(base, data=body, headers=headers, method="DELETE")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)

ids=[]
url=base+"?per_page=500"
while url:
    d=get(url)
    assert d.get("success"), d
    ids.extend(x["id"] for x in (d.get("result") or []) if "id" in x)
    curs=(d.get("result_info") or {}).get("cursors") or {}
    after=curs.get("after")
    url=(base+f"?per_page=500&cursor={after}") if after else None
print(f"found {len(ids)} existing items", flush=True)
for i in range(0, len(ids), 500):
    resp=delete(ids[i:i+500])
    op=(resp.get("result") or {}).get("operation_id")
    if op:
        # best-effort wait
        import time
        for _ in range(60):
            st=get(f"https://api.cloudflare.com/client/v4/accounts/{account}/rules/lists/bulk_operations/{op}")
            if (st.get("result") or {}).get("status") in ("completed","failed"):
                break
            time.sleep(1)
print("cleared", flush=True)
PY

echo "→ upload $(wc -l < "$CSV" | tr -d ' ') redirects from $CSV"
python3 - "$ACCOUNT_ID" "$LIST_ID" "$CLOUDFLARE_API_TOKEN" "$CSV" <<'PY'
import csv, json, sys, time, urllib.request
from pathlib import Path
account, list_id, token, csv_path = sys.argv[1:5]
base = f"https://api.cloudflare.com/client/v4/accounts/{account}/rules/lists/{list_id}/items"
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

def post(items):
    body = json.dumps(items).encode()
    req = urllib.request.Request(base, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)

def wait(op):
    url=f"https://api.cloudflare.com/client/v4/accounts/{account}/rules/lists/bulk_operations/{op}"
    for _ in range(90):
        req=urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as r:
            d=json.load(r)
        st=(d.get("result") or {}).get("status")
        if st=="completed": return
        if st in ("failed","canceled"):
            raise SystemExit(d)
        time.sleep(2)
    raise SystemExit(f"timeout {op}")

items=[]
for line in Path(csv_path).read_text().splitlines():
    line=line.strip()
    if not line: continue
    parts=next(csv.reader([line]))
    src, tgt = parts[0], parts[1]
    status=int(parts[2]) if len(parts)>2 and parts[2] else 301
    items.append({"redirect": {"source_url": src, "target_url": tgt, "status_code": status}})

for i in range(0, len(items), 1000):
    chunk=items[i:i+1000]
    resp=post(chunk)
    assert resp.get("success"), resp
    op=(resp.get("result") or {}).get("operation_id")
    if op: wait(op)
    print(f"  uploaded {i+len(chunk)}/{len(items)}", flush=True)
PY

echo "→ ensure Bulk Redirect Rule"
ENTRY="$(api GET "/rulesets/phases/http_request_redirect/entrypoint" || echo '{}')"
python3 - "$ENTRY" "$ACCOUNT_ID" "$LIST_NAME" "$CLOUDFLARE_API_TOKEN" <<'PY'
import json, sys, urllib.request
entry_raw, account, list_name, token = sys.argv[1:5]
headers={"Authorization":f"Bearer {token}","Content-Type":"application/json"}

def call(method, url, body=None):
    data=None if body is None else json.dumps(body).encode()
    req=urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)

try:
    entry=json.loads(entry_raw)
except json.JSONDecodeError:
    entry={"success": False}

rule={
  "ref": "teach_retired_uni_shells",
  "expression": f"http.request.full_uri in ${list_name}",
  "description": "Teach retired uni shells → /teach/universities/",
  "action": "redirect",
  "action_parameters": {"from_list": {"name": list_name, "key": "http.request.full_uri"}},
  "enabled": True,
}

if entry.get("success") and entry.get("result",{}).get("id"):
    ruleset_id=entry["result"]["id"]
    rules=[r for r in (entry["result"].get("rules") or []) if r.get("ref")!="teach_retired_uni_shells" and r.get("description")!=rule["description"]]
    rules.append(rule)
    resp=call("PUT", f"https://api.cloudflare.com/client/v4/accounts/{account}/rulesets/{ruleset_id}", {
        "rules": rules
    })
else:
    resp=call("POST", f"https://api.cloudflare.com/client/v4/accounts/{account}/rulesets", {
        "name": "Bulk Redirects",
        "kind": "root",
        "phase": "http_request_redirect",
        "rules": [rule],
    })
assert resp.get("success"), resp
print("rule ok")
PY

echo
echo "Done. Test: https://zlu.me/teach/universities/st-cloud-technical-and-community-college/preview/"
