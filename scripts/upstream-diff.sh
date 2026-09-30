#!/usr/bin/env bash
# Shows what changed upstream since the pinned commit, grouped by the C# file
# that has to follow. Run before an update; nothing is modified.
#
#   scripts/upstream-diff.sh rade_c <new-ref>   (checkout at ../rade_c)
#   scripts/upstream-diff.sh opus   <new-sha>   (clone at $OPUS_DIR, default ../opus)
#   OLD=<ref> overrides the pinned starting point (e.g. to preview history).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"
which="${1:?rade_c|opus}"; new="${2:?new ref}"
pin() { sed -n "s/^set(${1} \"\([0-9a-f]*\)\").*/\1/p" "$REPO/reference/pins.cmake"; }
case "$which" in
    rade_c) dir="$(dirname "$REPO")/rade_c"; old="${OLD:-$(pin RADE_C_SHA)}" ;;
    opus)   dir="${OPUS_DIR:-$(dirname "$REPO")/opus}"; old="${OLD:-$(pin OPUS_SHA)}" ;;
    *) echo "unknown upstream $which" >&2; exit 1 ;;
esac
git -C "$dir" fetch --quiet || true
echo "== $which $old..$new"
git -C "$dir" log --oneline "$old..$new" | head -50
echo
changed="$(git -C "$dir" diff --name-only "$old" "$new")"
[ -z "$changed" ] && { echo "no file changes"; exit 0; }
echo "== changed upstream files -> C# targets (kind)"
while read -r f; do
    hit="$(awk -F'\t' -v u="$which" -v f="$f" '
        $1==u { pat=$2; gsub(/\./,"\\.",pat); gsub(/\*/,".*",pat); gsub(/\[ch\]/,"[ch]",pat);
                n=split(pat, alts, /, /); for (i=1;i<=n;i++) if (f ~ "^" alts[i] "$") { print $3 " (" $4 ")"; exit } }' \
        "$REPO/scripts/port-map.tsv")"
    printf '  %-40s %s\n' "$f" "${hit:-UNMAPPED -- decide: port, gen, ref or skip, then add to port-map.tsv}"
done <<< "$changed"
echo
echo "== diffstat"
git -C "$dir" diff --stat "$old" "$new" | tail -40
