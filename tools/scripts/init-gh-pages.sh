#!/usr/bin/env bash
# init-gh-pages.sh — one-time bootstrap of the orphan `gh-pages` branch
# that backs the bench-history publication site (audit-R10 T3.5).
#
# Run this from a clean working tree on `main`. It creates an orphan
# `gh-pages` branch with a placeholder index page + `.nojekyll` marker,
# pushes it to `origin`, and switches you back to `main`.
#
# After running:
#   1. In repo settings -> Pages: source = `gh-pages` branch, root path.
#   2. Wait for the first `bench-history.yml` run on a main push (or
#      trigger it manually via `gh workflow run bench-history.yml`) to
#      populate the actual archive + rendered site.
#
# Idempotent: if the branch already exists locally or remotely, the
# script bails out with a clear message rather than overwriting.

set -euo pipefail

if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
  echo "error: not inside a git working tree" >&2
  exit 1
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "error: working tree has uncommitted changes; commit or stash first" >&2
  exit 1
fi

current_branch=$(git symbolic-ref --short HEAD)

if git rev-parse --verify --quiet gh-pages > /dev/null; then
  echo "error: local 'gh-pages' branch already exists" >&2
  echo "       delete it first if you want to re-bootstrap: git branch -D gh-pages" >&2
  exit 1
fi

if git ls-remote --exit-code --heads origin gh-pages > /dev/null 2>&1; then
  echo "error: remote 'origin/gh-pages' branch already exists" >&2
  echo "       fetch it instead: git fetch origin gh-pages:gh-pages" >&2
  exit 1
fi

echo "==> creating orphan gh-pages branch"
git checkout --orphan gh-pages

echo "==> wiping working tree (orphan branches inherit it)"
git rm -rf --quiet . > /dev/null 2>&1 || true
# `git rm -rf .` does not touch ignored files; clean those too so the
# initial commit only contains what we explicitly add below.
git clean -fdx --quiet > /dev/null 2>&1 || true

echo "==> seeding placeholder index + .nojekyll"
touch .nojekyll
cat > index.html <<'EOF'
<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>tokenfs-algos bench history (initializing)</title>
<style>body{font-family:sans-serif;margin:2rem auto;max-width:48rem;padding:0 1rem;color:#222}</style>
</head><body>
<h1>tokenfs-algos bench history</h1>
<p>This page will be replaced by an auto-generated bench-history index after
the first successful <code>bench-history.yml</code> workflow run on
<code>main</code>.</p>
<p>If you are reading this hours after a green main push, check the
<a href="https://github.com/mjbommar/tokenfs-algos/actions/workflows/bench-history.yml">workflow runs</a>.</p>
</body></html>
EOF

git add .nojekyll index.html
git commit --quiet -m "init gh-pages branch (placeholder; replaced by bench-history.yml)"

echo "==> pushing gh-pages to origin"
git push -u origin gh-pages

echo "==> returning to ${current_branch}"
git checkout --quiet "${current_branch}"

cat <<EOF

==> done.

Next steps:
  1. Open https://github.com/mjbommar/tokenfs-algos/settings/pages
     and set source = 'gh-pages' branch, root path '/'.
  2. Trigger the publish workflow manually to verify the flow:
       gh workflow run bench-history.yml
  3. After it succeeds, browse https://mjbommar.github.io/tokenfs-algos/

The publish workflow will then run automatically after every successful
bench-regression.yml on main.
EOF
