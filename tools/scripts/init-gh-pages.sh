#!/usr/bin/env bash
# init-gh-pages.sh — one-time bootstrap of the orphan `gh-pages` branch
# that backs the bench-history publication site (audit-R10 T3.5).
#
# Uses `git worktree add --orphan` (git >= 2.42) so the gh-pages branch
# is built in a separate working directory — your main checkout's
# `target/`, `.env`, and other gitignored state stay untouched. No
# destructive `git clean` happens against your primary worktree.
#
# Run from anywhere inside the repo on any branch. Idempotent: refuses
# to proceed if `gh-pages` already exists locally or remotely.
#
# After running:
#   1. In repo settings -> Pages: source = `gh-pages` branch, root path
#      (one-time; GitHub remembers it across pushes).
#   2. Wait for the first `bench-history.yml` run on a main push (or
#      `gh workflow run bench-history.yml`) to populate the actual
#      archive + rendered site.

set -euo pipefail

if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
  echo "error: not inside a git working tree" >&2
  exit 1
fi

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

worktree_dir="${TMPDIR:-/tmp}/tokenfs-pages-init.$$"
trap 'git worktree remove --force "${worktree_dir}" > /dev/null 2>&1 || true' EXIT

echo "==> creating orphan gh-pages worktree at ${worktree_dir}"
git worktree add --orphan -b gh-pages "${worktree_dir}" > /dev/null

echo "==> seeding placeholder index + .nojekyll"
touch "${worktree_dir}/.nojekyll"
cat > "${worktree_dir}/index.html" <<'EOF'
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

echo "==> committing in worktree"
git -C "${worktree_dir}" add -A
git -C "${worktree_dir}" commit --quiet \
  -m "init gh-pages branch (placeholder; replaced by bench-history.yml)"

echo "==> pushing gh-pages to origin"
git -C "${worktree_dir}" push -u origin gh-pages

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

(Worktree at ${worktree_dir} is auto-removed on exit.)
EOF
