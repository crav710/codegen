#!/usr/bin/env bash
# One-time: create crav710/codegen on GitHub and push mentor-extension + main.
set -euo pipefail

REPO="crav710/codegen"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Checking GitHub CLI auth..."
if ! gh auth status >/dev/null 2>&1; then
  echo "Not logged into gh. Starting login (follow browser prompts)..."
  gh auth login --git-protocol ssh --hostname github.com --web
fi

echo "==> Creating repo ${REPO} (skip if exists)..."
gh repo view "${REPO}" >/dev/null 2>&1 || \
  gh repo create "${REPO}" --public --source=. --remote=mygithub --description "AIML PGCP CodeGen capstone"

git remote remove mygithub 2>/dev/null || true
git remote add mygithub "git@github.com:${REPO}.git"

echo "==> Pushing branches..."
git push -u mygithub mentor-extension
git push mygithub main || true

echo ""
echo "Done. Colab will clone:"
echo "  https://github.com/${REPO}.git"
echo "  branch: mentor-extension"
