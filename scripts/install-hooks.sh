#!/bin/bash
# Install git hooks from .git/hooks into the actual git directory.
# Run this once after cloning the repository.

REPO="$(cd "$(dirname "$0")/.." && pwd)"
HOOKS_DIR="$REPO/.git/hooks"

if [[ ! -d "$HOOKS_DIR" ]]; then
  echo "❌ .git/hooks not found - are you in a git repository?"
  exit 1
fi

if [[ -f "$HOOKS_DIR/pre-commit" ]]; then
  chmod +x "$HOOKS_DIR/pre-commit"
  echo "✅ pre-commit hook installed and made executable"
else
  echo "⚠️  pre-commit hook template not found in .git/hooks"
  exit 1
fi
