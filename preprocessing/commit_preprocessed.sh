#!/bin/bash
set -e

git config --local user.email "action@github.com"
git config --local user.name "GitHub Action"

git add preprocessing/namadataset_preprocessing/

if git diff --staged --quiet; then
  echo "No changes to commit"
else
  git commit -m "Auto-update: Processed banknote_authentication data [$(date +'%Y-%m-%d %H:%M')]"
  git push origin main
  echo "Changes pushed successfully"
fi