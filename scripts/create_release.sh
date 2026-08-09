#!/bin/bash
set -e

# To be run AFTER the campaign completes successfully
echo "Tagging the repository for the article release..."
git tag -a v1.0.0 -m "Official release accompanying the manuscript."

echo "Please run:"
echo "git push origin v1.0.0"
echo ""
echo "Then, create a GitHub Release and mint a Zenodo DOI."
