#!/usr/bin/env bash
#
# Build and test the site content
#
# Requirement: npm dependencies installed
#
# Usage: See help information

set -eu

help() {
  echo "Build and check the Astro site"
  echo
  echo "Usage:"
  echo
  echo "   bash $0"
  echo
  echo "Options: none"
}

main() {
  npm run check
  npm run build
}

while (($#)); do
  opt="$1"
  case $opt in
  -h | --help)
    help
    exit 0
    ;;
  *)
    # unknown option
    help
    exit 1
    ;;
  esac
done

main
