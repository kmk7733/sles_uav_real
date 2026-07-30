#!/bin/bash
# Our code that lives inside third-party checkouts.
#
# Two kinds:
#   overlays/<repo>/...        NEW files we added (git in the parent repo cannot
#                              track files inside a nested repository)
#   overlays/patches/*.patch   MODIFICATIONS to files upstream already tracks
#
#   restore : put both back onto a freshly imported workspace
#   capture : pull the current live state back into this repo (do this after
#             editing the live files, or the change only exists on one machine)
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$(cd "$HERE/.." && pwd)
REL=zed-ros-examples/examples/zed_rtabmap_example
DIRS="scripts launch params rviz"

copy() {  # copy <from-root> <to-root>
    for d in $DIRS; do
        [ -d "$1/$REL/$d" ] || continue
        mkdir -p "$2/$REL/$d"
        rsync -a --exclude __pycache__ "$1/$REL/$d/" "$2/$REL/$d/"
    done
}

case "${1:-}" in
restore)
    copy "$HERE" "$SRC"
    echo "overlay files restored -> $SRC/$REL"
    for p in "$HERE"/patches/*.patch; do
        [ -e "$p" ] || continue
        pkg=$(basename "$p" .patch)
        [ -d "$SRC/$pkg" ] || { echo "  skip $pkg (not checked out)"; continue; }
        if git -C "$SRC/$pkg" apply --check "$p" 2>/dev/null; then
            git -C "$SRC/$pkg" apply "$p"
            echo "  applied $pkg.patch"
        else
            echo "  SKIP $pkg.patch -- does not apply cleanly (already applied, or upstream moved)"
        fi
    done
    ;;
capture)
    copy "$SRC" "$HERE"
    echo "overlay files captured -> $HERE/$REL"
    mkdir -p "$HERE/patches"
    for pkg in vicon_bridge zed-ros-wrapper openmv_cam vrpn_client_ros zed-ros-examples; do
        [ -d "$SRC/$pkg/.git" ] || continue
        out="$HERE/patches/$pkg.patch"
        git -C "$SRC/$pkg" diff > "$out"
        if [ -s "$out" ]; then echo "  captured $pkg.patch"; else rm -f "$out"; fi
    done
    ;;
*)
    echo "usage: $0 {restore|capture}" >&2; exit 1 ;;
esac
