#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Set default container name
DOCKER_IMAGE_TAG="reg2026_algorithm"

# Identifiable version tag = git branch name so the saved image +
# model tarballs are distinguishable on disk. GC doesn't care about the upload filename;
# this is purely for local sanity (don't mix up which version you're uploading).
VERSION_TAG=$( git -C "${SCRIPT_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null | tr '/' '-' )
[ -z "$VERSION_TAG" ] && VERSION_TAG="nogit"

echo ""
echo "= STEP 1 = (Re)build the image"
export DOCKER_QUIET_BUILD=1
source "${SCRIPT_DIR}/do_build.sh"
echo "==== Done"
echo ""

# Get the build information from the Docker image tag
build_timestamp=$( docker inspect --format='{{ .Created }}' "$DOCKER_IMAGE_TAG" )

if [ -z "$build_timestamp" ]; then
    echo "Error: Failed to retrieve build information for container $DOCKER_IMAGE_TAG"
    exit 1
fi

# Format the build timestamp as YYYY-MM-DD_HHMMSS. Drop the fractional seconds and the timezone:
# docker's ".Created" is e.g. 2026-07-08T21:04:46.283329567+08:00 (offset, not "Z"), which the old
# Z-anchored regex failed to match - leaving the raw nanosecond+timezone string in the filename.
formatted_build_info=$(echo "$build_timestamp" | sed -E 's/T([0-9]{2}):([0-9]{2}):([0-9]{2}).*$/_\1\2\3/')

# Set the output filename with timestamp and build information
output_filename="${DOCKER_IMAGE_TAG}_${VERSION_TAG}_${formatted_build_info}.tar.gz"
output_path="${SCRIPT_DIR}/$output_filename"

# Save the Docker-container image and gzip it
echo "= STEP 2 = Saving the image"
echo "This can take a while."

docker save "$DOCKER_IMAGE_TAG" | gzip -c > "$output_path"
printf "Saved as: \e[32m${output_filename}\e[0m\n"

echo "==== Done"
echo ""


# Create the tarbal
echo "= STEP 3 = Packing the model"
echo "This can take a while."
output_tarball_name="${SCRIPT_DIR}/SIGMR_${VERSION_TAG}_${formatted_build_info}.tar.gz"

# Guard: Grand Challenge's model-tarball extraction REJECTS symlinks ("Tarfile could
# not be extracted"). HF caches store snapshots/<rev>/* as symlinks into blobs/ - those
# must be de-referenced into real files before packing. Fail loud rather than ship a
# tarball GC can't open (wastes a debug submission).
if find "${SCRIPT_DIR}/model" -type l | grep -q .; then
    echo "ERROR: symlinks found under model/ - GC cannot extract a tarball with symlinks:" >&2
    find "${SCRIPT_DIR}/model" -type l >&2
    echo "De-reference them (move blob into the snapshot path, rm blobs/) before packing." >&2
    exit 1
fi

tar -czf $output_tarball_name -C "${SCRIPT_DIR}/model" .
printf "Saved as: \e[32m$(basename "$output_tarball_name")\e[0m\n"

echo "==== Done"
echo ""

printf "\e[33mNext steps:\e[0m\n"
printf "  Version: \e[36m%s\e[0m\n" "$VERSION_TAG"
printf "  1. Upload \e[32m%s\e[0m  →  Grand Challenge > Algorithm > Container images\n" "$output_filename"
printf "  2. Upload \e[32m%s\e[0m  →  Grand Challenge > Algorithm > Models\n" "$(basename "$output_tarball_name")"
