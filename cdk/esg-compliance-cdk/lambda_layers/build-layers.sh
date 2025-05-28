#!/bin/bash

export DOCKER_DEFAULT_PLATFORM=linux/amd64

script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$script_dir" || {
    echo "Error: Failed to change to script directory."
    exit 1
}

if [ $# -eq 0 ]; then
    echo "Error: At least one layer name is required."
    exit 1
fi

# Grab folder names of layers
allowed_args=( $(ls -d */) )

# Remove trailing slash
allowed_args=( "${allowed_args[@]%/}" )

for arg in "$@"
do
    if [[ " ${allowed_args[@]} " =~ "$arg" ]]; then
        echo "Valid layer name: $arg"
    else
        echo "Error: Invalid layer name: $arg"
        echo "Allowed layer names: ${allowed_args[@]}"
        exit 1
    fi
done

echo "The layer/s to build are: $*"

echo "----- Starting colima VM -----"
    colima start --arch x86_64 --cpu-type max

for arg in "$@"
do
    export PKG_DIR="$arg/python"
    export DOCKER_BUILD="build"
    echo "----- Clearing previous $arg build -----"
    rm -rf ${PKG_DIR}
    mkdir -p ${PKG_DIR}

    echo "----- Building dependencies -----"
    docker run --platform linux/amd64 --rm -v $(pwd):/build -w /build public.ecr.aws/sam/build-python3.12:latest \
    pip3 install --upgrade -r $arg/requirements.txt -t ${PKG_DIR}

    echo "----- updating build permissions -----"
    find ${PKG_DIR} -type f -exec chmod 644 {} \;
    find ${PKG_DIR} -type d -exec chmod 755 {} \;

    echo "----- $arg layer packaged successfully -----"
done