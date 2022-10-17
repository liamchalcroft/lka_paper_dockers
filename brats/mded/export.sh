#!/usr/bin/env bash

./build.sh

docker save ploras | gzip -c > ploras.tar.gz
