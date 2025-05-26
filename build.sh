#!/usr/bin/env bash
################################################################
# Copyright 2025 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2025-05-26
################################################################

CUR_DIR=$(pwd)
SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
echo "CUR_DIR: $CUR_DIR"
echo "SCRIPT_DIR: $SCRIPT_DIR"

cd $SCRIPT_DIR

rm -rf dist
python3 -m build
pip3 uninstall hex_utils -y
pip3 install dist/hex_utils-*-py3-none-any.whl
rm -rf dist

cd $CUR_DIR
