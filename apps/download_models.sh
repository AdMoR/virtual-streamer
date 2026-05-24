#!/usr/bin/env bash

set -ex

wget -c -O checkpoints/mobilenet.pth 'https://github.com/elliottzheng/face-detection/releases/download/0.0.1/mobilenet0.25_Final.pth'
wget -c -O checkpoints/resnet50.pth 'https://github.com/elliottzheng/face-detection/releases/download/0.0.1/Resnet50_Final.pth'
