#!/usr/bin/env bash
while true; do
    if [ -f /image.png/ ]; then
        roborunner()
done



function roborunner() {
    python3 colorfind.py
    rm image.png
}
