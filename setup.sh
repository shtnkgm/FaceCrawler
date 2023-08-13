#!/bin/sh

echo "🏃install pip"
curl -kL https://bootstrap.pypa.io/get-pip.py | python3

echo "🏃install python dependencies"
sudo pip3 install -r requirements.txt

echo "🏃download OpenCV Cascade File"
wget -nc https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml

echo "✅Finished!"