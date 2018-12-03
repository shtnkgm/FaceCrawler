# FaceCrawler
Crawl Google Images, Fetch Images, Crop Face Image

## Setup

```bash
# Install pip
curl -kL https://bootstrap.pypa.io/get-pip.py | python3

pip install icrawler
pip install opencv-python
pip install matplotlib

# Download OpenCV Cascade File
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml
```

## Usage

```bash
# Fetch 100 Images for keyword 'person_name'
python3 crawler.py person_name 100
```

## Requirements
 - Python3
