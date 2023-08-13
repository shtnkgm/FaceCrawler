# 😉FaceCrawler
Python script to crawl people's images and crop face areas

## Setup

```bash
# Clone Repository
git clone https://github.com/shtnkgm/FaceCrawler.git

# Install pip
curl -kL https://bootstrap.pypa.io/get-pip.py | python3

sudo pip3 install icrawler
sudo pip3 install matplotlib
sudo pip3 install opencv-python
sudo pip3 install beautifulsoup4
sudo pip3 install lxml

# Download OpenCV Cascade File
cd FaceCrawler
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml
```

## Usage

```bash
# Fetch 100 Images for keyword person_name1, person_name2, person_name3 ...
python3 crawler.py 100 person_name1 person_name2 person_name3 ...
```

## Requirements
 - Python3
