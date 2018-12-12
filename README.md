# FaceCrawler
Crawl Google Images, Fetch Images, Crop Face Image

## Setup

```bash
# Clone Repository
git clone https://github.com/shtnkgm/FaceCrawler.git

# Install pip
curl -kL https://bootstrap.pypa.io/get-pip.py | python3

sudo pip install icrawler
sudo pip install opencv-python
sudo pip install matplotlib

# Download OpenCV Cascade File
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml
```

## Usage

```bash
# Fetch 100 Images for keyword person_name1, person_name2, person_name3 ...
python3 crawler.py 100 person_name1 person_name2 person_name3 ...
```

## Requirements
 - Python3
