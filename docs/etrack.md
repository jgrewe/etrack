# E-Fish tracking

Tool for easier handling of tracking results.

## Installation

### 1. Clone git repository

```shell
git clone https://whale.am28.uni-tuebingen.de/git/jgrewe/efish_tracking.git
```

### 2. Change into directory

```shell
cd efish_tracking
````

### 3. Install with pip

```shell
pip3 install -e . --user
```

The ```-e``` installs the package in an *editable* model that you do not need to reinstall whenever you pull upstream changes.

If  you leave away the ```--user``` the package will be installed system-wide.

## TrackingResults

Is a class that wraps around the *.h5 files written by DeepLabCut

## ImageMarker

Class that allows for creating MarkerTasks to get specific positions in a video.