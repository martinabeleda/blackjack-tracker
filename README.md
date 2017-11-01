# Blackjack Tracker
Blackjack game tracker using OpenCV and Python

## Download
Clone the repository: 
```
git clone https://github.com/martinabeleda/Blackjack-Tracker
```
or download zip:
```
https://github.com/martinabeleda/Blackjack-Tracker/archive/master.zip
```
## Usage instruction

### Create a symbolic link to OpenCV on your machine
Note 1 : Skip this section if you've already created this symbolic link

Note 2: You need to have OpenCV compiled on your machine for python 3. If you haven't, I would recommend this link:
https://www.learnopencv.com/install-opencv3-on-ubuntu/

Navigate to the top directory of the project:
```
cd Blackjack-Tracker
```

Find the location of OpenCV python binary (cv2.so) on your machine
```
find /usr/local/lib/ -type f -name "cv2*.so"
```

You should see something like the following (copy this path). 
```
/usr/local/lib/python3.5/dist-packages/cv2.cpython-35m-x86_64-linux-gnu.so
```

Forcefully remove all symbolic link to OpenCV in the current directory to avoid conflict. They are likely linked from other (my) machine. If rm indicates no such file present, just move on.
```
rm -f cv2*.so
```

Create a symbolic link to OpenCV on your local machine
```
ln -s "the file path copied above"
```

### Run
Plug in the external webcam, wait for about 30 seconds for the device to be recognized.

Navigate to the top directory of the project:
```
cd Blackjack-Tracker
```

Work on the virtual environment included in this folder.

To activate:
```
source blackjack_env/bin/activate
```
You should now see (blackjack_env) at the front of your command line.

Run the program with optional argument to indicate which camera to use (default: 0). Index 0 is usually the webcam on your laptop, and index 1 is an external webcam.
```
python3 main.py
python3 main.py -camera 1
```

To deactivate:
```
deactivate
```

## About the program:
* This program only works on Python 3
* OpenCV version 3.0 and above is recommended.
* This program only works in the virtal environment included. Alternatively, if you wish to run them in your local virtual environment, you can install the requirements included:
```
pip3 install -r requirements.txt
```
* Program starts in the calibration stage with 100 seconds countdown 
* Press 'a' if you're happy with the transformation found
* Program enters card recognition state.
* Press 't' to toggle between gesture state and card recognition state.
* Press 'c' to recalibrate and 'a' to accept the calibration
* Press 'q' to quit the program

## Live demo
<a href="http://www.youtube.com/watch?feature=player_embedded&v=uAGf70MoNyM
" target="_blank"><img src="http://img.youtube.com/vi/uAGf70MoNyM/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="480" height="360" border="10" /></a>

## Software Modules
* Playing surface detection and transformation
* Card recognition
* Coin detection
* Gesture Recognition
