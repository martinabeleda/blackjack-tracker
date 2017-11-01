# Blackjack Tracker
Blackjack game tracker using OpenCV and Python

## Download
Clone the repository or download zip:
```
git clone https://github.com/martinabeleda/Blackjack-Tracker
```
## Usage instruction
Plug in the external webcam, wait for about 30 seconds for the device to be recognized.

Navigate to the top directory of the project:
```
cd Blackjack-Tracker
```

Work on the virtual environment included in this folder
To activate:
```
source blackjack_env/bin/activate
```
You should now see (blackjack_env) at the front of your command line

To deactivate:
```
Deactivate
```

### Create a symbolic link to OpenCV on your machine
Note 1 : Skip this section if you've done this before

Note 2: You need to have OpenCV compiled on your machine for python 3. If you haven't, I would recommend this link:
https://www.learnopencv.com/install-opencv3-on-ubuntu/

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
ln -s "the file path above"
```

### Run the program
```
python3 main.py
```


## Live demo
<a href="http://www.youtube.com/watch?feature=player_embedded&v=uAGf70MoNyM
" target="_blank"><img src="http://img.youtube.com/vi/uAGf70MoNyM/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="480" height="360" border="10" /></a>

## Software Modules
* Playing surface detection and transformation
* Card recognition
* Coin detection
* Gesture Recognition
