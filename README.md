# Virtual Drums #
Have you ever wanted to play the drums? Is a drum set too bulky to fit in your home or maybe just too expensive to buy? Well this repository contains the solution to these problems. Using just your camera and some colored drumsticks you can play the virtual drums in your own home. We will walk you through the process of playing virtual drums with your computer through use of python coding and OpenCV. 

Project Video Below:

[![Link to Project Video](https://img.youtube.com/vi/1mVam41cNMA/0.jpg)](https://www.youtube.com/watch?v=1mVam41cNMA)

## Software Requirements ##

To get started make sure you have downloaded OpenCV. (OpenCV 4.5.1 was used in this project) 

Python 3.6 or newer is also required for this project. 

Now you can clone the repository onto your local computer. 

We have supplied the necessary python packages in the `requirements.txt` file that can be installed using the command: 

    pip install -r requirements.txt

## Hardware Required ##
You will need two different sets of colored "drumsticks" for full functionality of the program. One set needs to be red, and the other green, preferably as pure of hue as possible, not off shades if possible. We just used red and green expo markers in our case. We also suggest not having any red or green objects in your background and try to have a shirt that will not be picked up in the mask. Good, even white lighting also suggested. Since everyone may have different hardware and lighting, you may need to change the following values:

    # Green threshold (might need to widen for universal use)
    lower_green = np.array([70, 80, 80])
    upper_green = np.array([90, 255, 255])
    
    # red threshold (might need to widen for universal use)
    lower_red = np.array([160, 100, 100])
    upper_red = np.array([179, 255, 255])

    # Threshold for setting off sound
    pixelThresh = 300
    movementThresh = 10000
    movementMax = 100000

## Project Startup ##

The virtual drums uses computer vision to find and follow the drumsticks from frame to frame of the webcams video. Using some of OpenCV's methods we have trained the camera to look for colored drumsticks that are red and green. The program begins by starting your webcam.

    camera = cv2.VideoCapture(0)
The camera then takes the first frame and initializes the prev_frame variable to be used in the optical flow analysis. The frame needs to be flipped so that it is like looking in the mirror for the user.
    
    _, prev_frame = camera.read()
    prev_frame = cv2.flip(prev_frame, 1)

The height and width of the frame is saved for future use.

    h, w = prev_frame.shape[:2]

After the first frame is processed, we create the upper and lower bounds of the desired drumstick color (red and green) in HSV format. Then the initial frame we captured is masked, defaulted to red, to eliminate all other objects that are not within the specified color thresholg. The code that accomplishes this process is as follows:

    # Green threshold (might need to widen for universal use)
    lower_green = np.array([70, 80, 80])
    upper_green = np.array([90, 255, 255])

    # red threshold (might need to widen for universal use)
    lower_red = np.array([160, 100, 100])
    upper_red = np.array([179, 255, 255])

    # Masks out all colors except ones in the range 
    old_mask = cv2.inRange(prev_frame, lower_red, upper_red)


## Audio Logistics ##
Using the pygame library, we import our sounds into the mixer object which will allow us to play them when we detect a drumstick in the designated area. It will also allow us to adjust the volume.

    # Initializing sounds
    mixer.init()
    soundListRed = [mixer.Sound('./sounds/hihat-acoustic01.wav'), mixer.Sound('./sounds/tom-acoustic01.wav'), mixer.Sound('./sounds/snare-dist01.wav'), mixer.Sound('./sounds/kick-electro01.wav')]
    soundListGreen = [mixer.Sound('./sounds/hihat-acoustic02.wav'), mixer.Sound('./sounds/tom-acoustic02.wav'), mixer.Sound('./sounds/snare-dist03.wav'), mixer.Sound('./sounds/kick-electro02.wav')]

We have two different sound banks for the different color drum sticks and are ordered in relation to placement of the instrument on the screen.

## hit() Function ## 
To use the velocities found from the optical flow the hit function was developed to play the sound. The hit function determines when to play a note of the drum, and what volume to output. First, the movement min and max thresholds are set to determine the volume. After the function receives what sound has been selected, it checks if the user’s drumstick movement is above the threshold, if it is, the function calls the sound according to the speed.
    
    # Threshold for setting off sound
    movementThresh = 10000
    movementMax = 100000
 
    # Method to play sound with adjusting volume
    def hit(sect, mag, list):
        global inBox
        sound = list[sect]
        magSum = np.sum(mag)
        if magSum > movementThresh:
            sound.set_volume(magSum / movementMax)
            sound.play()
        inBox = False


## Color Processing and Detection ##
Now hat we have everything initialized, we will enter our infinite while loop that can be exited by hitting 'q' on your keyboard.

    while 1:

        ...


        # Q key to quit
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break

In the loop we start off by reading in the next frame of the webcam and getting the masks of the red and green for our drumsticks like before.

    _, frame = camera.read()
    frame = cv2.flip(frame, 1)

    # Masks out all colors except ones in the range depending if the current tracking color is red or green.
    if isRed:
        mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), lower_red, upper_red)
    else:
        mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), lower_green, upper_green)

This masking creates a frame of the same size as the original frame, but now only displaying pixels within the passed threshold as white. An image of the masked drumsticks is shown below:

![Screenshot](./images/mask.PNG)
## Drumstick Switching ##
In order to switch which color of drumstick you want to use, we implemented a detection box which you can put your drumstick into and it will switch the tracking color. You must play a note in order to switch again. The box also sets the sound bank and color indicator accordingly.

    inBox = False
    isRed = True

    ...

    if np.sum(mask[0:w // 6, 0:w // 6]) > pixelThresh and not inBox:
        if isRed:
            isRed = False
            color = (0, 255, 0)  #BGR Format Green
            soundList = soundListGreen
        else:
            isRed = True
            color = (0, 0, 255)  #BGR Format Red
            soundList = soundListRed
        inBox = True

## Optical Flow ##

Now that the drumsticks are being tracked and masked we need to know how hard we are swinging them to hit the drums. This requires the use of optical flow to find the velocities of the drumsticks as they hit the target. In this program we used Gunnar Farneback’s algorithm to find the velocities of the drumsticks in the allocated regions. The velocities are then converted from cartesian to polar to make better use of the magnitudes.
The Farneback Algorithm is used instead of the other options was that most other options need corners to detect and those corners to be seen continuously which is hard given the speed and translation of the object tracking. The sticks will move positions nad not have optimal tacking, but with the Farneback, it is a whole body track.

    # optical flow of the drumsticks
       flow = cv2.calcOpticalFlowFarneback(old_mask, mask, flow, .5, 3, 15, 3, 5, 1.2, 0)
       mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
 
From this we know the intensity that we hit the drums with and can determine the sound level from the velocities.


## Drum Image Subprogram ##
 
The Drum Image subprogram is called right before the camera frame is displayed to display four drum images in their respective locations.

Before this can be called, the images need loaded in the main program with:
 
    images = [cv2.imread('images/hi_hat.jpg'),cv2.imread('images/kick_drum.jpeg'), cv2.imread('images/snare.jpeg'), cv2.imread('images/bongo.jpg')]


When the subprogram is called, it creates four image locations in a row along the bottom of the frame. Then, it resizes the image to fit the area and makes a region of interest for each image..

    #Creates the location for the drums
    row_loc = [330,330,330,330]
    col_loc = [0,160,320,480]
    i = 0

    # Resizes the drums to fit into each location on the video
    for ii in images:
        new_width =150
        new_height=150
        drum_size = (new_width, new_height)
        img2 = cv2.resize(ii, drum_size, interpolation = cv2.INTER_AREA)

    # Create an ROI in the location that the image is wanted.
        rows,cols,channels = img2.shape
        roi = frame[row_loc[i]:row_loc[i]+rows, col_loc[i]:col_loc[i]+cols ]

Each Image uses a mask to remove the background color. This is done by making an inverse of the image, so the white background appears black. The black pixels are determined by a threshold, then these pixel locations are removed from the image. Lastly, the image is placed in the region of interest. The subprogram runs through this process 4 times, then returns to the main program to show the final frame to the user.

## Drum Placement ##
Now that the user knows where the drums are going to be placed we need the computer to know where they are as well. To place the drums we needed first create bounds for them as shown in this section:
   
    # Get h and w for bounding box calc
    h, w = prev_frame.shape[:2]
 
Which just pulls the shape of the frame and sets them as the height and width. The height and width for each of the individual drums is specified by dividing the bottom of the image into 4 zones each of which are one different drum noise.(Note it is programmed to detect both red and green drumsticks and each colored stick plays a different noise.)

    # Mask zones
    bottom_right = mask[h - h // 3:h, w - w // 4:w]
    bottom_left = mask[h - h // 3:h, 0:w // 4]
    bottom_left_middle = mask[h - h // 3:h, w // 4:w // 2]
    bottom_right_middle = mask[h - h // 3:h, w // 2:w // 2 + w // 4]
 
Then when the drumstick enters the mask zone the hit function is activated for the drum sound of that section only once, and it will allow you to play the sound again once you leave the area.

        if section1 > pixelThresh:
        if not inSection[0]:
            hit(0, mag[h - h // 3:h, 0:w // 4], soundList)
            inSection[0] = True
        else:
            inSection[0] = False
        if section2 > pixelThresh:
            if not inSection[1]:
                hit(1, mag[h - h // 3:h, w // 4:w // 2], soundList)
                inSection[1] = True
        else:
            inSection[1] = False
        if section3 > pixelThresh:
            if not inSection[2]:
                hit(2, mag[h - h // 3:h, w // 2:w // 2 + w // 4], soundList)
                inSection[2] = True
        else:
            inSection[2] = False
        if section4 > pixelThresh:
            if not inSection[3]:
                hit(3, mag[h - h // 3:h, w - w // 4:w], soundList)
                inSection[3] = True
        else:
            inSection[3] = False
 
To determine whether the drumstick has hit the mask zone we simply count the number of pixels that aren’t zero in the zone for the mask.

    # Counts pixels in certain zone to see if drumstick inside
    section1 = np.count_nonzero(bottom_left)
    section2 = np.count_nonzero(bottom_left_middle)
    section3 = np.count_nonzero(bottom_right_middle)
    section4 = np.count_nonzero(bottom_right)

 
If the threshold is passed a sound is played. 

Now our drums are complete and we can play the drums from our computer.

## Results ##
We successfully implemented a virtual drumkit using python and opencv! Given you adjust the parameters to your environment, the drums work pretty seamlessly!

## Discussion ##
The biggest problems were tuning the color mask correctly. It can be very tedious to find what will fully encapsulate the drumsticks and nothing else. Another difficulty we came across was a lot of background noise in the optical flow method, so we cut out a lot of the noise and did the optical flow only on the masked drumsticks.
The next steps to the project would be optimizations in detection and flow. Right now it isn't too laggy but can get delayed a little if a lot is happening all at once. Possibly even try and get better frame rate to track the drum sticks better in the optical flow.
We could have taken the approach of object detection of certain drumsticks, but we chose to do color detection due to the universality of it. Anyone can use it if the have a red or green object, and it allows for us to change the sound bank of each color. It was also beneficial to get drumsticks that are very close to the fully saturated hue because it's easier to differentiate among other objects that may be in the background; the tighter the threshold, the better.

## Resource Links ##
Drum Samples:
https://99sounds.org/drum-samples/

Bongo Drum:
https://www.escribircanciones.com.ar/instrumentos-musicales/bongo.html

Hi-hat:
https://www.musicinfo.pl/sabian-b8x-14--hi-hat,2791601,p,2,51,156,0,0.html

Kick drum:
https://www.ebay.com/itm/DW-Performance-Series-Bass-Drum-14-x-24-Black-Diamond-FinishPly/264533277170?_trkparms=aid%3D1110006%26algo%3DHOMESPLICE.SIM%26ao%3D1%26asc%3D20200818143230%26meid%3De6f58c7b8b71473293ac2722075c3f37%26pid%3D101224%26rk%3D1%26rkt%3D5%26mehot%3Dnone%26sd%3D124573040859%26itm%3D264533277170%26pmt%3D0%26noa%3D1%26pg%3D2047675%26algv%3DDefaultOrganic%26brand%3DDW&_trksid=p2047675.c101224.m-1

Snare drum:
https://www.guitarcenter.com/Pearl/Symphonic-Snare-Drum.gc

Timpani:
http://clipart-library.com/timpani-cliparts.html
