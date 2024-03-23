import cv2
import numpy as np
from pygame import mixer
from image_paste import drum_image

images = [cv2.imread('images/hi_hat.jpg'), cv2.imread('images/bongo.jpg'), cv2.imread('images/snare.jpeg'), cv2.imread('images/kick_drum.jpeg')]


# Threshold for setting off sound
pixelThresh = 300
movementThresh = 10000
movementMax = 100000


# Method to play sound with adjusting volume
def hit(sect, mag, list):
    global inBox
    sound = list[sect]
    magSum = np.sum(mag)
    # print(magSum)
    if magSum > movementThresh:
        sound.set_volume(magSum / movementMax)
        sound.play()
    inBox = False


# Activating camera, CAP_DSHOW for HD webcams
#camera = cv2.VideoCapture(0)
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Initialize prev frame for optical flow
_, prev_frame = camera.read()
prev_frame = cv2.flip(prev_frame, 1)

# Initializing sounds
mixer.init()

soundListRed = [mixer.Sound('./sounds/hihat-acoustic01.wav'), mixer.Sound('./sounds/tom-acoustic01.wav'), mixer.Sound('./sounds/snare-dist01.wav'), mixer.Sound('./sounds/kick-electro01.wav')]
soundListGreen = [mixer.Sound('./sounds/hihat-acoustic02.wav'), mixer.Sound('./sounds/tom-acoustic02.wav'), mixer.Sound('./sounds/snare-dist03.wav'), mixer.Sound('./sounds/kick-electro02.wav')]

# Get h and w for bounding box calc
h, w = prev_frame.shape[:2]

# Green threshold (might need to widen for universal use)
lower_green = np.array([70, 80, 80])
upper_green = np.array([90, 255, 255])

# red threshold (might need to widen for universal use)
lower_red = np.array([160, 100, 100])
upper_red = np.array([179, 255, 255])

old_mask = cv2.inRange(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV), lower_red, upper_red)

flow = None
soundList = soundListRed
isRed = True
inBox = False
color = (0, 0, 255)
inSection = [False, False, False, False]

# Continuous Webcam stream
while 1:
    # Take each frame
    _, frame = camera.read()
    frame = cv2.flip(frame, 1)

    if isRed:
        mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), lower_red, upper_red)
    else:
        mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), lower_green, upper_green)

    # Checks to see if switch drum stick color
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

    # Optical flow for mask
    flow = cv2.calcOpticalFlowFarneback(old_mask, mask, flow, .5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Mask zones
    bottom_right = mask[h - h // 3:h, w - w // 4:w]
    bottom_left = mask[h - h // 3:h, 0:w // 4]
    bottom_left_middle = mask[h - h // 3:h, w // 4:w // 2]
    bottom_right_middle = mask[h - h // 3:h, w // 2:w // 2 + w // 4]

    # Counts pixels in certain red zone to see if drumstick inside
    section1 = np.count_nonzero(bottom_left)
    section2 = np.count_nonzero(bottom_left_middle)
    section3 = np.count_nonzero(bottom_right_middle)
    section4 = np.count_nonzero(bottom_right)

    # Pixel zone detection
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

    old = mask
    # Draws bounding box in top left to switch color
    cv2.rectangle(frame, (0, 0), (w//6, w//6), color, 3)
    cv2.putText(frame, 'Switch', (3, w//11), cv2.QT_FONT_NORMAL, 1, (0, 0, 0), 2)
    drum_image(images, frame)


    cv2.imshow('frame', frame)

    # Uncomment if you want to see the mask
    # cv2.imshow('mask', mask)

    # Q key to quit
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break



camera.release()
cv2.destroyAllWindows()
