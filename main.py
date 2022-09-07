import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np

# if you don't know what to do with the module, ctrl+click on that, it will open the module and shows its structure,
# from there we can learn how to use it another advantage of it is that it will always show the latest updated code

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(2,720)

imgBackground = cv2.imread("Resources/Background.png")
imgGameover = cv2.imread("Resources/gameOver.png")
# the above two are normal images. we are importing them normally. without any preprocessing.
imgBall = cv2.imread("Resources/Ball.png", cv2.IMREAD_UNCHANGED)
imgBat1 = cv2.imread("Resources/bat1.png", cv2.IMREAD_UNCHANGED)
imgBat2 = cv2.imread("Resources/bat2.png", cv2.IMREAD_UNCHANGED)

# the above 3 images will be used while we're playing our game
# i.e parallely other operations will be processing while these images will be displayed.
# So in order to do that, we're passing the additional parameter name-> "cv2.IMREAD_UNCHANGED",
# what this function does is that it asks the 'cv' to just import the images, have the aplha channel,
# have the transparency so that we can play along with it, otherwise we can't.

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=2)
# HandDetector is our class within which we've passed the below 2
# maxHands – Maximum number of hands to detect
# detectionCon – Minimum Detection Confidence Threshold

# Variables
ballPos = [100, 100]
speedX = 15
speedY = 15
gameOver = False
score = [0, 0]



while(True):
    _, img = cap.read()
    imgRaw = img.copy()
    img = cv2.flip(img, 1)
    # if we remove this line then run it, we'll find that the image of ours is flipped,
    # which will create confusion while playing the game. so this line resolves the problem
    # In this func. we've passed img and `1` this one tells that we've to flip horizontally.
    # similarly `0` for vertically flipping the image

    # Find the hand and its landmarks
    hands, img = detector.findHands(img,flipType=False)
    # after we detect our image, we're passing this
    # Now look, just by adding one line we can detect our hands. This is why we're using 'cvzone' otherwise,
    # we can use mediapipe, but then we have to de-normalize the values, we've to extract all the values one by one.
    # This is a bit lengthy process so instead we save our time using cvzone.
    # fliptype is a flag is set to false otherwise
    # after flipping the image it will show our right hand as left hand and vice-versa.
    # i.e cvzone by-default flip the image but this flag prevents this.

    # Overlaying the background image
    img = cv2.addWeighted(img,0.2, imgBackground,0.8,0)
    # so to overlay the over our image we use this Fn. @L19
    # this func. takes input as source 1, alpha channel, src2, beta,gamma
    # so for the src1, src2 we'll pass the  imported the images.
    # img is passed as the src1 which is basically the image shown by our webcam,
    # we've passed alpha as 0.2, src2 as imgBackground, then 0.8 is passed as beta(it is 1 - alpha value)
    # then 0 is passed as gamma

# ALPHA BLENDING: It is the process of overlaying the overlaying a foreground image with transparency
    # over the background image.
    # When alpha = 0, the output pixel color is the background.
    # When \alpha = 1, the output pixel color is simply the foreground.
    # When 0 < \alpha < 1 the output pixel color is a mix of the background and the foreground.
    # For realistic blending, the boundary of the alpha mask usually has pixels that are between 0 and 1.

    # Check for hands
    if hands:
        for hand in hands:
            x,y,w,h = hand['bbox']
            h1, w1, _ = imgBat1.shape
            y1 = y - h//2
            y1 = np.clip(y1,20,415)
            if hand['type'] == "Left":
                img = cvzone.overlayPNG(img, imgBat1, (59, y1))
                # 59,100 position hai apne bat ki.. uss bracket mein (either side per jo hai)
                # Now to check if the ball is hitting the bat or not (check below statement)
                if 59< ballPos[0]< 59+w1 and y1< ballPos[1]< y1+h1:
                    speedX = -speedX
                    # working for the above condition:
                    # for the left hand the ball position must be between 59< ballPos[0]< 59+w1 and y1< ballPos[1]< y1+h1
                    # in order to check if the ball has been hit by the bat.

                    ballPos[0]+=30 # to show the bouncy effect when the ball hits the bat.
                    score[0] +=1

            if hand['type'] == "Right":
                img = cvzone.overlayPNG(img, imgBat1, (1195, y1))
                if 1149< ballPos[0]< 1195-30 +w1 and y1< ballPos[1]< y1+h1:
                    speedX = -speedX
                    ballPos[0] -= 30 # to show the bouncy effect when the ball hits the bat.
                    score[1] +=1


    # Game Over
    if ballPos[0]<40 or ballPos[0]>1200:
        gameOver = True
    if gameOver:
        img = imgGameover
        cv2.putText(img, str(score[1] + score[0]).zfill(2), (585, 360), cv2.FONT_HERSHEY_COMPLEX,
                    2.5, (200, 0, 200), 5)
    else:

        # Moving the ball

        if ballPos[1] >=500 or ballPos[1] <=10:
            speedY = -speedY
        # so in order to bounce the ball within the court we've to set the boundaries. so the above function is doing that
        ballPos[0] += speedX
        # This determines the speed of the in vertical direction
        ballPos[1] += speedY
        # This determines the speed of the in horizontal direction


        # Draw the ball
        img = cvzone.overlayPNG(img, imgBall, ballPos)
        # generally this is a long process as we have to overlay 'png' over an 'image' in open cv.
        # We have this function in cvzone which helps us write efficiently.
        # param1 is the image on which we have to overlay, param2 is the image which needs to be overlayed
        # and param3 is the position of the param2.

        cv2.putText(img, str(score[0]), (300, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)
        cv2.putText(img, str(score[1]), (900, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)


    img[580:700, 20:233] = cv2.resize(imgRaw,(213,120))
    # To show our own image while playing the game we've built this game
    cv2.imshow('Image',img)
    key = cv2.waitKey(1)
    if key == ord('r'):
        ballPos = [100, 100]
        speedX = 15
        speedY = 15
        gameOver = False
        score = [0, 0]
        imgGameOver = cv2.imread("Resources/gameOver.png")

    # inorder to restart the game we've built this function. Basically we're again initializing our variables
