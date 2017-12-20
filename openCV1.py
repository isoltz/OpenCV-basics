import cv2
import numpy as np
import matplotlib.pyplot as plt


#LESSON 1 - Loading Images
def load_images():
    img = cv2.imread('football.png',cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#LESSON 2 - Loading Video Source
def load_video():
    cap = cv2.VideoCapture(0) #can put video file here
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('myframe',frame)
        cv2.imshow('grayframe',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

#LESSON 3 - Drawing and Writing on Image
def draw_write():
    img = cv2.imread('football.png', cv2.IMREAD_COLOR)
    cv2.line(img, (150,0), (150,425), (255,0,0), 15)
    cv2.rectangle(img, (0,100), (330,350), (0,0,255), 3)
    cv2.circle(img, (150, 200), 30, (100,100,100), -5)

    pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(img, [pts], True, (0,255,255), 3)

    font = cv2.FONT_HERSHEY_SIMPLEX
#                     content   start         size          thickness   
    cv2.putText(img, 'openCV', (0,130), font, 1, (200,255,0), 2, cv2.LINE_AA)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows

#LESSON 4 - Image Operations
def img_ops():
    img = cv2.imread('football.png', cv2.IMREAD_COLOR)
    img[55,55] = [0,0,0]
    px = img[55,55]
    print(px)
    img[100:150, 100:150] = [0,0,0]

    laces = img[70:150, 50:190]
    img[0:80, 0:140] = laces
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows

#LESSON 5 - Image Arithmetic and Logic
def img_arith_logic():
    shutter = cv2.imread('Shutter_Island.jpg')
    linc = cv2.imread('Lincoln.jpg')
    baseball = cv2.imread('Baseball.jpg')

    #add = shutter + linc
    #add = cv2.add(shutter, linc)                       
                                                        #gamma
    #weighted = cv2.addWeighted(shutter, 0.75, linc, 0.25, 0)
    rows,cols,channels = baseball.shape
    roi = linc[500:(rows+500), 200:cols+200]

    greyball = cv2.cvtColor(baseball, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(greyball, 5, 0, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)
    linc_back = cv2.bitwise_and(roi, roi, mask = mask_inv)
    ball_fore = cv2.bitwise_and(baseball, baseball, mask = mask_inv)
    dst = cv2.add(linc_back, ball_fore)
    linc[500:rows+500, 200:cols+200] = dst

    #cv2.imshow('mask',mask)
    small = cv2.resize(linc, (500,750))
    cv2.imshow('image', small)
    cv2.waitKey(0)
    cv2.destroyAllWindows

def main():
    img_arith_logic()

if __name__ == "__main__":
    main()

