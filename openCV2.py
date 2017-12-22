import cv2
import numpy as np
import matplotlib.pyplot as plt

#LESSON 6 - Thresholding
def threshs():
    img = cv2.imread('bookpage.jpg')
    retval, threshold = cv2.threshold(img, 12, 255, cv2.THRESH_BINARY)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval2, threshold2 = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY)
    #                                                                                       block size  constant to subtract
    gaussian = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 2)
    retval3, otsu = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    cv2.imshow('thresh', gaussian)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#LESSON 7 - Color Filtering
def filtering():
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        # hsv hue saturation value
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #use HSV for filtering because BGR are dependent on each other, HSV isn't
        low_pink = np.array([150,150,0])
        upper_pink = np.array([255,255,255])

        mask = cv2.inRange(hsv, low_pink, upper_pink)
        result = cv2.bitwise_and(frame, frame, mask = mask)
        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)
        cv2.imshow('result', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows
    cap.release()

#LESSON 8 - Blurring and Smoothing
def blur_smooth():
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        # hsv hue saturation value
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #use HSV for filtering because BGR are dependent on each other, HSV isn't
        low_pink = np.array([150,50,50])
        upper_pink = np.array([255,255,255])

        mask = cv2.inRange(hsv, low_pink, upper_pink)
        result = cv2.bitwise_and(frame, frame, mask = mask)
        kernel = np.ones((15,15), np.float32)/225
        smoothed = cv2.filter2D(result,-1,kernel)
        blur = cv2.GaussianBlur(result, (15,15), 0)
        median = cv2.medianBlur(result, 15)
        bilateral = cv2.bilateralFilter(result, 15, 75, 75)

        #cv2.imshow('frame', frame)
        #cv2.imshow('mask', mask)
        cv2.imshow('result', result)
        #cv2.imshow('smoothed', smoothed)
        #cv2.imshow('blurred', blur)
        #cv2.imshow('median', median)
        cv2.imshow('bilateral', bilateral)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows
    cap.release()

#LESSON 9 - Morphological Transformation
def morph_transform():
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        low_pink = np.array([150,150,50])
        upper_pink = np.array([255,255,255])

        mask = cv2.inRange(hsv, low_pink, upper_pink)
        result = cv2.bitwise_and(frame, frame, mask = mask)
        kernel = np.ones((5,5), np.uint8)
        erosion = cv2.erode(mask, kernel, iterations = 1)
        dilation = cv2.dilate(mask, kernel, iterations = 1)

        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        #cv2.imshow('frame', frame) 
        cv2.imshow('opening', opening)
        cv2.imshow('closing', closing)   
        cv2.imshow('result', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows
    cap.release()

#LESSON 10 - Edge Detection and Gradients
def edges_grad():
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        #laplacian relaly emphasized edges
        laplacian = cv2.Laplacian(frame, cv2.CV_64F)
        sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize = 5)
        sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize = 5)
        edges = cv2.Canny(frame, 250, 250)

        cv2.imshow('og', frame)
        cv2.imshow('lap', laplacian)
        cv2.imshow('canny', edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows
    cap.release

#LESSON 11 - Template Matching
def template_match():
    img_bgr = cv2.imread('contacts.jpg')
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('contacts_crop.jpg', 0)
    #                   reverse array to give W x H instead of H x W
    w, h = template.shape[::-1]
    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.3
    loc = np.where(result >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_bgr, pt, (pt[0]+w, pt[1]+h), (0,255,255), 2)

    small = cv2.resize(img_bgr, (1000,600))
    cv2.imshow('detect', small)
    cv2.waitKey(0)
    cv2.destroyAllWindows

#LESSON 12 - GrabCut Foreground Extraction
def grabCut():
    img = cv2.imread('wanderer_fog.jpg')
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    rect = (100, 224, 520 ,791)

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask == 0), 0, 1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    plt.imshow(img)
    plt.colorbar()
    plt.show()

#LESSON 13 - Corner Detection
def corner_detect():
    img = cv2.imread('corner_detection.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    corners = cv2.goodFeaturesToTrack(gray, 10, 0.01, 10)
    corners = np.int0(corners)
    for corner in corners:
        x,y = corner.ravel()
        cv2.circle(img, (x,y), 3, 255, -1)

    cv2.imshow('Corner', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows

#LESSON 14 - Feature Matching (Homography) Brute Force
def feature_matching_BF():
    img1 = cv2.imread('contacts.jpg', 0)
    img2 = cv2.imread('contacts_crop.jpg', 0)

    orb = cv2.ORB_create()
    keypoint1, descriptor1 = orb.detectAndCompute(img1,None)
    keypoint2, descriptor2 = orb.detectAndCompute(img2,None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matches = bf.match(descriptor1, descriptor2)
    matches = sorted(matches, key = lambda x:x.distance)
    img3 = cv2.drawMatches(img1, keypoint1, img2, keypoint2, matches[:200], None, flags = 2)

    small = cv2.resize(img3, (1000,600))
    cv2.imshow('matches', small)
    cv2.waitKey(0)
    cv2.destroyAllWindows

#LESSON 15 - MOG Background Reduction
def mog_back_reduction():
    cap = cv2.VideoCapture(0)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    while True:
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)
        cv2.imshow('original', frame)
        cv2.imshow('fg', fgmask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows
    cap.release()

def main():
    morph_transform()

if __name__ == "__main__":
    main()