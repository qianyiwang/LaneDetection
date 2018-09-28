import cv2
import numpy as np
import time

cap = cv2.VideoCapture('video/test.mp4')

last_time = time.time()

def printTime():
    print('loop took {} seconds'.format(time.time()-last_time))
    global last_time
    last_time = time.time()

while(True):
    ret, frame = cap.read()
    imshape = frame.shape

    white_threshold = [200, 200, 200]
    yellow_threshold = [200, 200, 0]
    white_thresholds = (frame[:,:,0] < white_threshold[0]) | (frame[:,:,1] < white_threshold[1]) | (frame[:,:,2] < white_threshold[2])
    yellow_thresholds = (frame[:,:,0] > yellow_threshold[0]) | (frame[:,:,1] < yellow_threshold[1]) | (frame[:,:,2] < yellow_threshold[2])
    color_select = np.copy(frame)
    color_select[white_thresholds & yellow_thresholds] = [0,0,0]

    gray = cv2.cvtColor(color_select, cv2.COLOR_RGB2GRAY)
    gray_gaussian = cv2.GaussianBlur(gray,(3,3),0)
    edge = cv2.Canny(gray_gaussian, 100, 200)

    mask = np.zeros_like(edge)
    ignore_mask_color = 255

    vertices = np.array([[(0,imshape[0]), \
        (imshape[1]*1/3, imshape[0]*2/3), \
        (imshape[1]*2/3, imshape[0]*2/3), \
        (imshape[1],imshape[0])]], \
        dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edge = cv2.bitwise_and(edge, mask)

    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 1     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 100 #minimum number of pixels making up a line
    max_line_gap = 20    # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(masked_edge, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)

    try:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),10)
    except Exception as e:
        pass

    cv2.imshow('frame', frame)

    printTime()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
