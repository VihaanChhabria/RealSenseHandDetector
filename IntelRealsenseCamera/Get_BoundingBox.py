import cv2
import mediapipe as mp
import imutils

import pyrealsense2 as rs
import numpy as np

def empty(var1):
    pass

def getContours(img, colorImgContour):
    triCoord, quadCoord = (0, 0), (0, 0)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            
            perimeter = cv2.arcLength(cnt, True)
            approxSides = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            approxSidesLen = len(approxSides)

            x, y, w, h = cv2.boundingRect(approxSides)

            if approxSidesLen == 3:
                triCoord = x, y
            elif approxSidesLen == 4:
                quadCoord = x, y

            cv2.drawContours(colorImgContour, contours, -1, (255, 255, 0), 7)
            cv2.circle(colorImgContour, (x, y), 7, (255, 0, 255), -1)

            print(approxSidesLen)

    return triCoord, quadCoord

def main():

    pipeline = ready_realsense()

    # Creates a track bar to make it easy to tune thresholds for the canny operator
    # Once tuned, this can be deleted
    #cv2.namedWindow("Threshold Track bar")
    #cv2.createTrackbar("Threshold1", "Threshold Track bar", 150, 255, empty)
    #cv2.createTrackbar("Threshold2", "Threshold Track bar", 150, 255, empty)

    #cv2.namedWindow("Result")

    while True:
        frames = pipeline.wait_for_frames()
        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        colorImgContour = color_image


        # Blurs the color frame for every 7 by 7 area of pixels 
        # to eliminate any slight inconsistencies in the frame
        colorImgBlur = cv2.GaussianBlur(color_image, (7, 7), 1)

        # Converts the image to gray scale to make major differences easy to see
        colorImgGray = cv2.cvtColor(colorImgBlur, cv2.COLOR_BGR2GRAY)

        # Gets track bar position to it easy to tune thresholds for the canny operator
        # Once tuned, this can be deleted
        #threshold1 = cv2.getTrackbarPos("Threshold1", "Threshold Track bar") # 68
        #threshold2 = cv2.getTrackbarPos("Threshold2", "Threshold Track bar") # 56

        # Highlights all edges
        colorImgCanny = cv2.Canny(colorImgGray, 68, 56)

        # Creates a numpy array with ones
        kernel = np.ones((5, 5))
        # Makes outlines bigger allowing for the shapes to be more apparent
        colorImg_Dil = cv2.dilate(colorImgCanny, kernel, 1)

        getContours(colorImg_Dil, colorImgContour)

        # Displaying the output
        cv2.imshow("Result", colorImgContour)

        # Program terminates when q key is pressed
        if cv2.waitKey(1) == ord('q'):
            pipeline.stop()
            cv2.destroyAllWindows()

def ready_realsense():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    return pipeline

if __name__ == "__main__":
    main()