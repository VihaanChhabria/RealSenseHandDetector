import cv2
import mediapipe as mp
import imutils

import pyrealsense2 as rs
import numpy as np


mpHands = mp.solutions.hands # MP hand solution
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils # Draws the lines of hand joints

# Processing the input image
def process_image(img):
    # Converting the input to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(gray_image)

    # Returning the detected hands to calling function
    return results

# Drawing landmark connections
def draw_hand_connections(img, results, depth_colormap):
    if results.multi_hand_landmarks:
        print(results.multi_hand_landmarks[0].landmark[8])
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                print(lm)
                # Finding the coordinates of each landmark
                cx, cy = int(lm.x * w), int(lm.y * h)

                # Printing each landmark ID and coordinates
                # on the terminal
                #print(id, cx, cy)

                # Creating a circle around each landmark
                cv2.circle(img, (cx, cy), 10, (0, 255, 0),
                           cv2.FILLED)
                cv2.circle(depth_colormap, (cx, cy), 10, (0, 255, 0),
                           cv2.FILLED)
                # Drawing the landmark connections
                mpDraw.draw_landmarks(img, handLms,
                                      mpHands.HAND_CONNECTIONS)
                mpDraw.draw_landmarks(depth_colormap, handLms,
                                      mpHands.HAND_CONNECTIONS)

        return img
    
def main():
   # Replace 0 with the video path to use a
   # pre-recorded video
    #cap = cv2.VideoCapture(1)
    pipeline = ready_realsense()

    while True:
        frames = pipeline.wait_for_frames()
        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        image = color_image

        depth_frame = aligned_frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        if not color_frame:
            continue
        # Taking the input
        #success, image = cap.read()
        #image = imutils.resize(image, width=500, height=500)

        results = process_image(image)
        draw_hand_connections(image, results, depth_colormap)

        depth_colormap_dim = depth_colormap.shape
        
        if depth_colormap_dim != image.shape:
            resized_color_image = cv2.resize(image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((image, depth_colormap))


        # Displaying the output
        cv2.imshow("Hand tracker", images)

        # Program terminates when q key is pressed
        if cv2.waitKey(1) == ord('q'):
            pipeline.stop()
            cv2.destroyAllWindows()

def ready_realsense():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    return pipeline

if __name__ == "__main__":
    main()