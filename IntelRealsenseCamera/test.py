import cv2

def onTrackbarChange(value):
    print("Trackbar value:", value)

# Create a named window
cv2.namedWindow("Parameters")

# Create the first trackbar with initial value 0 and maximum value 255
cv2.createTrackbar("Trackbar 1", "Parameters", 0, 255, onTrackbarChange)

# Create the second trackbar with initial value 0 and maximum value 100
cv2.createTrackbar("Trackbar 2", "Parameters", 0, 100, onTrackbarChange)

while True:
    # Retrieve the positions of both trackbars
    trackbar1_pos = cv2.getTrackbarPos("Trackbar 1", "Parameters")
    trackbar2_pos = cv2.getTrackbarPos("Trackbar 2", "Parameters")
    print("Trackbar 1 position:", trackbar1_pos)
    print("Trackbar 2 position:", trackbar2_pos)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
