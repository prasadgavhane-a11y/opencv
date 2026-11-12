import cv2
import sys
import numpy

PREVIEW = 0 # Preview Mode
BLUR    = 1 # BLurring Filter
FEATURES = 2 # Corner Features Detector
CANNY    = 3 # Canny Edge Detector


feature_params =  dict( maxCorners = 500, # maximum number of corners to detect in a single video frame.
                       qualityLevel = 0.2,
                       # qualityLevel=0.2: Sets the minimum acceptable quality for a 
                       # corner on a scale from 0 to 1. 
                       # OpenCV finds the single strongest corner in the image, 
                       # multiplies its score by 0.2 (20%), 
                       # and discards any weaker corners that fall below that threshold.
                       minDistance = 15,
# • Sets the minimum distance (in pixels) between two detected corners. 
# This prevents OpenCV from drawing dozens of overlapping green circles on the exact same spot, 
# forcing it to spread detections out across the frame.
                       blockSize = 9)
# •  Defines the size of the pixel neighborhood (9 x 9 pixels) 
# that OpenCV examines around each point to calculate 
# whether it is a sharp corner or just a flat edge.



s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]


image_filter = PREVIEW
alive = True


win_name = ' Camera Filters'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
result = None

source = cv2.VideoCapture(s)

while alive:
    has_frame, frame = source.read()
    if not has_frame:
        break

    frame = cv2.flip(frame, 1)

#  1 (flipCode): Tells OpenCV which axis to flip the image around:
#  1: Flips horizontally around the vertical y-axis (left becomes right).
#  0: Flips vertically around the horizontal x-axis (upside down).
# -1: Flips both horizontally and vertically at the same time.


    if image_filter == PREVIEW:
        result = frame
    elif image_filter == CANNY:
        result = cv2.Canny(frame, 145, 150)
    elif image_filter == BLUR:
        result = cv2.blur(frame, (13, 13))
    elif image_filter == FEATURES:
        result = frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
        if corners is not None:
            for x, y in numpy.float32(corners).reshape(-1, 2):
                cv2.circle(result, (int(x),int(y)), 10, (0, 255, 0), 1)

    cv2.imshow(win_name, result)

    key = cv2.waitKey(1)

    if key == ord('Q') or key == ord('q') or key == 27:
        alive = False
    elif key == ord('C') or key ==ord('c'):
        image_filter = CANNY
    elif key == ord('B') or key ==ord('b'):
        image_filter =  BLUR
    elif key == ord('F') or key ==ord('f'):
        image_filter = FEATURES
    elif key == ord('P') or key == ord("p"):
        image_filter = PREVIEW

source.release()
cv2.destroyWindow(win_name)


# types of blurring or smoothing filter
# median filter()
# gaussian filter()
# bilateral filter() can reduce unwanted noise while keeping edges fairly sharp