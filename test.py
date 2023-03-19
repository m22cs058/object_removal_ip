import cv2
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# Load the image
img = cv2.imread('image.jpg')

# Create a window to show the image
cv2.namedWindow('image', cv2.WINDOW_NORMAL)

# Show the image in the window
cv2.imshow('image', img)

# Define the mouse callback function
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode, mask
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.circle(mask, (x, y), 5, (255, 255, 255), -1)
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            else:
                cv2.circle(mask, (x, y), 5, (0, 0, 0), -1)
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.circle(mask, (x, y), 5, (255, 255, 255), -1)
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        else:
            cv2.circle(mask, (x, y), 5, (0, 0, 0), -1)
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

# Initialize the drawing parameters
drawing = False
mode = True
ix, iy = -1, -1

# Create the mask
mask = np.zeros(img.shape[:2], dtype=np.uint8)

# Set the mouse callback function
cv2.setMouseCallback('image', draw_circle)

while True:
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break
    elif k == ord('c'):
        img = cv2.imread('images.jpg')
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
    elif k == ord('i'):
        mask1 = cv2.bitwise_not(mask)
        distort = cv2.bitwise_and(img, img, mask=mask1)
        restored1 = img.copy()
        cv2.xphoto.inpaint(distort, mask1, restored1, cv2.xphoto.INPAINT_FSR_FAST)
        cv2.imshow('INPAINT_FSR_FAST', restored1)
    elif k == ord('j'):
        mask1 = cv2.bitwise_not(mask)
        distort = cv2.bitwise_and(img, img, mask=mask1)
        restored2 = img.copy()
        cv2.xphoto.inpaint(distort, mask1, restored2, cv2.xphoto.INPAINT_FSR_BEST)
        cv2.imshow('INPAINT_FSR_BEST', restored2)
        
    elif k == ord('k'):
        #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        inpaint_area = cv2.inpaint(img, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        cv2.imshow('INPAINT_TELEA', inpaint_area)
        
    elif k == ord('l'):
        cv2.imshow('Mask', mask1)
        #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        inpaint_area = cv2.inpaint(img, mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
        cv2.imshow('INPAINT_NS', inpaint_area)

cv2.destroyAllWindows()