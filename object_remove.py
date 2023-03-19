import cv2 as cv
import numpy as np
import timeit



import os


main_folder = 'Image_data'
src = 'Pre_image'
mask_des = 'Mask'
TELEA_des = 'Final_image/INPAINT_TELEA'
NS_des = 'Final_image/INPAINT_NS'
FAST_des = 'Final_image/FAST'
BEST_des = 'Final_image/BEST'

src_folder = f'{main_folder}/{src}/'
mask_des_folder = f'{main_folder}/{mask_des}/'
TELEA_des_folder = f'{main_folder}/{TELEA_des}/'
NS_des_folder = f'{main_folder}/{NS_des}/'
FAST_des_folder = f'{main_folder}/{FAST_des}/'
BEST_des_folder = f'{main_folder}/{BEST_des}/'



def extract_number(s):
    try:
        return int(s[:-5])
    except ValueError:
        return s
    
files = sorted(os.listdir(src_folder), key=extract_number)



def telea(files, src_folder, TELEA_des_folder):
    count = 1
    mask = cv.imread(f'{main_folder}/mask_sq.jpg', cv.IMREAD_UNCHANGED)
    time = []
    for file_name in files:
        start = timeit.default_timer()

        # Construct old file name

        source = src_folder + file_name

        image = cv.imread(source, cv.IMREAD_UNCHANGED)

        hight,width = image.shape[:2]

        # w,h =  width//5, hight//5

        # y = int(np.clip(np.random.normal(hight/2.,hight/5.),0,hight-1))
        # x = int(np.clip(np.random.normal(width/2.,width/5.),0,width-1))


        # Create a binary mask of the same shape as the image
        # mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Draw a white rectangle on the mask using the coordinates of the bounding box
        # cv.rectangle(mask, (x, y), (x + w, y + h), color=(255), thickness=-1)

        # selected_area = image[y:y+h, x:x+w]

        # Create the inpainting mask
        # mask_sa = np.zeros(selected_area.shape[:2], np.uint8)

        # Draw a white rectangle on the mask to indicate the area that needs to be inpainted
        # cv.rectangle(mask_sa, (10, 10), (190, 190), 255, -1)

        #cv.INPAINT_NS
        # Apply the inpainting algorithm on the selected area
        # print('hi')
        inpaint_area = cv.inpaint(image, mask, inpaintRadius=3, flags=cv.INPAINT_TELEA)

        # Replace the selected area in the original image with the inpainted area
        # image[y:y+h, x:x+w] = inpaint_area

        # # Display the results
        # cv.imshow("Original image", image)
        # cv.imshow("Selected area", mask)
        # cv.imshow("Inpainted area", inpaint_area)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        # mask_destination = mask_des_folder + 'mask' +str(count) + ".jpeg"
        org_destination = TELEA_des_folder + 'telea_' + str(count) + ".jpeg"

        # Renaming the file
        # cv.imwrite(mask_destination, mask)
        cv.imwrite(org_destination, inpaint_area)
        # os.rename(source, destination)
        count += 1
        #Your statements here

        stop = timeit.default_timer()

        time.append(stop - start) 
        # if count%10 == 0:
        #     print(count)
        # break


    print('All Image obejcts are removed using TELEA')
    # print('New Names are')
    # verify the result
    res = os.listdir(TELEA_des_folder)
    print(len(res))

    return time


def ns(files, src_folder, NS_des_folder):
    count = 1
    mask = cv.imread(f'{main_folder}/mask_sq.jpg', cv.IMREAD_UNCHANGED)
    time = []
    for file_name in files:
        # Construct old file name
        start = timeit.default_timer()

        source = src_folder + file_name

        image = cv.imread(source, cv.IMREAD_UNCHANGED)

        
        inpaint_area = cv.inpaint(image, mask, inpaintRadius=3, flags=cv.INPAINT_NS)

        org_destination = NS_des_folder + 'ns_' + str(count) + ".jpeg"

        # Renaming the file

        cv.imwrite(org_destination, inpaint_area)

        count += 1
        stop = timeit.default_timer()

        time.append(stop - start) 
        # if count%10 == 0:
        #     print(count)
        # break


    print('All Image obejcts are removed using NS')
    # print('New Names are')
    # verify the result
    res = os.listdir(NS_des_folder)
    print(len(res))

    return time


def fast(files, src_folder, FAST_des_folder):
    count = 1
    mask = cv.imread(f'{main_folder}/mask_sq.jpg', cv.IMREAD_UNCHANGED)
    time = []
    for file_name in files:
        # Construct old file name
        start = timeit.default_timer()

        source = src_folder + file_name

        image = cv.imread(source, cv.IMREAD_UNCHANGED)

        
        # hight,width = image.shape[:2]
        # print(hight, width)
        # point = (x,y)

        # Define the coordinates of the bounding box
        # x1, y1, x2, y2 = 50, 100, 250, 300

        # Create a binary mask of the same shape as the image
        # mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Draw a white rectangle on the mask using the coordinates of the bounding box
        # cv.rectangle(mask, (x, y), (x+w, y+h), color=(255), thickness=-1)

        # print(image.shape)
        # mask = cv.resize(mask, (image.shape[1], image.shape[0]))
        # _, mask = cv.threshold(mask, 128, 255, cv.THRESH_BINARY)
        mask1 = cv.bitwise_not(mask)
        distort = cv.bitwise_and(image, image, mask=mask1)

        restored1 = image.copy()
        # restored2 = image.copy()
        cv.xphoto.inpaint(distort, mask1, restored1, cv.xphoto.INPAINT_FSR_FAST)
        # cv.xphoto.inpaint(distort, mask1, restored2, cv.xphoto.INPAINT_FSR_BEST)


        org_destination = FAST_des_folder + 'fast_' + str(count) + ".jpeg"

        # Renaming the file

        cv.imwrite(org_destination, restored1)

        count += 1
        stop = timeit.default_timer()

        time.append(stop - start) 
        # if count%10 == 0:
        #     print(count)
        # break


    print('All Image obejcts are removed using FAST')
    # print('New Names are')
    # verify the result
    res = os.listdir(FAST_des_folder)
    print(len(res))

    return time


def best(files, src_folder, BEST_des_folder):
    count = 1
    mask = cv.imread(f'{main_folder}/mask_sq.jpg', cv.IMREAD_UNCHANGED)
    time = []

    for file_name in files:
        # Construct old file name
        start = timeit.default_timer()

        source = src_folder + file_name

        image = cv.imread(source, cv.IMREAD_UNCHANGED)

        
        mask1 = cv.bitwise_not(mask)
        distort = cv.bitwise_and(image, image, mask=mask1)


        restored2 = image.copy()

        cv.xphoto.inpaint(distort, mask1, restored2, cv.xphoto.INPAINT_FSR_BEST)


        org_destination = BEST_des_folder + 'best_' + str(count) + ".jpeg"

        # Renaming the file

        cv.imwrite(org_destination, restored2)

        count += 1
        stop = timeit.default_timer()

        time.append(stop - start) 
        # if count%10 == 0:
        #     print(count)
        # break


    print('All Image obejcts are removed using BEST')
    # print('New Names are')
    # verify the result
    res = os.listdir(BEST_des_folder)
    print(len(res))

    return time


print()
print(f"TELEA")
telea_time = telea(files, src_folder, TELEA_des_folder)
print(f"MAX time: {max(telea_time)} | MIN time: {min(telea_time)} | MEAN time: {(sum(telea_time)/len(telea_time))}")
print()

print()
print(f"NS")
ns_time = ns(files, src_folder, NS_des_folder)
print(f"MAX time: {max(ns_time)} | MIN time: {min(ns_time)} | MEAN time: {(sum(ns_time)/len(ns_time))}")
print()

print()
print(f"FAST")
fast_time = fast(files, src_folder, FAST_des_folder)
print(f"MAX time: {max(fast_time)} | MIN time: {min(fast_time)} | MEAN time: {(sum(fast_time)/len(fast_time))}")
print()

print()
print(f"BEST")
best_time = best(files, src_folder, BEST_des_folder)
print(f"MAX time: {max(best_time)} | MIN time: {min(best_time)} | MEAN time: {(sum(best_time)/len(best_time))}")

# TIME
# TELEA
# All Image obejcts are removed using TELEA
# 302
# MAX time: 0.4295058059942676 | MIN time: 0.34758844699535985 | MEAN time: 0.3544479060560752


# NS
# All Image obejcts are removed using NS
# 302
# MAX time: 0.3505540289916098 | MIN time: 0.3317721069906838 | MEAN time: 0.3355730801227307


# FAST
# All Image obejcts are removed using FAST
# 302
# MAX time: 2.162906326004304 | MIN time: 0.355255723989103 | MEAN time: 1.3708214036550697


# BEST
# All Image obejcts are removed using BEST
# 293
# MAX time: 44.723446532996604 | MIN time: 7.015455540007679 | MEAN time: 24.77946511923122