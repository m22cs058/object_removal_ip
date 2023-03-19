import os
import cv2 as cv

main_folder = 'Image_data'
src = 'Original'
des = 'Pre_image'

src_folder = f'{main_folder}/{src}/'
des_folder = f'{main_folder}/{des}/'
count = 1

print(len(os.listdir(src_folder)))

for file_name in os.listdir(src_folder):
    # Construct old file name
    source = src_folder + file_name

    image = cv.imread(source, cv.IMREAD_UNCHANGED)
    # cv.imshow('Bounding Box', image)

    # cv.waitKey(0)
    # cv.destroyAllWindows()
    resized_image = cv.resize(image, (300, 300), interpolation = cv.INTER_LINEAR)
    # Adding the count to the new file name and extension

    # destination = des_folder + str(count) + ".jpeg"

    # Renaming the file
    cv.imwrite(f'{des_folder}/' + str(count) + ".jpeg", resized_image)
    # os.rename(source, destination)
    count += 1
    # break


print('All Files Renamed and Resize')

# print('New Names are')
# verify the result
res = os.listdir(des_folder)
print(len(res))