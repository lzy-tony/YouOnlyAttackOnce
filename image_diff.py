from PIL import Image
from PIL import ImageChops
import os
input_path1 = "/home/duanchengqi20/Patch/image2"
input_path2 = "/home/duanchengqi20/Patch/image4"
def compare_images(path_one, path_two, diff_save_location):
    try:
        image_one = Image.open(path_one)
        image_two = Image.open(path_two)
    except:
        return
    diff = ImageChops.difference(image_one, image_two)
    if diff.getbbox() is None:
        return
    else:
        diff.save(diff_save_location)

if __name__ == '__main__':
    files = os.listdir(input_path1)
    for i, file in enumerate(files):
        print(i)
        compare_images('/home/duanchengqi20/Patch/image2/{}'.format(file), '/home/duanchengqi20/Patch/image4/{}'.format(file), '/home/duanchengqi20/Patch/diff/{}'.format(file))
    print(files)    