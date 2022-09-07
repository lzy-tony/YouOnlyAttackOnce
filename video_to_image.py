import cv2
import os

output_path = "/home/duanchengqi20/Patch/image4/"
input_path = "/home/duanchengqi20/Patch/data/video"
cnt = 0
files = os.listdir(input_path)
files.sort()
for i, file in enumerate(files):
    num = 1
    # print(os.path.join(input_path, file))
    if "test02" not in file:
        continue
    print(file)
    cnt += 1
    video = cv2.VideoCapture(os.path.join(input_path, file))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    print(fps, frames)
    while video.isOpened():
        is_read, frame = video.read()
        if is_read:
            file_name = "{}_{}".format(cnt, num)
            cv2.imwrite(output_path + str(file_name) + '.jpg', frame)#输出图片
            cv2.waitKey(1)
            num += 1
        else:
            break