
import cv2

video_path = "./town04_test1790.mp4"  # 视频地址
output_path = './pics/'  # 输出文件夹


if __name__ == '__main__':
    num = 721
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)#帧率
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)#总的帧数
    print("fps=",int(fps),"frames=",int(frames))#打印一下（以上三行可以不要，写在这里是为了大概知道视频会输出多少张图片来合理的取帧的间隔）
    while video.isOpened():
        is_read, frame = video.read()
        if is_read:
            file_name = '%d' % num
            cv2.imwrite(output_path + str(file_name) + '.png', frame)#输出图片
            cv2.waitKey(1)
            num += 1
        else:
            break


