import os
import cv2
from PIL import Image

from pkg.ocr import seal_ocr, text_ocr


def read_images_from_folder(folder_path):
    image_list = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                image_path = os.path.join(root, filename)
                image = cv2.imread(image_path)
                if image is not None:
                    image_list.append(image)
                    image_dir_list.append(image_path)
    return image_list, image_dir_list


if __name__ == "__main__":
    import time

    flag = 1  # 0:文件夹图片测试,1:单张图片测试
    image_list = []
    image_dir_list = []
    if flag == 0:
        # 文件夹图片测试
        img_dir = "./train_data/"
        image_list, image_dir_list = read_images_from_folder(img_dir)
    else:
        # 单张图片测试
        img_path = "./train_data/train/test.png"
        frame = cv2.imread(img_path)
        image_list.append(frame)
        image_dir_list.append(img_path)

    cv2.namedWindow("expand", 0)
    cv2.resizeWindow("expand", 1256, 200)

    time3 = time.time()
    for i, frame in enumerate(image_list):
        print(image_dir_list[i])
        img_drawed = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        resize_img, ad_resize_img = seal_ocr(frame, img_drawed)
        show_img = resize_img
        if resize_img is None:
            print("检测失败!")
            continue

        texts = text_ocr(resize_img, "ch_PP-OCRv4_xx")
        is_second = True  # 是否进行第二次检测，进行中心点校正
        for sub_list in texts[0]:
            # 获取文本内容，只保留识别出来的文本内容
            text = sub_list[1][0]
            if "局" in text or "监督" in text:
                is_second = False
                break

        seal_ans = []
        seal_pos = []
        if is_second:
            print("调整中心后第二次检测")
            time4 = time.time()
            texts = text_ocr(ad_resize_img, "ch_PP-OCRv4_xx", det=False)
            print('ocr use%.2f' % (time4 - time3))
            show_img = ad_resize_img

            # 第二次检测只做分类，没有位置检测
            for sub_list in texts[0]:
                # 获取文本内容，只保留识别出来的文本内容
                text = sub_list[0]
                seal_ans.append(text)
        else:
            for sub_list in texts[0]:
                # 获取文本内容，只保留识别出来的文本内容
                text = sub_list[1][0]
                # print("识别印章文本", text)
                seal_pos.append(sub_list[0])
                seal_ans.append(text)

        print(texts)
        cv2.imshow('expand', resize_img)
        key = cv2.waitKey(0)
        if key == 27:
            break
