import os
import cv2
import math
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw
import numpy as np
import paddleocr.tools.infer.utility as utility
from paddleocr.tools.infer.utility import check_gpu
from paddleocr.tools.infer.predict_det import TextDetector

from pkg.deploy.python.infer import Detector
from shapely.geometry import Point, Polygon, LineString

print("GPU", check_gpu(True))
print("CPU count", os.cpu_count())
print(os.getcwd())

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
OCR = {
    "chinese_cht_v3.0": PaddleOCR(lang="chinese_cht",
                                  det_model_dir="inference/Multilingual_PP-OCRv3_det_infer",
                                  cls_model_dir="inference/ch_ppocr_mobile_v2.0_cls_infer",
                                  rec_model_dir="inference/chinese_cht_PP-OCRv3_rec_infer",
                                  use_gpu=False, total_process_num=os.cpu_count(), use_mp=True, show_log=False),
    "ch_PP-OCRv3_xx": PaddleOCR(lang="ch",
                                det_model_dir="inference/ch_PP-OCRv3_det_infer",
                                cls_model_dir="inference/ch_ppocr_mobile_v2.0_cls_infer",
                                rec_model_dir="inference/ch_PP-OCRv3_rec_infer",
                                use_gpu=False, total_process_num=os.cpu_count(), use_mp=True, show_log=True,
                                use_angle_cls=True),
    "ch_PP-OCRv4_xx": PaddleOCR(lang="ch",
                                det_model_dir="inference/ch_PP-OCRv4_det_infer",
                                cls_model_dir="inference/ch_ppocr_mobile_v2.0_cls_infer",
                                rec_model_dir="inference/ch_PP-OCRv4_rec_infer",
                                use_gpu=False, total_process_num=os.cpu_count(), use_mp=True, show_log=True,
                                use_angle_cls=True),
}
# 模型文件目录文件夹
ppyolo_model_dir = "inference/ppyolo_mbv3_large_seal_infer"
db_model_dir = "inference/det_r50_seal_0203_all"

detector = Detector(ppyolo_model_dir,
                    device='CPU',
                    run_mode='paddle',
                    batch_size=1,
                    trt_min_shape=1,
                    trt_max_shape=1280,
                    trt_opt_shape=640,
                    trt_calib_mode=False,
                    cpu_threads=1,
                    enable_mkldnn=False,
                    enable_mkldnn_bfloat16=False,
                    output_dir='output',
                    threshold=0.5,
                    delete_shuffle_pass=False)

parser = utility.init_args()
args, _ = parser.parse_known_args()
args.det_model_dir = db_model_dir
args.rec_model_dor = db_model_dir
args.use_gpu = False
args.det_algorithm = "DB++"
args.det_db_box_thresh = 0.4
args.det_box_type = "poly"
args.det_limit_side_len = 320
text_detector = TextDetector(args)


def get_real_rotation_when_null_rect(rect_list):
    w_div_h_sum = 0
    count = 0
    for rect in rect_list:
        p0 = rect[0]
        p1 = rect[1]
        p2 = rect[2]
        p3 = rect[3]
        width = abs(p1[0] - p0[0])
        height = abs(p3[1] - p0[1])
        w_div_h = width / height
        if abs(w_div_h - 1.0) < 0.5:
            count += 1
            continue
        w_div_h_sum += w_div_h
    length = len(rect_list) - count
    if length == 0:
        length = 1
    if w_div_h_sum / length >= 1.5:
        return 1
    else:
        return 0


def get_real_rotation_flag(rect_lists):
    ret_rect = []
    w_div_h_list = []
    w_div_h_sum = 0
    for rect_list in rect_lists:
        for rect in rect_list:
            p0 = rect[0]
            p1 = rect[1]
            p2 = rect[2]
            p3 = rect[3]
            width = abs(p1[0] - p0[0])
            height = abs(p3[1] - p0[1])
            w_div_h = width / height
            # w_div_h_list.append(w_div_h)
            # print(w_div_h)
            if 5 <= abs(w_div_h - 1.0) <= 25 or 0.04 <= abs(w_div_h) <= 0.2:
                ret_rect.append(rect)
                w_div_h_sum += w_div_h

    if w_div_h_sum / len(ret_rect) >= 1.5:
        return 1, ret_rect
    else:
        return 0, ret_rect


def crop_image(rect, image):
    p0 = rect[0]
    p1 = rect[1]
    p2 = rect[2]
    p3 = rect[3]
    crop = image[int(p0[1]):int(p2[1]), int(p0[0]):int(p2[0])]
    # crop_image = Image.fromarray(crop)
    return crop


def get_img_real_angle(img: Image, ocr_model: str):
    # 用识别矩形框的坐标的分布规律判断是否需要旋转，目前计算还有问题
    ret_angle = 0
    ocr = OCR.get(ocr_model, OCR["ch_PP-OCRv3_xx"])
    # ocr = PaddleOCR(use_angle_cls=True)
    # angle_cls = ocr.ocr(img_path, det=False, rec=False, cls=True)

    rect_list = ocr.ocr(np.array(img), rec=False)
    print(rect_list)
    if rect_list != [[]]:
        try:
            real_angle_flag, rect_good = get_real_rotation_flag(rect_list)
            # rect_crop = choice(rect_good)
            rect_crop = rect_good[0]
            image_crop = crop_image(rect_crop, np.array(img))
            # ocr_angle = PaddleOCR(use_angle_cls=True)
            angle_cls = ocr.ocr(image_crop, det=False, rec=False, cls=True)
            print(angle_cls)
        except:
            real_angle_flag = get_real_rotation_when_null_rect(rect_list)
            # ocr_angle = PaddleOCR(use_angle_cls=True)
            angle_cls = ocr.ocr(np.array(img), det=False, rec=False, cls=True)
            print(angle_cls)
    else:
        return 0
    print('real_angle_flag:  {}'.format(real_angle_flag))
    if angle_cls[0][0][0] == '0':
        if real_angle_flag:
            ret_angle = 0
        else:
            ret_angle = 270
    if angle_cls[0][0][0] == '180':
        if real_angle_flag:
            ret_angle = 180
        else:
            ret_angle = 90
    return ret_angle


def text_ocr(img: Image, ocr_model: str) -> list:
    # angle=get_img_real_angle(img,ocr_model)
    # print(angle)
    ocr = OCR.get(ocr_model, OCR["ch_PP-OCRv4_xx"])
    return ocr.ocr(np.array(img), cls=True)


def replace_color(image, low_target_color, high_target_color, replacement_color):
    # 定义目标颜色的上下界
    lower_bound = np.array(low_target_color, dtype=np.uint8)
    upper_bound = np.array(high_target_color, dtype=np.uint8)

    # 在 BGR 颜色空间中创建一个掩码
    mask = cv2.inRange(image, lower_bound, upper_bound)

    # 使用替换颜色填充图像中的目标颜色区域
    image[mask != 0] = replacement_color

    return image


def seal_text_expand(img, angle):
    """
    将图片按照笛卡尔坐标系展开
    :param img:
    :param angle:
    :return:
    """
    # 根据计算的角度进行旋转
    ROI = img.copy()
    rows, cols, channel = ROI.shape
    img_r = rows // 2
    center = (cols // 2, rows // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle=angle, scale=1.0)
    rotated_gray = cv2.warpAffine(ROI, rotation_matrix, (cols, rows))
    # cv2.imshow('rotation', rotated_gray)
    # cv2.waitKey(0)

    # step 4 对图像进行坐标系的转换
    polarImg = cv2.warpPolar(rotated_gray, (int(img_r), int(2 * math.pi * img_r)), center, img_r,
                             cv2.INTER_LINEAR + cv2.WARP_POLAR_LINEAR)
    polarImg = cv2.flip(polarImg, 1)  # 镜像
    polarImg = cv2.transpose(polarImg)  # 转置
    # cv2.imshow('test', polarImg)
    # cv2.waitKey(0)
    return polarImg


def seal_calculate_angle(img, text_poly, flag=1):
    """
    从图片中心衍射一条线，从水平角度开始每5度生成一个射线，长度为宽度的一半
    如果该射线与曲线文本相交则继续生成，
    射线与曲线文本不想交时保留，并返回角度
    :param flag:由于图像是倒置状态 1为逆时针转动，-1为顺时针转动
    :param img:
    :param box:
    :return:
    """
    h, w, c = img.shape
    cen_x, cen_y = w // 2, h // 2
    init_point = Point(cen_x, cen_y)
    expand_geom = text_poly.buffer(15)
    # 初始角度和长度
    angle = 0 if flag == 1 else 360
    step = cen_x

    while (flag == 1 and angle != 360) or (flag == -1 and angle != 0):
        # 根据角度和步长计算新的点
        dx = step * math.cos(math.radians(angle))
        dy = step * math.sin(math.radians(angle))
        p1 = Point(init_point.x + dx, init_point.y + dy)

        # 生成直线
        line = LineString([init_point, p1])

        # 检查直线是否与多边形相交
        if line.intersects(expand_geom):
            angle += 5 * flag  # 增加角度
        else:
            # x, y = expand_geom.exterior.xy
            # l_x, l_y = line.xy
            # # 绘制线段
            # plt.plot(l_x, l_y, '-r', linewidth=2)
            # # 绘制点
            # plt.plot(x, y)  # 'ro' 表示红色圆点
            # plt.xlabel('X')
            # plt.ylabel('Y')
            # plt.title('Shapely 点的可视化')
            # plt.grid(True)
            # plt.axis('equal')  # 设置坐标轴比例相等
            # plt.show()
            break  # 如果不相交，停止循环

    return angle


def draw_box(img, np_boxes):
    """
    Args:
        img : opencv image
        np_boxes (np.ndarray): shape:[N,6], N: number of box,
                               matix element:[class, score, x_min, y_min, x_max, y_max]
    Returns:
        im (PIL.Image.Image): visualized image
    """
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # PIL image
    draw_thickness = min(im.size) // 320
    draw = ImageDraw.Draw(im)
    xmin, ymin, xmax, ymax = np_boxes[2:]
    # draw bbox
    draw.line(
        [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
         (xmin, ymin)],
        width=draw_thickness,
        fill="red")

    return im


def seal_ocr(img, img_drawed):
    img_copy = img.copy()
    det_result = detector.predict_image([img_copy[:, :, ::-1]], visual=False)  # bgr-->rgb
    result_array = det_result["boxes"]
    result_num = det_result['boxes_num']
    if result_num == 0:  # 不存在印章
        print("不存在印章！")
        return None

    max_indices = np.argsort(-result_array[:, 1])
    max_result = result_array[max_indices[0]].astype(int)
    print("检测印章位置坐标:", max_result)
    img_drawed = draw_box(img_drawed, max_result)  # 可视化印章检测框
    x_min, y_min, x_max, y_max = max_result[2:]
    _img = img_copy[y_min:y_max, x_min:x_max]
    seal_img = cv2.resize(_img, (320, 320))
    dt_boxes, _ = text_detector(seal_img)

    # 可视化印章上的文字检测框
    # src_im = utility.draw_text_det_res(dt_boxes, img_drawed)
    # cv2.imshow("test", src_im)

    for box in dt_boxes:  # 检测出曲线文字
        text_poly = Polygon(box)
        if text_poly.area / text_poly.minimum_rotated_rectangle.area < 0.9:  # 弯曲文本
            calculate_angle1 = seal_calculate_angle(seal_img, text_poly, flag=1)  # 逆时针
            calculate_angle2 = seal_calculate_angle(seal_img, text_poly, flag=-1)  # 顺时针
            res_angle = (calculate_angle1 + calculate_angle2) / 2
            print("旋转角度", res_angle)
            expand_img = seal_text_expand(seal_img, angle=res_angle)
            return expand_img

    return None


def seal_text_ocr(img, img_drawed=None):
    """
    印章文字检测
    :param img: 带有印章的图片
    :param img_drawed: 可视化印章检测位置图片
    :return:
    resize_img 印章展开图片
    texts 印章检测到所有文本
    ans 印章分段文本
    pos 分段文本位置
    """
    print("进入印章文字检测")
    if img_drawed is None:
        img_drawed = img.copy()
    resize_img = seal_ocr(np.array(img), img_drawed)
    if resize_img is None:
        return None

    texts = text_ocr(resize_img, "ch_PP-OCRv4_xx")
    # print("印章文本检测内容:", texts)
    ans = []
    pos = []
    for sub_list in texts[0]:
        # 获取文本内容，只保留识别出来的文本内容
        text = sub_list[1][0]
        # print("识别印章文本", text)
        pos.append(sub_list[0])
        ans.append(text)
    return resize_img, texts, ans, pos
