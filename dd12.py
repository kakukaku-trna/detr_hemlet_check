import cv2
import numpy as np

# 初始化 ORB 特征检测器
orb = cv2.ORB_create()

# 初始化 Vibe 背景建模
class VibeModel:
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    def apply(self, frame):
        return self.bg_subtractor.apply(frame)

# 实例化 Vibe 背景建模
vibe_model = VibeModel()

# 初始化混合高斯模型（Background Subtractor）
fgbg_gmm = cv2.createBackgroundSubtractorMOG2()

# 参数配置
morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 用于形态学操作的核
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 视频读取
cap = cv2.VideoCapture('1.mp4')
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# 主循环
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 转换到 HSV 空间
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 当前帧灰度图
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ORB 特征点检测与匹配
    prev_keypoints, prev_descriptors = orb.detectAndCompute(prev_gray, None)
    curr_keypoints, curr_descriptors = orb.detectAndCompute(gray_frame, None)

    # 特征点匹配
    if prev_descriptors is not None and curr_descriptors is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(prev_descriptors, curr_descriptors)
        matches = sorted(matches, key=lambda x: x.distance)

        # 提取匹配的关键点位置
        prev_pts = np.float32([prev_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        curr_pts = np.float32([curr_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # 计算单应矩阵
        H, mask = cv2.findHomography(prev_pts, curr_pts, cv2.RANSAC, 5.0)

        # 剔除背景运动：根据单应矩阵，估算整体背景运动
        if H is not None:
            height, width = frame.shape[:2]
            warped_prev_frame = cv2.warpPerspective(prev_frame, H, (width, height))

            # 通过背景运动估算，将前一帧的背景剔除
            motion_mask = np.abs(warped_prev_frame.astype(np.float32) - frame.astype(np.float32))
            motion_mask = np.sum(motion_mask, axis=2) > 50  # 阈值控制背景运动影响
        else:
            motion_mask = np.zeros_like(gray_frame)
    else:
        motion_mask = np.zeros_like(gray_frame)

    # 应用 Vibe 背景建模
    vibe_mask = vibe_model.apply(gray_frame)

    # 应用混合高斯模型背景建模
    gmm_mask = fgbg_gmm.apply(gray_frame)

    # 形态学操作去噪
    vibe_mask = cv2.morphologyEx(vibe_mask, cv2.MORPH_OPEN, morph_kernel)
    vibe_mask = cv2.morphologyEx(vibe_mask, cv2.MORPH_CLOSE, morph_kernel)

    gmm_mask = cv2.morphologyEx(gmm_mask, cv2.MORPH_OPEN, morph_kernel)
    gmm_mask = cv2.morphologyEx(gmm_mask, cv2.MORPH_CLOSE, morph_kernel)

    # 检测并绘制运动目标（基于Vibe背景建模）
    contours_vibe, _ = cv2.findContours(vibe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_vibe:
        if cv2.contourArea(contour) > 500:  # 过滤掉较小的目标
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 检测并绘制运动目标（基于混合高斯模型）
    contours_gmm, _ = cv2.findContours(gmm_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_gmm:
        if cv2.contourArea(contour) > 500:  # 过滤掉较小的目标
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # 显示结果
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Vibe Foreground Mask', vibe_mask)
    cv2.imshow('GMM Foreground Mask', gmm_mask)
    cv2.imshow('Background Motion Mask', motion_mask.astype(np.uint8) * 255)

    # 更新前一帧
    prev_gray = gray_frame.copy()
    prev_frame = frame.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
