import cv2
import numpy as np

cap = cv2.VideoCapture("1.mp4")

# ---- 初始化稳定化 ----
ret, prev_frame = cap.read()
if not ret:
    raise ValueError("视频无法读取！")

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---- 平移稳定化 ----
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)

    if curr_pts is not None and prev_pts is not None:
        valid_prev = prev_pts[status.flatten() == 1]
        valid_curr = curr_pts[status.flatten() == 1]

       
        valid_prev = valid_prev.reshape(-1, 2)
        valid_curr = valid_curr.reshape(-1, 2)

        if len(valid_prev) > 0 and len(valid_curr) > 0:
            dx = np.mean(valid_curr[:, 0] - valid_prev[:, 0])
            dy = np.mean(valid_curr[:, 1] - valid_prev[:, 1])
        else:
            dx, dy = 0, 0
    else:
        dx, dy = 0, 0

    # 平移矩阵
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    stabilized_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

    # 更新 prev_gray
    prev_gray = gray.copy()

    # ---- 前景掩码 (MOG2) ----
    if 'fgbg' not in globals():
        fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50, detectShadows=False)

    fgmask = fgbg.apply(stabilized_frame)

    # ---- 掩码融合 (形态学处理) ----
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    # ---- 连通域分析 ----
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask)

    for i in range(1, num_labels):  # 跳过背景
        x, y, w, h, area = stats[i]

        # 面积过滤
        if area < 500:
            continue

        # 提取候选区域
        mask_i = (labels == i).astype(np.uint8) * 255

        # ---- 光流特征筛选 ----
        # 在当前区域提取特征点
        pts = cv2.goodFeaturesToTrack(gray[y:y+h, x:x+w], maxCorners=50, qualityLevel=0.01, minDistance=5)
        if pts is not None:
            pts[:, 0, 0] += x
            pts[:, 0, 1] += y

            # 光流计算
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
            if next_pts is not None:
                valid_move = np.mean(np.linalg.norm(next_pts - pts, axis=2))
                if valid_move < 0.5:  # 基本静止，排除
                    continue

        # ---- 绘制结果 ----
        cv2.rectangle(stabilized_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("stabilized", stabilized_frame)
    cv2.imshow("mask", fgmask)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
