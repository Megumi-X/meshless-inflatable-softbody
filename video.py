import cv2
import numpy as np
from pathlib import Path

def compute_histogram(frame):
    """计算帧的彩色直方图"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten()

project_folder = "D:/IIIS_THU/Research/Seeing-through-surface/tumbler"

def main(video_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return
    
    # 读取视频帧并计算直方图
    histograms = []
    frames = []
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        hist = compute_histogram(frame)
        histograms.append(hist)
        frames.append(frame)
        frame_id += 1

    cap.release()
    
    # 找到最相似的两帧
    min_distance = float('inf')
    frame1, frame2 = -1, -1
    for i in range(len(histograms)):
        for j in range(i + 5, len(histograms)):
            dist = cv2.compareHist(histograms[i], histograms[j], cv2.HISTCMP_CORREL)
            if dist < min_distance:
                min_distance = dist
                frame1, frame2 = i, j

    # 显示最相似的两帧
    if frame1 != -1 and frame2 != -1:
        print(f"Frames {frame1} and {frame2} are the most similar.")
        cv2.imshow('Frame 1', frames[frame1])
        cv2.imshow('Frame 2', frames[frame2])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(project_folder + "/tumbler.mp4")
