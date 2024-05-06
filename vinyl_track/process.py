from itertools import count
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

DRAW = True


def resize_to_width(mat, w):
    h = int(mat.shape[0] * w / mat.shape[1])
    return cv.resize(mat, (w, h), interpolation=cv.INTER_AREA)


sift = cv.SIFT_create()
img1 = cv.imread("darktable_exported/P3160003.jpg", cv.IMREAD_GRAYSCALE)
img1 = resize_to_width(img1, 500)
kp1, des1 = sift.detectAndCompute(img1, None)


matrices = []

cap = cv.VideoCapture("P3160002.MOV")
# matcher = cv.FlannBasedMatcher(
#     {"algorithm": 1, "trees": 5},
#     {"checks": 50},
# )

matcher = cv.BFMatcher()

for i in tqdm(count(0), total=56 * 60):
    ret, img2 = cap.read()
    if not ret:
        break
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    img2 = resize_to_width(img2, 500)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matches = matcher.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
            -1, 1, 2
        )

        matrices.append(M)
        with open("matrices.pickle", "wb") as f:
            pickle.dump(matrices, f)
        if DRAW:
            dst = cv.perspectiveTransform(pts, M)
            img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
            cv.imwrite(f"out-{i:05d}.jpg", img2)

    else:
        print("Not enough matches are found - {}/{}".format(len(good), 10))
        matchesMask = None
        if DRAW:
            cv.imwrite(f"out-{i:05d}.jpg", img2)
        matrices.append(None)
        with open("matrices.pickle", "wb") as f:
            pickle.dump(matrices, f)

cap.release()
