"""Exact KGI PG sketch colours, ordering and rasterization at 1024×768.
Adapted from codes/KGI/generate_demo.py; see THIRD_PARTY.md.
"""
import math
import cv2
import numpy as np
import torch

def draw_skeleton(sk_pos):
    sk_pos[:, 0] = sk_pos[:, 0] * 768
    sk_pos[:, 1] = sk_pos[:, 1] * 1024

    sk_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    sk_Seq = [[0,1], [1,8], [1,9], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7]]

    stickwidth = 10

    jk_colors = [[255, 85, 0], [0, 255, 255], [255, 170, 0], [255, 255, 0], [255, 255, 0], [255, 170, 0], [85, 255, 0], [85, 255, 0], [0, 255, 255], [0, 255, 255]]
    sk_colors = [[255, 85, 0], [0, 255, 255], [0, 255, 255], [255, 170, 0], [255, 255, 0], [255, 255, 0], [255, 170, 0], [85, 255, 0], \
	          [85, 255, 0]]

    canvas = np.zeros((1024,768,3),dtype = np.uint8) # B,G,R order

    for i in range(len(sk_idx)):
        cv2.circle(canvas, (int(sk_pos[sk_idx[i]][0]),int(sk_pos[sk_idx[i]][1])), stickwidth, jk_colors[i], thickness=-1)
    
    for i in range(len(sk_Seq)):
        index = np.array(sk_Seq[i])
        cur_canvas = canvas.copy()
        Y = [sk_pos[index[0]][0],sk_pos[index[1]][0]]
        X = [sk_pos[index[0]][1],sk_pos[index[1]][1]]
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, sk_colors[i])
        canvas = cv2.addWeighted(canvas, 0, cur_canvas, 1, 0)
    
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    canvas = torch.from_numpy(canvas.transpose(2, 0, 1).copy()).float() / 255.0
    
    return canvas

def draw_cloth(ck_pos):
    ck_pos[:, 0] = ck_pos[:, 0] * 768
    ck_pos[:, 1] = ck_pos[:, 1] * 1024

    canvas = np.zeros((1024,768,3),dtype = np.uint8) # B,G,R order

    ck_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    ck_Seq = [[0,1], [1,2], [2,3], [3,4], \
        [4,31], [31,30], [30,29], [29,28], [28,27], [27,26], [26,25], [25,24], [24,23], [23,22], [22,21], [21,20], [20,19],\
        [19,18], [18,17], [17,16], [16,15], [15,14], [14,13], [13,12], [12,11], [11,10], [10,9], [9,8], [8,7], [7,6], [6,5], [5,0]]

    stickwidth = 10
    ck_colors = [255, 0, 0]

    for i in ck_idx:
        cv2.circle(canvas, (int(ck_pos[i][0]),int(ck_pos[i][1])), stickwidth, ck_colors, thickness=-1)
    
    for i in range(len(ck_Seq)):
        index = np.array(ck_Seq[i])
        cur_canvas = canvas.copy()
        Y = [ck_pos[index[0]][0], ck_pos[index[1]][0]]
        X = [ck_pos[index[0]][1], ck_pos[index[1]][1]]
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, ck_colors)
        canvas = cv2.addWeighted(canvas, 0, cur_canvas, 1, 0)
    
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    canvas = torch.from_numpy(canvas.transpose(2, 0, 1).copy()).float() / 255.0
    
    return canvas
