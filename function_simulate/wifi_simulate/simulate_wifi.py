from tkinter import *
from tkinter import filedialog, messagebox, simpledialog
import cv2
import matplotlib.pyplot as plt
import numpy as np

################################################################################
#               BRESENHAM LINE ALGORITHM
################################################################################
def bresenham(start, end):
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1

    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points

################################################################################
#                         Heatmap Data to RGB IMAGE
################################################################################
def data2heatmap(data, dynamicRange = 'linear'):
    dataShape = data.shape

    # normalizing the data
    data = data.reshape((-1, 1))
    alpha = np.min(np.min(data)) 
    beta = np.max(np.max(data)) 

    gamma = beta - alpha
    data = data - alpha
    data = data / gamma


    # Intensity Transformation
    if dynamicRange.lower() == 'log':# be aware this manipulates the dynamic range
        # Constarst Streching
        for i in range(2):
            data = 1*np.log2(1+data)


    # Defining Colormap Transissions
    Rpx = np.array([0, .125, .38, .62, .88, 1])
    Gpx = Rpx
    Bpx = Rpx
    
    ### color original
    # Rpy = np.array([0, 0, .00, 1, 1, .5])
    # Gpy = np.array([0, 0, 1, 1, 0, 0])
    # Bpy = np.array([.5, 1, 1, 0, 0, 0])

    ### color modify
    Rpy = np.array([0.0,    0.3,     0.3,       0.4,     0.15,       0.0])
    Gpy = np.array([0.28,  0.58,     1,       1,           0.99,      0.85])
    Bpy = np.array([0.98,   1,     1,     0.1,            0.2,        0.0])

    RGBmap3D = np.zeros((1, data.size, 3))


    for i in range(data.size):
        if data[i] <= Rpx[1]:
            RGBmap3D[0, i, :] = [(np.diff(Rpy[0:2]) / np.diff(Rpx[0:2])) * (data[i] - Rpx[0]) + Rpy[0] ,
                                 (np.diff(Gpy[0:2]) / np.diff(Gpx[0:2])) * (data[i] - Gpx[0]) + Gpy[0] ,
                                 (np.diff(Bpy[0:2]) / np.diff(Bpx[0:2])) * (data[i] - Bpx[0]) + Bpy[0]] 
        elif data[i] <= Rpx[2]:
            RGBmap3D[0, i, :] = [(np.diff(Rpy[1:3]) / np.diff(Rpx[1:3])) * (data[i] - Rpx[1]) + Rpy[1],
                                 (np.diff(Gpy[1:3]) / np.diff(Gpx[1:3])) * (data[i] - Gpx[1]) + Gpy[1],
                                 (np.diff(Bpy[1:3]) / np.diff(Bpx[1:3])) * (data[i] - Bpx[1]) + Bpy[1]]

        elif data[i] <= Rpx[3]:
            RGBmap3D[0, i, :] = [(np.diff(Rpy[2:4]) / np.diff(Rpx[2:4])) * (data[i] - Rpx[2]) + Rpy[2],
                                 (np.diff(Gpy[2:4]) / np.diff(Gpx[2:4])) * (data[i] - Gpx[2]) + Gpy[2],
                                 (np.diff(Bpy[2:4]) / np.diff(Bpx[2:4])) * (data[i] - Bpx[2]) + Bpy[2]]

        elif data[i] <= Rpx[4]:
            RGBmap3D[0, i, :] = [(np.diff(Rpy[3:5]) / np.diff(Rpx[3:5])) * (data[i] - Rpx[3]) + Rpy[3],
                                 (np.diff(Gpy[3:5]) / np.diff(Gpx[3:5])) * (data[i] - Gpx[3]) + Gpy[3],
                                 (np.diff(Bpy[3:5]) / np.diff(Bpx[3:5])) * (data[i] - Bpx[3]) + Bpy[3]]

        elif data[i] <= Rpx[5]:
            RGBmap3D[0, i, :] = [(np.diff(Rpy[4:6]) / np.diff(Rpx[4:6])) * (data[i] - Rpx[4]) + Rpy[4],
                                 (np.diff(Gpy[4:6]) / np.diff(Gpx[4:6])) * (data[i] - Gpx[4]) + Gpy[4],
                                 (np.diff(Bpy[4:6]) / np.diff(Bpx[4:6])) * (data[i] - Bpx[4]) + Bpy[4]]

    RGBmap = np.reshape(RGBmap3D, (dataShape[0], dataShape[1], 3))




    return RGBmap, RGBmap3D

################################################################################
def wallExtraction(image):
        image = np.array(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        labledLines = np.zeros(gray.shape)
        # Extract the line 
        edges = cv2.Canny(gray, 20, 160, apertureSize =3)
        # Detect points that form a line
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 20, minLineLength=10, maxLineGap=30)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # cv2.line(labledLines, (x1, y1), (x2, y2), (255, 255, 255), thickness=1, lineType=8)
            cv2.line(labledLines, (x1, y1), (x2, y2), color=255, thickness=1, lineType=cv2.LINE_AA)
        labledLines = np.asarray(labledLines,dtype=np.int64)
        return labledLines, lines

def roi_image(image, width, height):
        # Find the largest contour
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_area = 0
        best_cnt = None
        for c in contours:
            _, _, w, h = cv2.boundingRect(c)
            if w * h > max_area:
                max_area = w * h
                best_cnt = c
        x, y, w, h = cv2.boundingRect(best_cnt)
        # Adjust the coordinates and size of the bounding box based on the binary image shape
        x -= 10 
        y -= 10 
        w += 20 
        h += 20
        x = max(x, 0)
        y = max(y, 0)
        w = min(w, width - x)
        h = min(h, height - y)
        
        # Extract the cut image
        cut_img = image[y:y+h, x:x+w]
        return cut_img
################################################################################
class multiWallModel:
    lightVel = 3e8   # light velocity
    nodePerMeter = 0.25 # This identifies the resolution of the estimation
    propFreq = 2.4e9 # 5 gHz
    d0 = 1           # reference distance of 1 meters
    propagationModel = "MW"
    wallsAtten = np.ones((1,256))*5 # 5 dB attenuation for each wall
    wallsAtten[0, 0] = 0  # do not change this (this for a case of clear LoS
    TxSuperposition = "CW" # ['CW'& 'Ind']Continuous Waveform (results in standing wave), Independent
    
    def __init__(self):
        if np.sum(self.wallsAtten) < 1:
            messagebox.showwarning("Warning","Please make sure wall attenuations ('wallAtten') are defined correctly.")

  
    def cal_ratio(self,binary_img,width,height):
        # Find the largest contour
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_area = 0
        best_cnt = None
        for c in contours:
            _, _, w, h = cv2.boundingRect(c)
            if w * h > max_area:
                max_area = w * h
                best_cnt = c
        x, y, w, h = cv2.boundingRect(best_cnt)

        # Adjust the coordinates and size of the bounding box based on the binary image shape
        x -= 5 
        y -= 5 
        w += 10 
        h += 10
        # x = max(x, 0)
        # y = max(y, 0)
        # w = min(w, width - x)
        # h = min(h, height - y)
        
        # Extract the cut image
        cut_img = binary_img[y:y+h, x:x+w]

        user_area = width*height
        ratio = (user_area*2)/(w*h)
        return ratio,cut_img

    def calibration(self,bwImDil,calUnit, nodePerMeter):
        numNodesX = bwImDil.shape[1] * calUnit * nodePerMeter*2
        numNodesY = bwImDil.shape[0] * calUnit * nodePerMeter*2
        nodesX = np.linspace(0, bwImDil.shape[1] - 1, int(numNodesX), dtype=np.int64)
        nodesY = np.linspace(0, bwImDil.shape[0] - 1, int(numNodesY), dtype=np.int64)
        gridX, gridY = np.meshgrid(nodesX, nodesY)  # Gridding the environment
        gridXCul = np.reshape(gridX, (-1, 1))
        gridYCul = np.reshape(gridY, (-1, 1))
        return calUnit,gridX,gridY,gridXCul,gridYCul
    
    def lineOfSight(self,TxNum,TxLocation,gridX,gridY):
        LoS = [None] * TxNum
        for i in range(TxNum):
            LoS[i] = np.reshape(np.sqrt((gridX - TxLocation[i, 0]) ** 2 + (gridY - TxLocation[i, 1]) ** 2), (-1, 1))
            # LoS[i] contains the distnace from Tx[i] to all the grid points (nodes)
        return LoS
    
    def wallsOnLineOfSight(self,TxNum,TxLocation,gridXCul,gridYCul,labledImage):
        LoSImage = np.zeros(labledImage.shape, dtype=np.int8)
        wallsOnLoS = [0] * TxNum
        for i in range(TxNum):
            temp = []
            for j in range(gridXCul.size):
                LoSLineXY = bresenham(TxLocation[i, :], (gridXCul[j, 0], gridYCul[j, 0]))
                LoSLineXY = np.asarray(LoSLineXY, dtype=np.int64)
                LoSImage = LoSImage * 0
                LoSImage[LoSLineXY[:, 1], LoSLineXY[:, 0]] = 1  # Crearint an image of the LoS
                # End of for
                # Intersecting the LoS line image with the labled image of the walls
                temp.append(np.unique((LoSImage * labledImage)))  # temporary holding the walls in between
            # End of for
            wallsOnLoS[i] = temp
        # End of For
        return wallsOnLoS

    def mwModel(self,ghz,lossExp,txPower,LoS,walls,wallsAtten,measurements=None):

        LoS[np.where(LoS<self.d0)] = self.d0  ## distances under 1 m are not acceptable (log10(<1) problem)
        delaySpr = LoS / self.lightVel  # delay spread calculation
        # Calculating Total Loss Caused by Walls
        totalWallLoss = np.zeros((len(LoS), 1))
        for i in range(len(walls)):
            totalWallLoss[i, 0] = np.sum(wallsAtten[0, walls[i]])
        # end of for
        RSS = txPower - (20*np.log10(ghz*1e9) + 20*np.log10(self.d0) - 147.55) \
                - 10 * lossExp * np.log10(LoS) - totalWallLoss

        return RSS, delaySpr, lossExp, wallsAtten,(None,None)

