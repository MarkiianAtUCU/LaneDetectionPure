import math

import cv2
import numpy as np
from scipy import optimize, interp

old = 0.9
new = 0.1


class Interpolator_ARR:
    def __init__(self, thresh):
        self.prev_x = []
        self.threshold = thresh
    def compare(self, x):
        if not self.prev_x:
            self.prev_x = x
            return x
        for i in range(len(x)):
            if (self.prev_x[i] - x[i])**2 > self.threshold:
                self.prev_x[i] = self.prev_x[i]*old + x[i]*new
            else:
                self.prev_x[i] = x[i]
        return x

class Interpolator:
    def __init__(self, thresh, max_th = 9999999999):
        self.prev_x = None
        self.threshold = thresh
        self.max_th = max_th
    def compare(self, x):
        if not self.prev_x:
            self.prev_x = x
            return x

        if (self.prev_x - x)**2 > self.threshold:
            self.prev_x = self.prev_x*old + x*new
            return self.prev_x*old + x*new

        if (self.prev_x - x)**2 > self.max_th:
            return self.prev_x

        self.prev_x = x
        return x

class Delta:
    def __init__(self, capacity):
        self.capacity = capacity
        self.container = []

    def add(self, img):
        self.container.append(img)
        if len(self.container) > self.capacity:
            del self.container[0]
        res = self.container[0]

        if len(self.container) == 1:
            return res

        for i in range(1, len(self.container)):
            res = cv2.absdiff(res, self.container[i])

        return res


def transform_perspective(img, pts):
    # rows, cols, ch = img.shape
    pts2 = np.float32([[0, 0], [500, 0], [0, 500], [500, 500]])
    M = cv2.getPerspectiveTransform(pts, pts2)
    dst = cv2.warpPerspective(img, M, (500, 500))
    return dst


def transform_perspective_back(img, pts):
    rows, cols, _ = img.shape

    pts2 = np.float32([[0, 0], [500, 0], [0, 500], [500, 500]])
    M = cv2.getPerspectiveTransform(pts, pts2)
    #     dst = cv2.warpPerspective(img, M, (1280,720),cv2.INTER_LINEAR, cv2.WARP_INVERSE_MAP, cv2.BORDER_CONSTANT, 0);
    dst = cv2.warpPerspective(img, M, (1280, 720), cv2.INTER_LINEAR, cv2.WARP_INVERSE_MAP, cv2.BORDER_TRANSPARENT);
    return dst


def lr_split(lst):
    l = []
    r = []
    for i in lst:
        if i[0] < 250:
            l.append(i)

        else:
            r.append(i)

    return l, r


def draw_line(point_lists, point_lists2):
    width, height = 500, 500  # picture's size
    img = np.zeros((height, width, 4), np.uint8)  # make the background white
    line_width = 25
    for i in range(len(point_lists) - 2):
        # change color or make a color generator for your self
        pts = np.array([point_lists[i], point_lists[i + 1]], dtype=np.int32)
        cv2.polylines(img, [pts], False, (255, 0, 0, 255), thickness=line_width, lineType=cv2.LINE_AA)

        pts = np.array([point_lists2[i], point_lists2[i + 1]], dtype=np.int32)
        cv2.polylines(img, [pts], False, (0, 0, 255, 255), thickness=line_width, lineType=cv2.LINE_AA)

    return img


def draw_road(l, r):
    #     l, r = lr_split(pts)
    t = np.arange(0, 500, 20)
    xdata_1 = np.array(l)[:, 0]
    ydata_1 = np.array(l)[:, 1]
    z_1 = np.polyfit(ydata_1, xdata_1, 2)
    f_1 = np.poly1d(z_1)

    xdata = np.array(r)[:, 0]
    ydata = np.array(r)[:, 1]
    z_2 = np.polyfit(ydata, xdata, 2)
    f_2 = np.poly1d(z_2)

    l1, r1 = [(f_1(i), i) for i in t], [(f_2(i), i) for i in t]
    infill = np.ones((500, 500, 4), dtype=np.uint8)
    #     print(np.array(l1+r1[::-1]))
    #     print(r1+l1[::-1])
    #     print(r1)
    return draw_line(l1, r1), cv2.fillPoly(infill, [np.array(l1 + r1[::-1], dtype=np.int32)], (0, 255, 0, 255))


def frame_n(cap, frame):
    cap.set(1, frame)
    ret, frame = cap.read()
    return frame


def yw_data(img):
    # img = transform_perspective(frame_n(cap, 280),pts)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lower_yellow = np.array([20, 100, 100], dtype = 'uint8')
    upper_yellow = np.array([30, 255, 255], dtype='uint8')
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray_image, 200, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)
    return mask_yw_image
#
# def points_s(img_cap):
#     counter = 0
#     res_2=[]
#     r_r=[]
#     for i in range(10):
#         histogram = np.sum(img_cap[counter:counter+50], axis=0)
#         histogram[histogram <= 12000] = 0
#         histogram[histogram > 12000] = 1
#         count = 0
#         flag = 0
#         r_r=[]
#         for i in histogram:
#             count+=1
#             if i==1 and (flag==0):
#                 interm = count
#                 flag = 1
#             elif i==0 and flag == 1:
#     #             r_r.append(((count-interm)//2, counter+25))
#                 res_2.append((interm +(count-interm)//2, counter+25))
#                 flag=0
# #                 print((interm +(count-interm)//2, counter+50))
# #                 print(res_2)
#
#     #     res_2.append(r_r)
#     #     print(histogram)
# #         np.append(res,[histogram])
#     #     ax.plot(histogram)
#         counter+=50
#     return res_2
#     # plt.imshow(histogram)
# # x=points_s(dilated)
# # print(x)

def pt_2(img_cap):
    counter = 0
    l = []
    r = []
    for i in range(10):
        histogram = np.sum(img_cap[counter:counter + 50], axis=0)
        histogram[histogram <= 12000] = 0
        histogram[histogram > 12000] = 1
        count = 0
        flag = 0
        for i in histogram:
            count += 1
            if i == 1 and (flag == 0):
                interm = count
                flag = 1
            elif i == 0 and flag == 1:
                elem = (interm + (count - interm) // 2, counter + 25)
                if elem[0] < 250:
                    l.append(elem)
                else:
                    r.append(elem)

                flag = 0
        #         if (len(l)-len(r)>0):
        r += [(None, counter + 25) for i in range(len(l) - len(r))]

        #         elif (len(r)-len(l)>0):
        l += [(None, counter + 25) for i in range(len(r) - len(l))]

        counter += 50
    return l, r
    # plt.imshow(histogram)


def point_complement(l, r):
    ln = 0
    mid = 0
    for i in range(len(l)):
        if (l[i][0] != None and r[i][0] != None):
            width = r[i][0] - l[i][0]
            if width > 100:
                ln += 1
                mid += width

    res = mid // ln

    out_l = []
    out_r = []

    for i in range(len(l)):
        if (l[i][0] == None):
            #             out_l.append( (r[i][0]-res, r[i][1]))
            l[i] = (r[i][0] - res, r[i][1])
        elif (r[i][0] == None):
            #             out_r.append((l[i][0]+res, l[i][1]))
            r[i] = (l[i][0] + res, l[i][1])
    return l, r

def fit_circle(x, y):

    # x=np.array(list(map(lambda x:x[0], l)))
    # print(x)
    # exit(100)

    # y=np.array(list(map(lambda x:x[1], l)))
    x_m = np.mean(x)
    y_m = np.mean(y)

    def calc_R(xc, yc):
        """ calculate the distance of each 2D points from the center c=(xc, yc) """
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def f_2b(c):
        """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    def Df_2b(c):
        """ Jacobian of f_2b
        The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq"""
        xc, yc = c
        df2b_dc = np.empty((len(c), x.size))

        Ri = calc_R(xc, yc)
        df2b_dc[0] = (xc - x) / Ri  # dR/dxc
        df2b_dc[1] = (yc - y) / Ri  # dR/dyc
        df2b_dc = df2b_dc - df2b_dc.mean(axis=1)[:, np.newaxis]

        return df2b_dc

    center_estimate = x_m, y_m
    center_2b, ier = optimize.leastsq(f_2b, center_estimate, Dfun=Df_2b, col_deriv=True)


    xc_2b, yc_2b = center_2b
    Ri_2b = calc_R(xc_2b, yc_2b)
    R_2b = Ri_2b.mean()
    return (xc_2b, yc_2b), R_2b
    # print(f"({xc_2b}, {yc_2b}) R = {R_2b}")
    # cv2.circle(img, (int(xc_2b), int(yc_2b)), int(R_2b), (0,255,0), 7)
    # cv2.imshow("Ccc", img)
    # residu_2b = sum((Ri_2b - R_2b) ** 2)
    # residu2_2b = sum((Ri_2b ** 2 - R_2b ** 2) ** 2)

def fit_line(x,y):
    def f(x, A, B):
        return A * x + B

    A, B = optimize.curve_fit(f, x, y)[0]
    return A, B

def fit_polynom(x,y):
    z_1 = np.polyfit(y, x, 2)
    f_1 = np.poly1d(z_1)
    return f_1

def get_best_fit(l):
    x = np.array(list(map(lambda x: x[0], l)))
    y = np.array(list(map(lambda x: x[1], l)))
    circle = fit_circle(x, y)
    line = fit_line(x, y)
    polynom = fit_polynom(x, y)

    # err_line = 0
    # for i in range(len(x)):
    #     err_line += (y[i] - (line[0]*x[i] + line[1]))**2

    err_polynom = 0
    for i in range(len(x)):
        err_polynom += (x[i] - polynom(y[i]))**2

    err_circle = 0
    a = circle[0][0]
    b = circle[0][1]
    r = circle[1]

    for i in range(len(x)):
        y_new_one = math.sqrt(abs(r**2-(x[i]-a)**2))+b
        y_new_two = -math.sqrt(abs(r**2-(x[i]-a)**2))+b
        err_one = (y[i]-y_new_one)**2
        err_two = (y[i]-y_new_two)**2
        err_circle += err_one if err_one<err_two else err_two

    if err_circle < err_polynom-800:
        return circle, True

    return polynom, False

    # print(f"circle = {err_circle} line = {err_line} RES: {'circle' if err_circle<err_line else 'line'}")




