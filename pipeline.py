from help_functions import *
delta = Delta(10)


def process(frame, pts):
    """

    :param frame:
    :return: left_points, right_points, bird_eye_original
    """

    img_perspactive_bird_eye = transform_perspective(frame, pts)
    img_hlcs_channel = cv2.cvtColor(img_perspactive_bird_eye, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = img_hlcs_channel[:, :, 2].astype(np.uint8)
    # cv2.imshow("S_channel", s_channel)
    channels_combined_yw = cv2.bitwise_or(yw_data(img_perspactive_bird_eye), s_channel)
    # cv2.imshow("combined", channels_combined_yw)

    ret, th1 = cv2.threshold(channels_combined_yw, 140, 255, cv2.THRESH_BINARY)
    # cv2.imshow("threshold", th1)
    delt = delta.add(th1)
    kernel = np.ones((4, 4), np.uint8)
    img_dilated = cv2.dilate(delt, kernel, iterations=4)
    # cv2.imshow("dilated", img_dilated)

    # black = np.zeros((500, 500, 1), dtype=np.uint8)
    # mask = cv2.merge([black, black, black])
    try:
        l, r = point_complement(*pt_2(img_dilated))
    except:
        l, r =[], []
    # mask, poly = draw_road(l, r)
    return l, r, img_perspactive_bird_eye




