from pipeline import *

cap = cv2.VideoCapture("./../tests/Test_video.mp4")

cap.set(1, 100)

count = 0

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 25.0, (1280, 720))
LINE_WIDTH = 25

_, frame_0 = cap.read()
mask = np.full((frame_0.shape[0], frame_0.shape[1]), 0, dtype=np.uint8)
cv2.fillPoly(mask, [np.int32(pts_python)], 255)

# bar_outer = cv2.imread("hud_top.png", cv2.IMREAD_UNCHANGED)
# mask0 = bar_outer[:,:,3]
# color_overlay = bar_outer[::2]

# overlay = cv2.merge()


inter_R = Interpolator(200)
inter_X = Interpolator(300)
inter_Y = Interpolator(300)
inter_R2 = Interpolator(500)
inter_X2 = Interpolator(500)
inter_Y2 = Interpolator(500)

inter_A = Interpolator(5, max_th=10)
inter_B = Interpolator(5)

left = Interpolator_ARR(10)
right = Interpolator_ARR(10)


delta2 = Delta(2)
car_img = None
FH = FeatureContainer(5, 50)

while cap.isOpened():
    HUD_text_list = []
    HUD_rectangle_list = []
    count += 1
    # print(f"[{count}]")
    ret, frame = cap.read()
    # cv2.imshow("orig", frame)
    overlay = frame.copy()
    alpha = 0.4

    try:
        delt = delta2.add(cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float)[:, :, 1].astype(np.uint8))
    except:
        delt = delt

    thr = cv2.threshold(delt, 200, 255, cv2.THRESH_BINARY)[1]
    thr = cv2.bitwise_or(thr, thr, mask=mask)
    kernel = np.ones((4, 4), np.uint8)
    img_dilated = cv2.dilate(thr, kernel, iterations=4)
    # cv2.imshow("DIL", img_dilated)
    _,contours,_ = cv2.findContours(img_dilated, 1, 2)
    if contours:
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if 4000 < w*h < 30000:
                car_img = frame[y-80 :y+h, x:x+w].copy()
                FH.check_and_add(car_img, (x,y-80, w, h+80), frame)


    for (success, box) in FH.get_all_trackers(frame):
        if success:
            (x, y, w, h) = [int(v) for v in box]
            dist = np.linalg.norm((convert(640, 720), convert(x+w//2, y+h//2)))/100
            # TODO: car logo
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255) if dist< 10 else ((0,255,255 ) if dist < 13 else (0,255,0)), -1)
            HUD_text_list.append(("{0:.1f}m".format(dist), (x+2, y+h-5), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255), 2))
            HUD_rectangle_list.append(((x, y), (x + w, y + h), (0, 0, 255) if dist < 10 else ((0,255,255 ) if dist < 13 else (0,255,0)), 2))


    if ret:
        l_points, r_points, processed_img = process(frame, pts)
        cv2.circle(processed_img, l_points[-1], 10, (255,255,255), 10)
        cv2.circle(processed_img, r_points[-1], 10, (255,255,255), 10)
        l, r = 165-l_points[1][0], r_points[1][0]-165
        # print((1-l/(l+r))/2, (1 - r/(l+r))/2)
        # cv2.imshow("Original", processed_img)

        # ROAD LANE DETECTION
        if l_points and r_points:
            res_0 = get_best_fit(l_points)
            res_1 = get_best_fit(r_points)
            if res_0[1]:
                X= inter_X.compare(res_0[0][0][0])
                Y = inter_Y.compare(res_0[0][0][1])
                R = inter_R.compare(res_0[0][1])
                cv2.circle(processed_img, (int(X), int(Y)), int(R), (0,255,0), LINE_WIDTH)
            else:
                res_points_L = []
                for i in range(-50, 551, 50):
                    res_points_L.append(res_0[0](i))
                    res_points_L = left.compare(res_points_L)
                for i in range(12):
                    cv2.circle(processed_img, (int(res_points_L[i]), int(i*50)),10, (255,0,0))

            if res_1[1]:
                X = inter_X2.compare(res_1[0][0][0])
                Y = inter_Y2.compare(res_1[0][0][1])
                R = inter_R2.compare(res_1[0][1])
                cv2.circle(processed_img, (int(X), int(Y)), int(R), (0, 255, 0), LINE_WIDTH)
            else:
                res_points_R = []
                for i in range(-50, 551, 50):
                    res_points_R.append(res_1[0](i))
                    res_points_R = right.compare(res_points_R)

                for i in range(12):
                    cv2.circle(processed_img, (int(res_points_R[i]), int(i * 50)), 10, (255, 0, 0))




        # exit(1)
        res_l = [(res_points_L[i],i * 50) for i in range(12)]
        res_r = [(res_points_R[i],i * 50) for i in range(12)]


        blank_image = np.zeros((500, 500, 3), np.uint8)
        cv2.fillPoly(blank_image, [np.array(res_l+res_r[::-1], dtype=np.int32)], (255,0,0))
        dst = cv2.warpPerspective(blank_image, M, (1280, 720), cv2.INTER_LINEAR, cv2.WARP_INVERSE_MAP, cv2.BORDER_TRANSPARENT);
        overlay = cv2.bitwise_and(overlay, overlay, mask=cv2.bitwise_not(dst[:, :, 0]))+dst

        image_result_HUD = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        blank_image = np.zeros((500, 500, 3), np.uint8)

        for i in range(len(res_l) - 2):
            pts_l = np.array([res_l[i], res_l[i + 1]], dtype=np.int32)
            cv2.polylines(blank_image, [pts_l], False, (255, 0, 0), thickness=LINE_WIDTH)

            pts_r = np.array([res_r[i], res_r[i + 1]], dtype=np.int32)
            cv2.polylines(blank_image, [pts_r], False, (255, 0, 0), thickness=LINE_WIDTH)

        dst = cv2.warpPerspective(blank_image, M, (1280, 720), cv2.INTER_LINEAR, cv2.WARP_INVERSE_MAP,
                                  cv2.BORDER_TRANSPARENT);
        image_result_HUD = cv2.bitwise_and(image_result_HUD, image_result_HUD, mask=cv2.bitwise_not(dst[:, :, 0])) + dst

        for i in HUD_text_list:
            cv2.putText(image_result_HUD, *i)

        for i in HUD_rectangle_list:
            cv2.rectangle(image_result_HUD, *i)

        cv2.imshow("RES", image_result_HUD)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

cap.release()
# out.release()

cv2.destroyAllWindows()
