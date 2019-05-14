from pipeline import *

cap = cv2.VideoCapture("Test_video.mp4")
# fr = 25
# count = 0

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 25.0, (1280, 720))
LINE_WIDTH = 25
# _, frame_0 = cap.read()
pts = np.float32([(580, 450), (705, 450), (200, 650), (1160, 650)])
pts_python = [ (700, 435), (560, 435),(0, 650), (3200, 650)]
pts2 = np.float32([[0, 0], [500, 0], [0, 500], [500, 500]])
M = cv2.getPerspectiveTransform(pts, pts2)

# print(cv2.perspectiveTransform(np.array([(0, 0,1), (2, 0,1)]), M))
# for i in range(4):

# cv2.polylines(frame_0, [np.int32(pts_python)], 1, (140,255,0), 3)
mask = np.full((frame_0.shape[0], frame_0.shape[1]), 0, dtype=np.uint8)
cv2.fillPoly(mask, [np.int32(pts_python)], 255)

# for i in range(4):
#     cv2.circle(frame_0, pts_python[i], 5, (0,0,255), 10)
#     cv2.putText(frame_0, f"{i}", (pts_python[i][0]+20, pts_python[i][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (140,255,140))
# cv2.imshow("image", frame_0)
# while(1):
#     if cv2.waitKey(25) & 0xFF == ord('s'):
#         exit(100)
    #     break
# pts = np.float32(pts_python)


# tracker = cv2.TrackerMOSSE_create()

# blank_image = np.zeros((500,2000,3), np.uint8)
initBB = None
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
cap.set(1, 100)
while cap.isOpened():
    count += 1
    print(f"[{count}]")
    ret, frame = cap.read()
    try:
        delt = delta2.add(cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float)[:, :, 1].astype(np.uint8))
    except:
        delt = delt

    thr = cv2.threshold(delt, 200, 255, cv2.THRESH_BINARY)[1]
    thr = cv2.bitwise_or(thr, thr, mask=mask)
    kernel = np.ones((4, 4), np.uint8)
    img_dilated = cv2.dilate(thr, kernel, iterations=4)
    cv2.imshow("DIL", img_dilated)
    _,contours,_ = cv2.findContours(img_dilated, 1, 2)
    if contours :
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            if 4000 < w*h < 30000:
                print(f"area = {w * h}")
                # initBB = (x,y-80, w*2, h*2)
                # tracker.init(frame, initBB)
                car_img = frame[y-80 :y+h, x:x+w].copy()
                FH.check_and_add(car_img, (x,y-80, w, h+80), frame)
                # cv2.rectangle(frame, (x,y-80), (x + w, y + h), (0, 255, 0), 2)
                # cv2.circle(frame, (x+w//2, y+(h-80)//2), 5, (255,0,0),10)
            # print("!")


        # if len(cars) > 0:
        #     print("CAR")
        #
        # for (x, y, w, h) in cars:
        #     cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 255), 2)
    # cv2.imshow("delta_ROI", thr)
    # roi = frame[720:]
    # key = cv2.waitKey(1) & 0xFF
    # if key == ord("s"):
    #     # select the bounding box of the object we want to track (make
    #     # sure you press ENTER or SPACE after selecting the ROI)
    #     initBB = cv2.selectROI("Frame", frame, fromCenter=False,
    #                            showCrosshair=True)
    #
    #     # start OpenCV object tracker using the supplied bounding box
    #     # coordinates, then start the FPS throughput estimator as well
    #     tracker.init(frame, initBB)

    for (success, box) in FH.get_all_trackers(frame):
        if success:
            (x, y, w, h) = [int(v) for v in box]
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), -1)  # A filled rectangle
            alpha = 0.4  # Transparency factor.

            image_new = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            cv2.rectangle(image_new, (x, y), (x + w, y + h),
                          (0, 0, 255), 2)
            # cv2.line(image_new, (640, 720), (x+w//2, y+h//2), (255,0,0),5)

            def convert(x, y):
                return int((M[0][0] * x + M[0][1] * y + M[0][2]) / (M[2][0] * x + M[2][1] * y + M[2][2])), int((
                            M[1][0] * x + M[1][1] * y + M[1][2]) / (M[2][0] * x + M[2][1] * y + M[2][2]))

            dist = np.linalg.norm((convert(640, 720), convert(x+w//2, y+h//2)))

            cv2.putText(image_new, "{0:.1f}m".format(dist/100), (x+2, y+h), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255), 2)
            cv2.imshow("RES", image_new)


            # cv2.line(processed_img,convert(640, 720), convert(x + w, y + h), (0,255,0),5)
            # print())
    if ret:

        # gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        #
        # cars = cars_cascade.detectMultiScale(roi, 1.1, 5, 0 | cv2.CASCADE_SCALE_IMAGE, (30, 30))
        #
        # for (x, y, w, h) in cars:
        #     cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 255), 2)



        l_points, r_points, processed_img = process(frame, pts)
        cv2.imshow("Original", frame)
        #
        # # ROAD LANE DETECTION
        if l_points and r_points:
            res_0 = get_best_fit(l_points)
            res_1 = get_best_fit(r_points)
        #

            if res_0[1]:
                X= inter_X.compare(res_0[0][0][0])
                Y = inter_Y.compare(res_0[0][0][1])
                R = inter_R.compare(res_0[0][1])

                cv2.circle(processed_img, (int(X), int(Y)), int(R), (0,255,0), LINE_WIDTH)
            #
            else:
                res = []
                for i in range(0, 500, 20):
                    res.append(res_0[0](i))
                    res = left.compare(res)

                for i in range(25):
                    cv2.circle(processed_img, (int(res[i]), int(i*20)),10, (255,0,0))
            #     A = inter_A.compare(res_0[0][0])
            #     B = inter_B.compare( res_0[0][1])
            #     # print(A,B)
            #     cv2.line(processed_img, (0, int(B)), (1000, int( A* 1000 + B ) ), (255, 0, 0), LINE_WIDTH)
            # #
            #
            if res_1[1]:
                X = inter_X2.compare(res_1[0][0][0])
                Y = inter_Y2.compare(res_1[0][0][1])
                R = inter_R2.compare(res_1[0][1])
                cv2.circle(processed_img, (int(X), int(Y)), int(R), (0, 255, 0), LINE_WIDTH)
            else:
                res = []
                for i in range(0, 500, 20):
                    res.append(res_1[0](i))
                    res = right.compare(res)

                for i in range(25):
                    cv2.circle(processed_img, (int(res[i]), int(i * 20)), 10, (255, 0, 0))
                # cv2.line(processed_img, (0, int( res_1[0][1])), (1000, int( res_1[0][0]* 1000 + res_1[0][1] ) ), (255, 0, 0), LINE_WIDTH)
            #

        # for i in l_points:
        #     if i[0]:
        #         cv2.circle(processed_img, i, 10, (255, 0, 0), -1)
        # for i in r_points:
        #     if i[0]:
        #         cv2.circle(processed_img, i, 10, (0, 0, 255), -1)
        # cv2.line(processed_img, (408, 409), (613, 615), (255,0,0), 5)
        cv2.imshow("FinalRes", processed_img)
        # back = transform_perspective_back(mask, pts)
        # b_channel, g_channel, r_channel = cv2.split(frame)
        # alpha = np.zeros(b_channel.shape, dtype=b_channel.dtype)
        #
        # infill_tr = transform_perspective_back(poly, pts)
        #
        # fram = cv2.merge([b_channel, g_channel, r_channel, alpha])
        # fram = cv2.addWeighted(fram, 1, infill_tr, 0.4, 0)
        # masked_lines = cv2.bitwise_and(fram, fram, mask=cv2.bitwise_not(back[:, :, 3])) + back
        # r, g, b, _ = cv2.split(masked_lines)
        # out.write(cv2.merge([r, g, b]))
        # cv2.imshow('Frame', masked_lines)
        # a, b, c = cv2.split(res)
        # alpha = np.zeros(a.shape, dtype=a.dtype)
        #
        # res_4 = cv2.merge([a, b, c, alpha])
        # cv2.imshow('Frame2', cv2.bitwise_and(res_4, res_4, mask=cv2.bitwise_not(mask[:, :, 3])) + mask)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()
