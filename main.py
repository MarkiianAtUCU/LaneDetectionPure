from pipeline import *


cap = cv2.VideoCapture("Test_video.mp4")
fr = 25
count = 0

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 25.0, (1280, 720))
LINE_WIDTH = 25
_, frame_0 = cap.read()
pts = np.float32([(580, 450), (705, 450), (200, 650), (1160, 650)])

# pts_python = [ (1000, 420), (550, 420),(0, 650), (5000, 650)]

# for i in range(4):

# cv2.polylines(frame_0, [np.int32(pts_python)], 1, (140,255,0), 3)
# for i in range(4):
#     cv2.circle(frame_0, pts_python[i], 5, (0,0,255), 10)
#     cv2.putText(frame_0, f"{i}", (pts_python[i][0]+20, pts_python[i][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (140,255,140))
# cv2.imshow("image", frame_0)
# while(1):

    # if cv2.waitKey(25) & 0xFF == ord('s'):
    #     exit(100)
    #     break
# pts = np.float32(pts_python)
cars_cascade = cv2.CascadeClassifier("cars_cascade_3.xml")
# OPENCV_OBJECT_TRACKERS = {
#         "csrt": cv2.TrackerCSRT_create,
#         "kcf": cv2.TrackerKCF_create,
#         "boosting": cv2.TrackerBoosting_create,
#         "mil": cv2.TrackerMIL_create,
#         "tld": cv2.TrackerTLD_create,
#         "medianflow": cv2.TrackerMedianFlow_create,
#         "mosse": cv2.TrackerMOSSE_create
#     }
# tracker = OPENCV_OBJECT_TRACKERS["mosse"]()

# blank_image = np.zeros((500,2000,3), np.uint8)
# initBB = None
inter_R = Interpolator(200)
inter_X = Interpolator(300)
inter_Y = Interpolator(300)
inter_R2 = Interpolator(500)
inter_X2 = Interpolator(500)
inter_Y2 = Interpolator(500)

inter_A = Interpolator(5, max_th=10)
inter_B = Interpolator(5)

left = Interpolator_ARR(30)
while cap.isOpened():
    count += 1
    ret, frame = cap.read()
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
    # if initBB is not None:
    #     # grab the new bounding box coordinates of the object
    #     (success, box) = tracker.update(frame)
    #
    #     # check to see if the tracking was a success
    #     if success:
    #         (x, y, w, h) = [int(v) for v in box]
    #         cv2.rectangle(frame, (x, y), (x + w, y + h),
    #                       (0, 255, 0), 2)
    if ret:

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # cars = cars_cascade.detectMultiScale(gray, 1.1, 5, 0 | cv2.CASCADE_SCALE_IMAGE, (30, 30))

        # for (x, y, w, h) in cars:
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)



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
                for i in range(0, 500, 20):
                    cv2.circle(processed_img, (int(res_0[0](i)), int(i)),10, (255,0,0))
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
                for i in range(0, 500, 20):
                    cv2.circle(processed_img, (int(res_1[0](i)), int(i)),10, (255,0,0))
                # cv2.line(processed_img, (0, int( res_1[0][1])), (1000, int( res_1[0][0]* 1000 + res_1[0][1] ) ), (255, 0, 0), LINE_WIDTH)
            #

        # for i in l_points:
        #     if i[0]:
        #         cv2.circle(processed_img, i, 10, (255, 0, 0), -1)
        # for i in r_points:
        #     if i[0]:
        #         cv2.circle(processed_img, i, 10, (0, 0, 255), -1)

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
