import numpy as np
import cv2 as cv

# The given video and calibration data
video_file = '../Camera_Calibration/video_file.avi'
K = np.array([[2.83314803e+03, 0.00000000e+00, 1.91286089e+03],
              [0.00000000e+00, 2.93743676e+03, 1.00917492e+03],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist_coeff = np.array([0.15505307, -0.26711852, -0.00479147, 0.00182268, -0.67999317])
board_pattern = (10, 7)
board_cellsize = 0.025
board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

# Open a video
video = cv.VideoCapture(video_file)
assert video.isOpened(), 'Cannot read the given input, ' + video_file

# Prepare 3D points on a chessboard
obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

# Run pose estimation
while True:
    # Read an image from the video
    valid, img = video.read()
    if not valid:
        break

    # Estimate the camera pose
    success, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
    if success:
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        # 집 형태: 정육면체 바닥 + 삼각 지붕
        base = board_cellsize * np.array([
            [4, 2, 0], [5, 2, 0], [5, 3, 0], [4, 3, 0],     # 바닥
            [4, 2, -1], [5, 2, -1], [5, 3, -1], [4, 3, -1], # 윗면
        ])
        roof = board_cellsize * np.array([
            [4, 2, -1], [5, 2, -1], [4.5, 2.5, -1.5],       # 앞 삼각형 지붕
            [5, 3, -1], [4, 3, -1], [4.5, 2.5, -1.5]        # 뒤 삼각형 지붕
        ])
        all_points = np.vstack([base, roof])
        imgpts, _ = cv.projectPoints(all_points, rvec, tvec, K, dist_coeff)

        # 정육면체 부분 그리기
        imgpts = imgpts.reshape(-1, 2).astype(int)
        # 아래 사각형
        cv.polylines(img, [imgpts[0:4]], True, (255, 0, 0), 2)
        # 위 사각형
        cv.polylines(img, [imgpts[4:8]], True, (0, 0, 255), 2)
        # 세로 선
        for i in range(4):
            cv.line(img, imgpts[i], imgpts[i+4], (0, 255, 0), 2)

        # 지붕 삼각형
        cv.polylines(img, [np.array([imgpts[8], imgpts[9], imgpts[10]])], True, (200, 100, 255), 2)
        cv.polylines(img, [np.array([imgpts[11], imgpts[12], imgpts[13]])], True, (200, 100, 255), 2)

        # 지붕 선분 강조
        cv.line(img, imgpts[10], imgpts[13], (255, 150, 0), 2)

        # Print the camera position
        R, _ = cv.Rodrigues(rvec) # Alternative) `scipy.spatial.transform.Rotation`
        p = (-R.T @ tvec).flatten()
        info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    # Show the image and process the key event
    cv.imshow('Pose Estimation (Chessboard)', img)
    key = cv.waitKey(1)
    if key == ord(' '):
        key = cv.waitKey()
    if key == 27: # ESC
        break

video.release()
cv.destroyAllWindows()