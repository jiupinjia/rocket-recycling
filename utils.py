import numpy as np
import cv2

################ Some helper functions... ####################

def moving_avg(x, N=500):

    if len(x) <= N:
        return []

    x_pad_left = x[0:N]
    x_pad_right = x[-N:]
    x_pad = x_pad_left[::-1] + x + x_pad_right[::-1]
    y = np.convolve(x_pad, np.ones(N) / N, mode='same')
    return y[N:-N]


def load_bg_img(path_to_img, w, h):
    bg_img = cv2.imread(path_to_img, cv2.IMREAD_COLOR)
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
    bg_img = cv2.resize(bg_img, (w, h))
    return bg_img


def create_circle_poly(center, radius, N=50):
    pts = []
    for i in range(N):
        x = center[0] + radius*np.cos(i/N*2*np.pi)
        y = center[1] + radius*np.sin(i/N*2*np.pi)
        pts.append([x, y])
    return pts


def create_ellipse_poly(center, rx, ry, N=50):
    pts = create_circle_poly(center, radius=1.0, N=N)
    for pt in pts:
        pt[0] = pt[0] * rx
        pt[1] = pt[1] * ry
    return pts


def create_rectangle_poly(center, w, h):
    x0, y0 = center
    pts = [[x0-w/2, y0+h/2], [x0+w/2, y0+h/2], [x0+w/2, y0-h/2], [x0-w/2, y0-h/2]]
    return pts


################ Let's do some math... ####################

def scale_matrix(sx=1.0, sy=1.0, sz=1.0):

    ScaleMatrix = np.eye(4)
    ScaleMatrix[0, 0] = sx  # scale on x
    ScaleMatrix[1, 1] = sy  # scale on y
    ScaleMatrix[2, 2] = sz  # scale on z

    return ScaleMatrix

def rotation_matrix(rx=0., ry=0., rz=0.):

    # input should be radians (e.g., 0, pi/2, pi)

    Rx = np.eye(4)
    Rx[1, 1] = np.cos(rx)
    Rx[1, 2] = -np.sin(rx)
    Rx[2, 1] = np.sin(rx)
    Rx[2, 2] = np.cos(rx)

    Ry = np.eye(4)
    Ry[0, 0] = np.cos(ry)
    Ry[0, 2] = np.sin(ry)
    Ry[2, 0] = -np.sin(ry)
    Ry[2, 2] = np.cos(ry)

    Rz = np.eye(4)
    Rz[0, 0] = np.cos(rz)
    Rz[0, 1] = -np.sin(rz)
    Rz[1, 0] = np.sin(rz)
    Rz[1, 1] = np.cos(rz)

    # RZ * RY * RX
    RotationMatrix = np.mat(Rz) * np.mat(Ry) * np.mat(Rx)

    return np.array(RotationMatrix)


def translation_matrix(tx=0., ty=0., tz=0.):

    TranslationMatrix = np.eye(4)
    TranslationMatrix[0, -1] = tx
    TranslationMatrix[1, -1] = ty
    TranslationMatrix[2, -1] = tz

    return TranslationMatrix


def create_pose_matrix(tx=0., ty=0., tz=0.,
                       rx=0., ry=0., rz=0.,
                       sx=1.0, sy=1.0, sz=1.0,
                       base_correction=np.eye(4)):

    # Scale matrix
    ScaleMatrix = scale_matrix(sx, sy, sz)

    # Rotation matrix
    RotationMatrix = rotation_matrix(rx, ry, rz)

    # Translation matrix
    TranslationMatrix = translation_matrix(tx, ty, tz)

    # TranslationMatrix * RotationMatrix * ScaleMatrix
    PoseMatrix = np.mat(TranslationMatrix) \
                 * np.mat(RotationMatrix) \
                 * np.mat(ScaleMatrix) \
                 * np.mat(base_correction)

    return np.array(PoseMatrix)

