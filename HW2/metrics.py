import numpy as np
from skimage.metrics import structural_similarity


def ssd(A, B):
    return np.sum((A - B) ** 2)


def mse(A, B):
    return ((A - B) ** 2).mean()


def rmse(A, B):
    return np.sqrt(mse(A, B))


def psnr(A, B, max_value=255):
    mse_ = mse(A, B)
    if mse_ == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse_)))


def corr(A, B):
    return np.corrcoef(A.flatten(), B.flatten(), ddof=0)[0][1]


def ssim(A, B):
    return structural_similarity(A, B, channel_axis = -1)
