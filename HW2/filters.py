import numpy as np
import cv2
from tqdm import tqdm
from timeit import default_timer as timer


def get_filter_gaussian(sig, size):
    # size = np.round(2 * np.pi * sig).astype(int)
    center = int(size / 2)
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            diff = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            kernel[i, j] = np.exp(-(diff ** 2) / (2 * sig ** 2))
    return kernel / np.sum(kernel)


# Helper function for calculating weight
def p(val, sigma):
    sigma_sq = sigma * sigma
    normalization = np.sqrt(2 * np.pi) * sigma
    return (1 / normalization) * np.exp(-val / (2 * sigma_sq))


def get_bilateral_output(inp, spec_sig, spat_sig, window_size=5, return_time=False, show_tqdm=True):
    start = timer()
    inp = inp.astype(float)
    # Get dimensions of input image
    height, width = inp.shape[:2]

    # Create Gaussian kernel
    gaussian_kernel = get_filter_gaussian(spat_sig, window_size)

    # Set all values in output to 0
    output = np.zeros((height, width))  # np.zeros((height - window_size // 2, width - window_size // 2))

    # Iterate through image pixels
    rng = tqdm(range(window_size // 2, height - window_size // 2)) if show_tqdm else range(window_size // 2,
                                                                                           height - window_size // 2)

    for r in rng:
        for c in range(window_size // 2, width - window_size // 2):
            # Initialize sums
            sum_w = 0
            sum = 0

            # Iterate through window
            for i in range(-window_size // 2, window_size // 2 + 1):
                for j in range(-window_size // 2, window_size // 2 + 1):
                    # Calculate range difference
                    range_difference = np.abs(inp[r, c] - inp[r + i, c + j])

                    # Calculate weight
                    w = p(range_difference, spec_sig) * gaussian_kernel[i + window_size // 2][j + window_size // 2]

                    # Update sums
                    sum += inp[r + i, c + j] * w
                    sum_w += w

            # Set output pixel value
            output[r - window_size // 4, c - window_size // 4] = sum / sum_w

    if return_time:
        return output, timer() - start
    return output


def get_joint_bilateral_out(inp, inp_depth, window_size, spat_sig, spec_sig, return_time=False, show_tqdm=True):
    start = timer()

    inp = inp.astype(float)
    inp_depth = inp_depth.astype(float)

    # Get dimensions of input images
    height, width = inp.shape[:2]
    output = np.zeros((height, width))

    # Create Gaussian kernel
    gaussian_kernel = get_filter_gaussian(spat_sig, window_size)

    rng = tqdm(range(window_size // 2, height - window_size // 2)) if show_tqdm else range(window_size // 2,
                                                                                           height - window_size // 2)

    for r in rng:
        for c in range(window_size // 2, width - window_size // 2):
            # Initialize sums
            sum_w = 0
            sum = 0

            # Iterate through window
            for i in range(-window_size // 2, window_size // 2 + 1):
                for j in range(-window_size // 2, window_size // 2 + 1):
                    # Calculate range difference
                    range_difference = np.abs(inp[r, c] - inp[r + i, c + j])

                    # Calculate weight
                    w = p(range_difference, spec_sig) * gaussian_kernel[i + window_size // 2][j + window_size // 2]

                    # Update sums
                    sum += inp_depth[r + i, c + j] * w
                    sum_w += w

            # Set output pixel value
            output[r - window_size // 4, c - window_size // 4] = sum / sum_w
    if return_time:
        return output, timer() - start
    return output
