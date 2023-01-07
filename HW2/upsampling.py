from filters import *


def get_iterative_upsampling_out(inp, input_depth, w_size, spat_sig, spec_sig, return_time=False):
    start = timer()
    # Get upsample factor
    uf = int(np.log2(inp.shape[0] / input_depth.shape[0]))

    # Copy input images
    D = input_depth.copy()
    I = inp.copy()

    # Iteratively apply Joint_Bilateral filter
    for i in range(uf):
        # Double size of depth image and resize RGB image to match
        D = cv2.resize(D, None, fx=2, fy=2)
        I = cv2.resize(I, D.shape[:2][::-1])

        # Apply Joint_Bilateral filter with resized images
        D = get_joint_bilateral_out(I, D, w_size, spat_sig=spat_sig, spec_sig=spec_sig)

    # Resize depth image to match RGB image and apply Joint_Bilateral filter
    D = cv2.resize(D, inp.shape[:2][::-1])
    D = get_joint_bilateral_out(inp, D, w_size, spat_sig=spat_sig, spec_sig=spec_sig)

    if return_time:
        D, timer() - start
    return D


def get_joint_upsampling_out(inp, inp_depth, w_size, spat_sig, spec_sig, return_time=False):
    start = timer()

    # Get upsample factor
    uf = int(np.log2(inp.shape[0] / inp_depth.shape[0]))

    # Copy input images
    D = inp_depth.copy()
    I = inp.copy()

    # Double size of depth image and resize RGB image to match
    D = cv2.resize(D, None, fx=2, fy=2)
    I = cv2.resize(I, D.shape[:2][::-1])

    # Apply Joint_Bilateral filter with resized images

    D = get_joint_bilateral_out(I, D, w_size, spat_sig=spat_sig, spec_sig=spec_sig)

    if return_time:
        D, timer() - start
    return D
