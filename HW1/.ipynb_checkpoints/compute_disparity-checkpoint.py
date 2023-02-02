import numpy as np
from tqdm import tqdm

def NaiveStereoVision(l_img, r_img, k_size, max_offset, scale = 3, show_tqdm = True):
    
    def mse(l_win, r_win):
        return np.sum((l_window.flatten()-r_window.flatten())**2)
    
    def inv_corr_coeff(l_win, r_win):
        return 1/(np.corrcoeff(l_win.flatten(),r_win.flatten())[0,-1] + 2) # +2 against zero division
    
    
    
    img_h,img_w,color_depth = l_img.shape
    corres_dict = {}

    half_k_size = k_size // 2
    
    rng = tqdm(range(half_k_size, img_h - half_k_size)) if show_tqdm else range(half_k_size, img_h - half_k_size)
    for h in rng: # img_h - w_size
        for w in (range(half_k_size, img_w -  half_k_size)):
            corr_coeff_list = []   
            l_window = l_img[h - half_k_size:h+half_k_size,w - half_k_size:w+half_k_size,:]
            for w_check in (range(w,w + max_offset)):
                r_window = r_img[h - half_k_size:h+half_k_size,w_check - half_k_size:w_check+half_k_size,:]
                if r_window.shape==l_window.shape:
                    corr_coeff = mse(l_window, r_window) # mse, inv_corr_coeff
                    corr_coeff_list.append(corr_coeff)
            corres_dict[(h,w)] = w + np.argmin(corr_coeff_list) 

    disp_img = np.zeros((img_h,img_w))
    for h,w in corres_dict:
        disp_img[h,w] = np.abs(corres_dict[(h,w)] - w) * scale

    return disp_img


###

def SSD(pixelL,pixelR,sigma=2):
    ssd = np.mean((pixelL-pixelR)**2/(sigma**2))
    return ssd

def dynamic_matching_forward(rowL, rowR, occlusion_cost = 1):
    N = rowL.shape[0]
    M = rowR.shape[0]
    cost_matrix = np.ones((N,M))
    match_matrix = np.zeros((N,M))

    cost_matrix[0][0] = SSD(rowL[0], rowR[0])

    for i in range(1, N):
        cost_matrix[i][0] = i*occlusion_cost
    for j in range(1, M):
        cost_matrix[0][j] = j*occlusion_cost

    for i in range(1, N):
        for j in range(1, M):
            min_match = cost_matrix[i - 1][j -1] + SSD(rowL[i], rowR[j])
            min_l = cost_matrix[i - 1][j] + occlusion_cost
            min_r = cost_matrix[i][j - 1] + occlusion_cost

            cost_matrix[i][j] = np.min([min_match, min_l, min_r])
            match_matrix[i][j] = np.argmin([min_match, min_l, min_r]) + 1
    return (match_matrix, cost_matrix)


def dynyamic_matching_backtrack(match_matrix):
    rowL = rowR = match_matrix.shape[0]-1
    disp_vec = np.zeros((match_matrix.shape[0],1))
    disp_acc = 0
    
    while(rowL!=0 and rowR!=0):
        z = match_matrix[rowL][rowR]
        if(z==1):
            rowL -= 1 
            rowR -= 1
            disp_vec[rowL] = abs(rowR-rowL)
        elif(z==2):
            rowL -= 1 
            disp_acc += 1
            disp_vec[rowL] = 0
        elif(z==3):
            rowR -= 1
            disp_acc -= 1

    return disp_vec

def dynamic_disparity(imgL_bw, imgR_bw, lmbd = 1, show_tqdm = True):

    disp_map = np.zeros(imgR_bw.shape)
    
    rng = tqdm(range(imgL_bw.shape[0])) if show_tqdm else range(imgL_bw.shape[0])
    for row in rng:
        row_match_matrix, row_cost_matrix = dynamic_matching_forward(imgL_bw[row], imgR_bw[row], occlusion_cost = lmbd)

        disp_row= dynyamic_matching_backtrack(row_match_matrix)
        
        disp_map[row] = disp_row

    return disp_map

