import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

main_folder = 'Image_data'
gt_src = 'Pre_image'
telea_des = 'Final_image/INPAINT_TELEA'
ns_des = 'Final_image/INPAINT_NS'
fast_des = 'Final_image/FAST'
best_des = 'Final_image/BEST'



def Eval_Matrices(ground_truth, impainted_img):
  #Calculate mean square error
  mse = np.mean((ground_truth - impainted_img) ** 2)
  #Calculate normalized mean square error
  nmse = mse / np.mean(ground_truth ** 2)
  # if ground truth image and impainted image are indentical
  if mse == 0:
      return float('inf')
  #PSNR Value
  psnr = 20 * np.log10(np.amax(ground_truth) / np.sqrt(mse))

#   win_size = min(ground_truth.shape[:2]) // 10

#   cv.imshow('1', ground_truth)
#   cv.imshow('2', impainted_img)

#   cv.waitKey(0)
#   cv.destroyAllWindows()
  g_gray = cv.cvtColor(ground_truth, cv.COLOR_RGB2GRAY)
  i_gray = cv.cvtColor(impainted_img, cv.COLOR_RGB2GRAY)

#   cv.imshow('1', g_gray)
#   cv.imshow('2', i_gray)

#   cv.waitKey(0)
#   cv.destroyAllWindows()

#   ssim_score, _ = ssim(ground_truth, impainted_img, multichannel=True, full=True)
  ssim_r, _ = ssim(ground_truth[:,:,0], impainted_img[:,:,0], multichannel=True, full=True) 
  ssim_g, _ = ssim(ground_truth[:,:,1], impainted_img[:,:,1], multichannel=True, full=True) 
  ssim_b, _ = ssim(ground_truth[:,:,2], impainted_img[:,:,2], multichannel=True, full=True) 
  wssim = 1/3 * (ssim_r + ssim_g + ssim_b)
  return mse,nmse,psnr,wssim
#   return mse,nmse,psnr


src_folder = f'{main_folder}/{gt_src}/'
telea_folder = f'{main_folder}/{telea_des}/'
ns_folder = f'{main_folder}/{ns_des}/'
fast_folder = f'{main_folder}/{fast_des}/'
best_folder = f'{main_folder}/{best_des}/'

count = 1

# eval_list = []
# img_dict = dict()

def extract_number(s):
    try:
        return int(s[:-5])
    except ValueError:
        return s
    

def extract_number_telea(s):
    try:
        return int(s[6:-5])
    except ValueError:
        return s
    
def extract_number_ns(s):
    try:
        return int(s[3:-5])
    except ValueError:
        return s
    
def extract_number_fast(s):
    try:
        return int(s[5:-5])
    except ValueError:
        return s
    
def extract_number_best(s):
    try:
        return int(s[5:-5])
    except ValueError:
        return s

# Get the list of files and sort them based on the numerical part of their names
ori_files = sorted(os.listdir(src_folder), key=extract_number)
telea_files = sorted(os.listdir(telea_folder), key=extract_number_telea)
ns_files = sorted(os.listdir(ns_folder), key=extract_number_ns)
fast_files = sorted(os.listdir(fast_folder), key=extract_number_fast)
best_files = sorted(os.listdir(best_folder), key=extract_number_best)

# print(ori_files)
# print(imp_files)

# print(sorted(os.listdir(src_folder)))
# print(type(os.listdir(impaint_folder)))
def cal_mat(ori_files, or_files, folder):
  mse_list, nmse_list, psnr_list, ssim_list = [], [], [], []
  count = 1
  for o_file, i_file in zip(ori_files, or_files):
      # eval_dict = dict()

      # print(o_file, i_file)
      source = src_folder + o_file
      desti = folder + i_file
      
      o_image = cv.imread(source, cv.IMREAD_UNCHANGED)
      i_image = cv.imread(desti, cv.IMREAD_UNCHANGED)

      

      mse, nmse, psnr, ssim_score = Eval_Matrices(o_image, i_image)
      # mse, nmse, psnr = Eval_Matrices(o_image, i_image)

      mse_list.append(mse)
      nmse_list.append(nmse)
      psnr_list.append(psnr)
      ssim_list.append(ssim_score)

      # eval_dict.add('NMSE', nmse)
      # eval_dict.add('PSNR', psnr)
      # eval_dict.add('SSIM SCORE', ssim_score)

      # print(count, eval_dict)
      # img_dict.add(f'img_{count}', eval_dict)

      count += 1
      # break
  return mse_list, nmse_list, psnr_list, ssim_list


# print(img_dict)

telea_mse, telea_nmse, telea_psnr, telea_ssim = cal_mat(ori_files, telea_files, telea_folder)
# print(ns_files)
ns_mse, ns_nmse, ns_psnr, ns_ssim = cal_mat(ori_files, ns_files, ns_folder)
fast_mse, fast_nmse, fast_psnr, fast_ssim = cal_mat(ori_files, fast_files, fast_folder)
best_mse, best_nmse, best_psnr, best_ssim = cal_mat(ori_files, best_files, best_folder)


# TELEA
telea_max_mse, telea_min_mse, telea_mean_mse = max(telea_mse), min(telea_mse), (sum(telea_mse)/len(telea_mse))
telea_max_nmse, telea_min_nmse, telea_mean_nmse = max(telea_nmse), min(telea_nmse), (sum(telea_nmse)/len(telea_nmse))
telea_max_psnr, telea_min_psnr, telea_mean_psnr = max(telea_psnr), min(telea_psnr), (sum(telea_psnr)/len(telea_psnr))
telea_max_ssim, telea_min_ssim, telea_mean_ssim = max(telea_ssim), min(telea_ssim), (sum(telea_ssim)/len(telea_ssim))

print()
print(f"TELEA")
print(f"max mse: {telea_max_mse} | min mse: {telea_min_mse} | mean ssim: {telea_mean_mse}")
print(f"max nmse: {telea_max_nmse} | min nmse: {telea_min_nmse} | mean nmse: {telea_mean_nmse}")
print(f"max psnr: {telea_max_psnr} | min psnr: {telea_min_psnr} | mean psnr: {telea_mean_psnr}")
print(f"max ssim: {telea_max_ssim} | min ssim: {telea_min_ssim} | mean ssim: {telea_mean_ssim}")


# NS
ns_max_mse, ns_min_mse, ns_mean_mse = max(ns_mse), min(ns_mse), (sum(ns_mse)/len(ns_mse))
ns_max_nmse, ns_min_nmse, ns_mean_nmse = max(ns_nmse), min(ns_nmse), (sum(ns_nmse)/len(ns_nmse))
ns_max_psnr, ns_min_psnr, ns_mean_psnr = max(ns_psnr), min(ns_psnr), (sum(ns_psnr)/len(ns_psnr))
ns_max_ssim, ns_min_ssim, ns_mean_ssim = max(ns_ssim), min(ns_ssim), (sum(ns_ssim)/len(ns_ssim))

print()
print(f"NS")
print(f"max mse: {ns_max_mse} | min mse: {ns_min_mse} | mean ssim: {ns_mean_mse}")
print(f"max nmse: {ns_max_nmse} | min nmse: {ns_min_nmse} | mean nmse: {ns_mean_nmse}")
print(f"max psnr: {ns_max_psnr} | min psnr: {ns_min_psnr} | mean psnr: {ns_mean_psnr}")
print(f"max ssim: {ns_max_ssim} | min ssim: {ns_min_ssim} | mean ssim: {ns_mean_ssim}")


# FAST
fast_max_mse, fast_min_mse, fast_mean_mse = max(fast_mse), min(fast_mse), (sum(fast_mse)/len(fast_mse))
fast_max_nmse, fast_min_nmse, fast_mean_nmse = max(fast_nmse), min(fast_nmse), (sum(fast_nmse)/len(fast_nmse))
fast_max_psnr, fast_min_psnr, fast_mean_psnr = max(fast_psnr), min(fast_psnr), (sum(fast_psnr)/len(fast_psnr))
fast_max_ssim, fast_min_ssim, fast_mean_ssim = max(fast_ssim), min(fast_ssim), (sum(fast_ssim)/len(fast_ssim))

print()
print(f"FAST")
print(f"max mse: {fast_max_mse} | min mse: {fast_min_mse} | mean ssim: {fast_mean_mse}")
print(f"max nmse: {fast_max_nmse} | min nmse: {fast_min_nmse} | mean nmse: {fast_mean_nmse}")
print(f"max psnr: {fast_max_psnr} | min psnr: {fast_min_psnr} | mean psnr: {fast_mean_psnr}")
print(f"max ssim: {fast_max_ssim} | min ssim: {fast_min_ssim} | mean ssim: {fast_mean_ssim}")


# BEST
best_max_mse, best_min_mse, best_mean_mse = max(best_mse), min(best_mse), (sum(best_mse)/len(best_mse))
best_max_nmse, best_min_nmse, best_mean_nmse = max(best_nmse), min(best_nmse), (sum(best_nmse)/len(best_nmse))
best_max_psnr, best_min_psnr, best_mean_psnr = max(best_psnr), min(best_psnr), (sum(best_psnr)/len(best_psnr))
best_max_ssim, best_min_ssim, best_mean_ssim = max(best_ssim), min(best_ssim), (sum(best_ssim)/len(best_ssim))

print()
print(f"BEST")
print(f"max mse: {best_max_mse} | min mse: {best_min_mse} | mean ssim: {best_mean_mse}")
print(f"max nmse: {best_max_nmse} | min nmse: {best_min_nmse} | mean nmse: {best_mean_nmse}")
print(f"max psnr: {best_max_psnr} | min psnr: {best_min_psnr} | mean psnr: {best_mean_psnr}")
print(f"max ssim: {best_max_ssim} | min ssim: {best_min_ssim} | mean ssim: {best_mean_ssim}")


# TELEA
# max mse: 44.07716666666666 | min mse: 1.914711111111111 | mean ssim: 20.78150479519256
# max nmse: 0.5238253473957841 | min nmse: 0.016925997136830524 | mean nmse: 0.20703806658563623
# max psnr: 44.22261778709039 | min psnr: 31.688666910990513 | mean psnr: 35.347356285753364
# max ssim: 0.9848403540374433 | min ssim: 0.783699467515716 | mean ssim: 0.8942371562307038

# NS
# max mse: 41.28078518518519 | min mse: 1.4838111111111112 | mean ssim: 17.75131336767231
# max nmse: 0.4774210902418585 | min nmse: 0.013116852183350892 | mean nmse: 0.17685686240864684
# max psnr: 45.329864173709126 | min psnr: 31.97332411629808 | mean psnr: 36.17240383826767
# max ssim: 0.9889984646336629 | min ssim: 0.7922687833555144 | mean ssim: 0.916128681204296

# FAST
# max mse: 20.727974074074073 | min mse: 0.5072666666666666 | mean ssim: 5.615563073338234
# max nmse: 0.19367767847567333 | min nmse: 0.004484224329082779 | mean nmse: 0.05609529410676043
# max psnr: 50.436209763159674 | min psnr: 34.96523504092265 | mean psnr: 41.46586756769089
# max ssim: 0.9968997500939262 | min ssim: 0.946152537061114 | mean ssim: 0.9827801212958521

# BEST
# max mse: 21.629033333333332 | min mse: 0.5005555555555555 | mean ssim: 6.399564925190092
# max nmse: 0.2226420100149014 | min nmse: 0.004424898278906102 | mean nmse: 0.06390040917263598
# max psnr: 50.04912750346968 | min psnr: 34.78043250923988 | mean psnr: 40.724516159263565
# max ssim: 0.9967562666532085 | min ssim: 0.9442620625585036 | mean ssim: 0.9759724884590634