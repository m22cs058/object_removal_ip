# OBJECT REMOVAL USING IMAGE INPAINTING

- import required libraries from requirements.txt files
- now in dataset 3 folders are present 
- Final_image for output, Original for original dataset, Pre_image for preprocess image

preprocess.py
- it prepocess the image for reshape and rename and store in Pre_image folder

object_remove.py
- it has four method for object remove
- first it add some mask unwanted object in Pre_image folder image then apply different method for object remove and store different method output in Final_image methodwise folder

Eval.py
- it is for evaluation metcis
- it has four method to evaluate image namely MSE, NMSE, PSNR, SSIM
- outcome of eval.py
  TELEA
  max mse: 44.07716666666666 | min mse: 1.914711111111111 | mean ssim: 20.78150479519256
  max nmse: 0.5238253473957841 | min nmse: 0.016925997136830524 | mean nmse: 0.20703806658563623
  max psnr: 44.22261778709039 | min psnr: 31.688666910990513 | mean psnr: 35.347356285753364
  max ssim: 0.9848403540374433 | min ssim: 0.783699467515716 | mean ssim: 0.8942371562307038

  NS
  max mse: 41.28078518518519 | min mse: 1.4838111111111112 | mean ssim: 17.75131336767231
  max nmse: 0.4774210902418585 | min nmse: 0.013116852183350892 | mean nmse: 0.17685686240864684
  max psnr: 45.329864173709126 | min psnr: 31.97332411629808 | mean psnr: 36.17240383826767
  max ssim: 0.9889984646336629 | min ssim: 0.7922687833555144 | mean ssim: 0.916128681204296

  FAST
  max mse: 20.727974074074073 | min mse: 0.5072666666666666 | mean ssim: 5.615563073338234
  max nmse: 0.19367767847567333 | min nmse: 0.004484224329082779 | mean nmse: 0.05609529410676043
  max psnr: 50.436209763159674 | min psnr: 34.96523504092265 | mean psnr: 41.46586756769089
  max ssim: 0.9968997500939262 | min ssim: 0.946152537061114 | mean ssim: 0.9827801212958521

  BEST
  max mse: 21.629033333333332 | min mse: 0.5005555555555555 | mean ssim: 6.399564925190092
  max nmse: 0.2226420100149014 | min nmse: 0.004424898278906102 | mean nmse: 0.06390040917263598
  max psnr: 50.04912750346968 | min psnr: 34.78043250923988 | mean psnr: 40.724516159263565
  max ssim: 0.9967562666532085 | min ssim: 0.9442620625585036 | mean ssim: 0.9759724884590634


test.py
- Using this file we can manually remove the object with a brush
- After executing a window with the image will be opened.
- Using cursor select the area that needs to be removed.
- After selecting the area press the key 'i', 'j', 'k', 'l' as per the algorithm we need to execute
- 'i' = INPAINT_FSR_FAST, 'j' = INPAINT_FSR_BEST, 'k' = INPAINT_TELEA, 'l' = INPAINT_NS
