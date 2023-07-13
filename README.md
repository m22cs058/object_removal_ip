# Object Removal using Image Inpainting

Object removal using image inpainting is a computer vision project that involves removing unwanted objects or regions from an image and filling in the resulting gap with plausible content using inpainting techniques. This project uses traditional pre-deep learning algorithms to analyze the surrounding pixels and textures of the target object, then generates a realistic replacement that blends seamlessly into the original image. The objective is to create an aesthetically pleasing image that appears as though the removed object or region was never there. 

<p  align="center">
<img  width="600"  src="https://github.com/m22cs058/object_removal_ip/blob/main/info/cv_demo.gif?raw=true"  alt="Material Bread logo">
</p>
 
## Dataset
A carefully curated subset of 300 images has been selected from the massive ImageNet dataset, which contains millions of labeled images. ImageNet is a large-scale visual recognition database designed to support the development and training of deep learning models. It consists of over 14 million images belonging to more than 21,000 categories. The dataset has played a pivotal role in advancing computer vision research and has been used to develop state-of-the-art image classification algorithms. By using a subset of ImageNet, researchers can efficiently test their models on a smaller scale while still benefiting from the breadth and depth of the full dataset. This dataset is used here to check the performance of different inpainting algorithms.
The dataset is stored in Image_data/Original.
## Setup

Create a conda environment

    conda create -n object-removal python=3.9.13
    conda activate object-removal
 
Install Necessary Libraries

    pip install -r requirements.txt


## Implementation

Activate the environment

    conda activate object-removal

### Object Removal

Save the image file in the working directory as image.jpg and run the command

    python test.py

### Comparison of Different Inpainting Algorithms

After cloning this repository. Go to Image_data/ and delete all folders except Original. Then follow these steps:

    conda activate object-removal
Preprocess the dataset

    python preprocess.py
Apply the various inpainting algorithms and save the output images in Image_data/Final_Image

    python object_remove.py
Print the evaluation metrics

    python eval.py


### Methos used and Result Info

Four methods are used for Image inpainting
    INPAINT_FSR_FAST - work fast and give good result
    INPAINT_FSR_BEST - work slow but give best result
    INPAINT_TELEA - work fast but results are not good as first two methos
    INPAINT_NS - work fast but results are same as "INPAINT_TELEA"


Result:

    Metrics     TELEA       NS          FSRFAST     FSRBEST
    MSE         20.7815     17.7513     5.6155      6.3995
    NMSE        0.2070      0.1768      0.0560      0.0639
    PSNR        35.3473     36.1724     41.4658     40.7245
    SSIM        0.8942      0.9161      0.9827      0.9759
    Time        0.3544      0.3355      1.3708      24.7794
