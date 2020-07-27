# Brain-Tumor-Segmentation-Project

 Brain tumor segmentation is an important task in healthcare AI. Early diagnosis of brain tumors has an important role in improving treatment possibilities and increases the survival rate of the patients. Manual segmentation of the brain tumors for cancer diagnosis, from large amount of MRI images generated in clinical routine, is a difficult and time consuming task. There is a need for automatic brain tumor image segmentation.
 
 # Dataset
 
 Source of the dataset: https://figshare.com/articles/brain_tumor_dataset/1512427
 
 About the dataset: This brain tumor dataset containing 3064 T1
from 233 patients with three kinds of brain tumor: Meningioma (708 slices), 
Glioma (1426 slices), and pituitary tumor (930 slices). 

This data is organized in matlab data format (.mat file). Each file stores a struct
containing the following fields for an image:

cjdata.label: 1 for meningioma, 2 for glioma, 3 for pituitary tumor

cjdata.PID: patient ID

cjdata.image: image data

cjdata.tumorBorder: a vector storing the coordinates of discrete points on tumor border.We can use it to generate
		binary image of tumor mask.

cjdata.tumorMask: a binary image with 1s indicating tumor region

### Note: 

These 15 images have different size than others. (955, 956, 957, 1070, 1071, 1072, 1073, 1074, 1075 ,1076, 1203, 1204, 1205, 1206, 1207) It is necessary to omit them or find a way to handle tensor size problem. In one model I didn't use them, in the other one I read the data as numpy arrays and arranged the shapes of the arrays. 

# Method:

 In this project I am making classification and tumor mask with transfer learning in Pytorch. 
 
 In one model, I used both the structure and weights from imagenet ResNext(resnext50_32x4d). 
 
 In the second one, I used a pre-trained model Models Genesis from Arizona State University. The important point in this model is training is done on health images.(X-rays, magnetic resonance images(MRI), and Computed tomography(CT)). 
 
 For more detailed information  https://github.com/MrGiovanni/ModelsGenesis
 
 # Result: 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
