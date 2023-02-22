---
layout: home
permalink: index.html
repository-name: e16-4yp-Identification-of-Weeds-in-broadcasted-Paddy-fields-using-multispectral-UAV-images
title: Identification of Weeds in broadcasted Paddy fields using multispectral UAV images
---

[comment]: # "This is the standard layout for the project, but you can clean this and use your own template"

# Identification of Weeds in broadcasted Paddy fields using multispectral UAV images

Develop a model to Identify paddy crops and weeds by images taken from UAV (unmanned aerial vehicle) and develop a desktop application as user interface.

#### Team

- E/16/319, Rathnayaka R.P.V.N, [e16319@eng.pdn.ac.lk](mailto:name@email.com)
- E/16/320, Rathnayake E.W.S.P, [e16320@eng.pdn.ac.lk](mailto:name@email.com)
- E/16/377, Vindula I.B.S, [e16377@eng.pdn.ac.lk](mailto:name@email.com)

#### Supervisors

- Dr. Damayanthi Herath, [damayanthiherath@eng.pdn.ac.lk](mailto:name@eng.pdn.ac.lk)
- Dr. Sachith Seneviratne, [sachith.seneviratne@unimelb.edu.au](mailto:name@eng.pdn.ac.lk)
- Dr. Nuwan De Silva, [sssnuwanp@yahoo.com](mailto:name@eng.pdn.ac.lk)

#### Table of content

1. [Abstract](#abstract)
2. [Related works](#related-works)
3. [Methodology](#methodology)
4. [Experiment Setup and Implementation](#experiment-setup-and-implementation)
5. [Results and Analysis](#results-and-analysis)
6. [Conclusion](#conclusion)
7. [Publications](#publications)
8. [Links](#links)

---




## Abstract
Unwanted plants called weeds destroy the crops in an agricultural field by growing there. Occasionally aggressive weeds dramatically reduce agricultural yield, costing farmers a great deal of money. Using chemical pesticides to control weeds is a common practice. It is well recognized that these pesticides have a negative impact on the ecosystem. Following the Site-Specific Weed Management guidelines is one strategy to reduce the negative impacts of herbicides (SSWM). The proper herbicide should be applied on agricultural land in the right quantity as part of site-specific weed management. In order to categorize two different types of weeds in paddy fields, sedges and broadleaved weeds, this research investigates a semantic segmentation approach. The segmentation of paddy crop and two different types of weeds was done using three semantic segmentation models, including Residual Network (ResNet)  , Network (PSPNet), and UNet. With an accuracy rate of over 45%, only considering RGB images , results have been attained. We think this can be used to advise farmers on the best herbicide to use, promoting sustainable agriculture and site-specific weed control.

## Related works
In recent years, researchers have utilized neural network learning methods in computer vision technology to segment weeds using UAV images. This section previews the work of researchers in this field and summarizes the key findings of the literature review.


 	Weed detection challenges are varied because of the different types of weeds and crops. Considered studies were based on different types of crops such as beets, peaches, potatoes, rice, and sunflower. It is challenging to distinguish a crop when it resembles nearby weeds. This problem mainly affects rice fields because the rice plants are grown very close to each other. Another point is that the color of some weeds is similar to that of the crops. Due to these reasons, the accuracy and stability are sometimes poor for weed detection in the field. Moreover, the farming layout also varies between different locations. For example, the farming layout is different from the other areas on the hillsides. Sometimes main plant and the weeds stay closer to the field. Therefore, according to the facts, the task of weed detection is a diverse problem. 


Most studies have used CNN for weed detection when considering neural network approaches to segment the weed in the crop.ResNet-based networks, Segnet architecture, CNN- Yolo3, EffienNet, and different types of CNN have been used. An overall accuracy of 99.4 was given by the model, which used a CNN LVQ, Convolutional Neural Network (CNN) model, and Learning Vector Quantization (LVQ) algorithm) model. Based on the qualities of adaptation and topology, they created a CNN model that assimilates the Learning Vector Quantization (LVQ) technique. All the above methods are different types of CNN models with some modifications. 


Various different datasets were used during the studies. Currently, only very few accessible public datasets are avail- able. As a result, most of the research has been done using self-built data sets. The portability of the technique is poor in varied growth phases, lighting conditions, and actual field backgrounds, even though the significant component of some datasets is the same crop. In addition, because the dataset from which the algorithm was created was built on a separate premise, relevant assessment indicators are not comparable. Hence, determining the actual performance is challenging.


Through the literature review, it was also discovered that there are no convenient, user-friendly free tools to segment weeds in the crops. Weeds have diverse natural characteristics, including a vast range of species, widespread distribution, various leaf sizes and shapes, irregular growth, and different textural aspects. In addition, most weeds have small, variable- appearing buds with a high germination density. As a result, performing precise statistics is challenging. Some of the other factors which are affecting weed detection are as follows. 


Effects of varying lighting conditions. The shade provided by the plant canopy and the sun’s angle will impact the colour of the vegetation when the lighting is varied. Some researchers have employed the Otsu algorithm and the ultra-green index to address issues brought on by ambient light. 


The impact of various growth phases. In changing seasons or phases of growth and development, most plants alter their leaves’ morphology, texture, and spectral characteristics. 
Other facts in weed detection, Hardware, algorithm complexity, and plant density are some factors that restrict de- tection’s speed or accuracy. Therefore, reliable weed detection and quick picture processing remain as significant obstacles. Additionally, applying existing techniques in practical ap- plication is difficult because the major components such as segmentation algorithms and crops have complex data struc- tures. Even though progress has been made, more efforts are needed for further research in this area. Farmers and growers can use the results of these studies. It can help them achieve a clearer idea about what kind of weeds exist at different times and conditions, thus helping them to grow crops better with less crop damage or loss of yield or quality compared to others using the same crop varieties and growing conditions. In summary, developing a robust weed inspection method using Computer Vision is difficult. Therefore, the researchers who worked on this issue in recent years developed methods for weed segmentation using UAV images based on Neural networks. The goal is to develop tools that can be used effec- tively by farmers and growers. However, further research will be needed before they can provide reliable and precise ways to inspect weeds. The major limitation in all previous studies was the lack of publicly available datasets for training and testing. Therefore, there must be more publicly available data sets so that researchers can easily compare their results with those obtained by other research groups who may have come up with better solutions based on the different available methods and networks. The methods mentioned in this review are an important step toward improving the accuracy and robustness of weed detection. Additionally, the results provided some insight into approaches that farmers could use to understand how weeds affect various crops differently.


## Methodology

In the following section, we will offer an overview of the process of acquiring and annotating the data. Subsequently, we will present the machine learning models used in our study, along with the methods utilized for hyperparameter optimization and the evaluation metrics employed. Finally, we will detail the experimental setup.

####Data acquisition and annotation
Dr. Nuwan De Silva and K.M.K.I.Rathnayake conducted the research that yielded the dataset used in this study. A multi-spectral camera was used to photograph the entire field every three days. In total, 2725 pictures were obtained, each with six spectral bands. Picture processing was carried out with the help of the trial version of the software PIX4Dfields® (PIX4D. (n.d.). PIX4Dfields. https://www.pix4d.com/products/pix4dfields). A UAV was used to fly the camera at an altitude and pattern suitable for taking photos with the required resolution. Camera settings and parameters were tweaked to provide high-quality photos with the required spectral resolution. The photographs were taken at various times of day and in various weather conditions to create a diverse dataset for model training.

This study used multispectral photos with six spectral bands: RGB, blue, green, red, red edge, and near-IR. Each spectral band contains data on several elements of the vegetation, such as chlorophyll concentration, water stress, and biomass.
The RGB band, which is made up of red, green, and blue bands, is frequently used to represent images for human interpretation. The blue band is water content sensitive and can be used to identify regions with high moisture levels. The green band is chlorophyll-sensitive and can be used to differentiate between healthy and bad flora.
The red band is vegetation density sensitive and can be used to estimate biomass. The red edge band detects small changes in vegetation structure and is sensitive to chlorophyll concentration and leaf area index. Finally, the near-IR band can be utilized to measure vegetation health and biomass because it is sensitive to plant water content and biomass.
The dataset utilized in this study provides a rich supply of information for creating a deep learning model for weed detection and segmentation in precision agriculture by employing a multi-spectral camera and recording these six spectral bands.
The earlier approach of employing vegetation indices for weed detection and segmentation had a major flaw: it was inaccurate. To solve this issue, we chose to improve the accuracy by using a segmentation model.
Due to the time-consuming nature of the annotation process, we chose 109 photos for this investigation. We concentrated our efforts on refining the segmentation model and optimizing the training process by using a smaller number of photos. Furthermore, because each of these photos had six spectral bands, the same mask could be used for all of the spectral bands, simplifying the annotating process.The selected photos were acquired on successive weeks because there was expected to be some variance in the vegetation cover, making the model more robust. The goal was to construct a dataset that captured the temporal variations in the field and provided a varied range of photos for the model to learn from by selecting images from consecutive weeks.
The goal of the data selection procedure was to provide a concentrated and diversified dataset that would allow us to build an accurate and robust deep learning model.
The label-studioB(Label Studio (2021). Label Studio: An Open-Source Data Annotation Tool. Retrieved from https://labelstud.io/.), python library was used for the annotation process, but we ran into some issues with the brush type annotation, which made it impossible to precisely identify the margins of the weed patches. Furthermore, we had to annotate on everything, including the vegetation and soil, which took time.
The CVAT(CVAT (Computer Vision Annotation Tool) version 2.0.0 was used to annotate the dataset (https://github.com/openvinotoolkit/cvat/releases/tag/v2.0.0). CVAT is an open-source web-based tool that enables collaborative annotation of images and videos for computer vision tasks.) tool was chosen to address these difficulties. CVAT is a free and open-source annotation tool that works with a variety of input and output formats. One of CVAT's major advantages is that it is web-based and professionally developed, making it easier to use and more trustworthy.
The weed patches were annotated more correctly and effectively using CVAT. The polygon-based annotation functionality allowed us to construct boundaries with simpler shortcuts, which cut down on annotation time. Furthermore, the focus was solely on the weed patches, ignoring the plant and soil, which reduced annotation time and made the annotation process more efficient.
Overall, the annotation process was critical in constructing an accurate and resilient deep learning model for precision agricultural weed detection and segmentation.
We employed three key labels in our annotation process: Weed label, Crop label, and Additional marijuana types label. The Weed label was used to annotate the three target weed species present in the image: Echinochloa crus-galli, Ischaemum rugosum, and Cyperus iria. The Crop label was used to annotate the image's ten rice kinds.
Other weed types is an additional label used to annotate common weed types such as grass that are not one of the target weed species. Furthermore, the parts of the photograph that were devoid of crops or weeds were designated as Ground.
Overall, the labeling approach enabled precise identification and annotation of the target weed species and crops in the photos.
It was tough to annotate because the boundaries of the weed patches were not very distinct when zooming in to comment. This made it difficult to precisely identify the weed patch boundaries.
To address this issue, a peer-review method for picture annotations was established. Several annotators assessed each image to confirm the correctness and integrity of the annotations. This aided in identifying and correcting any differences in the annotations, thereby improving the dataset's overall correctness.
Using the CVAT annotation tool, 41 of the 109 selected photos were tagged as rice alone and 40 images as weed solely. The remaining 28 photos were not used in the training procedure.
The annotated dataset was divided into three sections for training, validation, and testing. The dataset was used for training 70% of the time, validation 15% of the time, and testing 15% of the time. To ensure that each set had a representative sample from each label category, a randomization approach was enabled.
Given the constrained dataset, the UNET model was chosen. A deep learning model for weed recognition and segmentation was developed and trained using the annotated data. We employed an Adam optimizer and a binary cross-entropy loss function in a UNET model. The model was trained for 100 epochs on a GPU with a batch size of 4.
## Experiment Setup and Implementation

In this project our object was  to segment the weed in the paddy field.Image segmentation is the process of dividing an image into multiple segments or regions based on certain characteristics such as color, texture, or shape. The goal of image segmentation is to extract meaningful and relevant information from an image, making it easier to analyze and understand. Image segmentation is commonly used in computer vision and image processing applications, including object recognition, medical imaging, and autonomous driving.
Semantic segmentation suite was used as our project environment.This repository serves as a Semantic Segmentation Suite. The goal is to easily be able to implement, train, and test new Semantic Segmentation models[1].
Throughout the literature review we found that Unet is going to be an ideal for our scenario for the main reason.Which is since images were annotated manually, there are around 100 images.Unet is performing better with small  datasets in the previous cases.There fore we mainly targeted for uNet and two other modules have trained and evaluated as well.Those are ResNet and PSPNet.For all model implementation python tensorflow is used. 
ResNet, short for "Residual Network," is a deep neural network architecture developed by researchers at Microsoft Research in 2015. It is a type of convolutional neural network (CNN) that is specifically designed for image classification tasks.ResNet addresses a common problem in deep neural networks known as the "vanishing gradient problem." This problem occurs when gradients become very small as they are propagated through many layers, making it difficult to train very deep networks. ResNet solves this problem by introducing residual connections, which allow information to bypass some layers and flow directly to later layers in the network.
PSPNet (Pyramid Scene Parsing Network) is a deep neural network architecture for semantic image segmentation. It was proposed by researchers at the University of California, Berkeley in 2017.
The main idea behind PSPNet is to use a pyramid pooling module to capture contextual information at multiple scales. This module takes as input a feature map produced by a convolutional neural network and divides it into regions of different sizes. For each region, it computes the average pooling operation to produce a feature vector, which is then concatenated with the original feature map. This process is repeated for multiple scales, creating a pyramid of feature maps with different levels of spatial detail.
However Unet performed better with the dataset.
U-Net is a convolutional neural network architecture that is commonly used for image segmentation tasks. It was first introduced in 2015 by Olaf Ronneberger, Philipp Fischer, and Thomas Brox, and has since become a popular model for various medical image analysis tasks such as tumor detection, cell segmentation, and brain tissue segmentation.
The U-Net architecture consists of an encoder network and a decoder network. The encoder network is composed of a series of convolutional and pooling layers that downsample the input image, extracting features and reducing the spatial dimensionality of the data. The decoder network then up samples the features and gradually recovers the original spatial resolution of the input image.
The key innovation of U-Net is the skip connections between the encoder and decoder networks. These connections allow the decoder network to access the features extracted by the encoder network at various scales, enabling the model to perform accurate segmentation even for small and complex objects.
Overall, U-Net has been shown to be effective for a variety of image segmentation tasks, particularly in the medical imaging domain where it has been used to achieve state-of-the-art performance on several benchmarks
The input to the UNet model is an image of size H x W x C, where H is the height, W is the width, and C is the number of channels.In here RGB channels.The contracting path is a sequence of convolutional and max-pooling layers that are designed to extract high-level features from the input image. Each convolutional layer is followed by a ReLU activation function, which introduces non-linearity into the model. The number of filters in each convolutional layer is doubled at each downsampling step, which allows the model to learn increasingly complex representations of the input image.The expanding path is a sequence of convolutional and upsampling layers that are designed to reconstruct the segmentation map from the extracted features. Each convolutional layer is followed by a ReLU activation function, and the number of filters is halved at each upsampling step. The upsampling operation is typically performed using either bilinear interpolation or transposed convolution.The output of the UNet model is a segmentation map of size H x W x N, where N is the number of classes. The segmentation map is generated by applying a softmax activation function to the output of the last convolutional layer in the decoder. Each pixel in the segmentation map corresponds to a class label, and the value of the pixel represents the probability that the pixel belongs to that class.
We have implemented the Unet model using python.Tensorflow is used.TensorFlow is an open-source library for numerical computation and machine learning developed by the Google Brain team. It allows users to create and deploy machine learning models for various tasks, such as image and speech recognition, natural language processing, and robotics.
Google colab was used to train and test the models.Google Colab, short for Google Colaboratory, is a free online platform that allows users to run and execute Python code in a web browser. It is a product of Google Research and is intended to provide a platform for machine learning education and research.
We have trained models for different epochs for 20, 50, 100 and 200.In each time the model was evaluated.and also model was evaluated and different learning rates.To improve results furthermore,  adams optimizer is used.Adam (Adaptive Moment Estimation) optimizer is a popular gradient-based optimization algorithm used in machine learning and deep learning. It is an extension of the stochastic gradient descent (SGD) optimizer that is designed to adaptively adjust the learning rate based on the historical gradients.Using adams optimizer precision was improved furthermore.
Data augmentation was used as well.Images randomly flipping horizontally, randomly rotating and changing brightness were the data augmentation techniques we have used. 

## Results and Analysis
In this section, we first provide an overview of our results. Then, we analyze the predictions of our best performing weed segmentation model in more detail. Finally, we discuss our results.
####Results overview



  

![](https://lh3.googleusercontent.com/qwICzHuGfSyYasW_CeO1-uKdN1QWjKO3FMapGRPzaTD19DKFYrT49KxzUzJQ-5VtZOZQQ2QY7QvBx4ZI5gc5ZctkyhbGsS8Ll-BdNM96nByc8Z_u-dZtmB9rHWlfYTSgii06spzumVjW1vyiIEJl--U)

First, we evaluated the architecture choice of the segmentation model in combination with three architecures. In [Table 4](https://www.sciencedirect.com/science/article/pii/S0168169922006962#t0020), show the results on the evaluation. From that observed that U net gives the highest value than other two architectures Resnet and pspNet.

![](https://lh4.googleusercontent.com/_COPPixadqbhRtGyXp9v-JVdCuZD1cc3QPMy6RG78SQegEwgZUQ67ZgA84p2orjpD9UskqtLQOOso0fe0GRRyHGqX-Etaae6jc2FBsAt_erCVmqqye6muZur8sTZyzXyqJAJMniUuu_HK9qUN59W1PE "Change in Average Precision with No of Epochs")

After fguring out that Unet architecture perform the best among three architectures moved with Unet architecture and changed the othe parameters. As per the table tested the model for different epochs. From the result observed that average precision increases with the number of epochs in the range of 0 to 100 but after 100 it seems stable around 40 percent.

![](https://lh6.googleusercontent.com/7ETucIoaG29KeU5-z0pBxWqoYV4BVRHkyBjl_aJulzhG7T9Ja-yf_2HzlwIc1cpYzRChoFbhtZ2vqvnbtpe0XxK6xRRgUQ1GeXTdsLabjjWV9x93-mfir_QCm6dUrPKNr5zcamLv2tPfZ8-4Z_uk0gE "Change in Average Precision with No of Epochs")  
  

Then the average accuracy was calculated by changing the learning rate for U net architecture with 100 epochs. From that got the result as show in figure , From that we observed that average precision increases with the learning rate and at 0.01 it comes to maximum and then reduce. Therefore the 0.01 was taken as learning rate to move forward.

In general, the deepest feature extractor ResNet-101 performed worse for all segmentation models. The best performing feature extractor was dependent on the selected segmentation architecture. We observed minor differences in the objective value when comparing different feature extractors while using the same semantic segmentation model. In addition, there were minor differences in the objective values between different segmentation models UNet and DeepLabv3+. When using dilated convolutions in the feature extractor with the FCN-8s architecture (no concatenation of intermediate feature maps), the objective values had only minor differences compared to UNet and DeepLabv3+. This indicates a minor effect of the selection of the semantic segmentation model.

Both, FCN-16s and FCN-8s combined multiple output layers together and then upsampled the result. Here we observed a higher objective value in general, meaning worse predictions. In addition, the standard deviation between the trials is approximately 28 times higher for FCN-16s and FCN-8s compared to FCN-32s. This holds true for all feature extractors examined in this study.

The best combination UNet with ResNet-34 was more robust during hyperparameter optimization, indicated by a small standard deviation of 0.0022.
Furthermore,as the best performing model Unet was selected and the model was improved using adams optimizer.Model was trained for 200 epochs to get more accurate results.

In this paper, we present a DL-based approach to weed detection and segmentation in rice fields under real-world conditions using UAV images. Our data set was collected in Bathalagoda rice research and development institute.Since we have a limited number of images data augmentation is used as well.After Unet was selected as a best performing model, we furthermore improved it using adams optimizer.But all this time we have only considered RGB images.We have archived the average accuracy of 45%.
Weed and paddy in the field are so close together and they have almost the same color as well.Therefore segmentation was challenging.


In our research, we identified a discrepancy between the evaluation metrics and a qualitative assessment of the predictions we made. We found that the confusion matrix treated all pixels equally, which may not be useful for identifying weeds. Upon further examination of the model's predictions, we discovered that most of the incorrectly classified pixels were located at the edges of the plants. It was crucial to accurately classify crop and weed pixels as this was the primary challenge we faced. Moreover, the presence of rare but large weeds and the high percentage of the background class had an adverse impact on the results obtained from the confusion matrix since the misclassification of these weeds had a more significant impact. In our dataset, the background class accounted for 68% of all pixels, which had a notable effect on both per-pixel metrics and the confusion matrix.

The prediction errors were partly attributed to the process of generating patches. Specifically, when a small portion of a plant was present on the patch boundary, it resulted in misclassification. This issue occurred because the network was trained mainly on intact plants and might have difficulty recognizing a partially cut plant. Additionally, the background pixels were frequently predicted near the patch borders. Although these errors were not severe, they were more likely to occur for larger plants that were split during the patch generation process. A potential solution to this issue could be to use a sliding window approach and create patches with overlap.

## Conclusion
This study introduces a machine learning framework capable of early detection and segmentation of weeds in degraded drone images encountered in real-world settings. The model exhibits mid-level segmentation abilities, enabling it to identify intra-row weeds and overlapping plants in captured images. Our qualitative analysis of the model's predictions shows that it accurately detects the general form of most plants, with minor issues at plant borders. Although plant borders are not crucial for downstream applications focused on site-specific weed management, our model performs well in detecting and localizing weeds, which can help improve agricultural productivity.
Additionally, the average precision is 45% only considering RGB images.Therefore the  Unet model should be updated to consider other multispectral images  as well.
This study presents the first dataset for weed detection, which can serve as a foundation for future research in this field. Furthermore, we have made our proposed method's code publicly available to ensure the reproducibility of our results and enable comparison with future studies.


## Publications
[//]: # "Note: Uncomment each once you uploaded the files to the repository"

<!-- 1. [Semester 7 report](./) -->
<!-- 2. [Semester 7 slides](./) -->
<!-- 3. [Semester 8 report](./) -->
<!-- 4. [Semester 8 slides](./) -->
<!-- 5. Author 1, Author 2 and Author 3 "Research paper title" (2021). [PDF](./). -->


## Links

[//]: # ( NOTE: EDIT THIS LINKS WITH YOUR REPO DETAILS )

- [Project Repository](https://github.com/cepdnaclk/repository-name)
- [Project Page](https://cepdnaclk.github.io/repository-name)
- [Department of Computer Engineering](http://www.ce.pdn.ac.lk/)
- [University of Peradeniya](https://eng.pdn.ac.lk/)

[//]: # "Please refer this to learn more about Markdown syntax"
[//]: # "https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet"
