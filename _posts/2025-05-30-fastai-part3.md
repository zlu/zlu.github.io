---
layout: post
title: "Deep Dive into MNIST Dataset - Fastai Part 3"
date: 2025-05-30
comments: true
categories: 
  - machine learning
tags:
  - python
  - artificial intelligence
  - machine learning
  - fastai
  - PyTorch
description: "Deep Dive into MNIST Dataset - Fastai Part 3"
---

In fastai part 1 we looked at how to categorize MNIST dataset.  In this tutorial, we will dive a bit deeper to see what happens under the hood.  First, we will take a deeper look at the MNIST dataset.

## Data Exploration


```python
# code from part 1
import torch
import random
from fastai.vision.all import *

# download the simple MNIST data set (not the full dataset yet, which we will do later)
path = untar_data(URLs.MNIST_SAMPLE)
train_path = path/'train'
img_files = list((train_path/'7').ls())
img = PILImage.create(img_files[0])
img.show();
```


    
![png](/assets/images/uploads/fastai-part3_files/fastai-part3_1_0.png)
    



```python
print(f"Image mode is: {img.mode}")
```

    Image mode is: RGB


MNIST images are grayscale (single channel).  When created by PILImage, they are converted to RGB, likely for model compatibility.  This is why as the print out shown above, the mode of the image is now RGB.  Grayscale image would have shown a "L" (1 channel), as opposed to RGB (3 channels).  If you want to keep the image as grayscale, you can use the following code:
```python
img = PILImage.create(img_files[0])
img.show();
print(f"Image mode is: {img.mode}")
```


```python
arr = array(img)
print(arr.shape)
```

    (28, 28, 3)


(28, 28, 3) is a 3D NumPy array of height, width, and channel.  It has a size of 28x28 pixels.  '3' means the RGB channels and they are likely have the same values as the image is grayscale.


```python
print(np.unique(arr))
```

    [  0   9  23  24  34  38  44  46  47  64  69  71  76  93  99 104 107 109
     111 115 128 137 138 139 145 146 149 151 154 161 168 174 176 180 184 185
     207 208 214 215 221 230 231 240 244 245 251 253 254 255]


`np.unique(arr)` gives a range of values showing that these pixel values span from 0 to 255.  '0' is black and '255' is white.  The values in between are shades of gray.


```python
np.all(arr[:, :, 0] == arr[:, :, 1]) and np.all(arr[:, :, 1] == arr[:, :, 2])
```




    np.True_



The code above checks if the pixel values are the same for all three channels.  Since the image is grayscale, the values are the same hence np.True


```python
img_t = tensor(arr[:, :, 0])
print(img_t.shape)
df = pd.DataFrame(img_t)
df.style.set_properties(**{'font-size':'6pt'}).background_gradient('Greys')
```

    torch.Size([28, 28])





<style type="text/css">
#T_8a8f1_row0_col0, #T_8a8f1_row0_col1, #T_8a8f1_row0_col2, #T_8a8f1_row0_col3, #T_8a8f1_row0_col4, #T_8a8f1_row0_col5, #T_8a8f1_row0_col6, #T_8a8f1_row0_col7, #T_8a8f1_row0_col8, #T_8a8f1_row0_col9, #T_8a8f1_row0_col10, #T_8a8f1_row0_col11, #T_8a8f1_row0_col12, #T_8a8f1_row0_col13, #T_8a8f1_row0_col14, #T_8a8f1_row0_col15, #T_8a8f1_row0_col16, #T_8a8f1_row0_col17, #T_8a8f1_row0_col18, #T_8a8f1_row0_col19, #T_8a8f1_row0_col20, #T_8a8f1_row0_col21, #T_8a8f1_row0_col22, #T_8a8f1_row0_col23, #T_8a8f1_row0_col24, #T_8a8f1_row0_col25, #T_8a8f1_row0_col26, #T_8a8f1_row0_col27, #T_8a8f1_row1_col0, #T_8a8f1_row1_col1, #T_8a8f1_row1_col2, #T_8a8f1_row1_col3, #T_8a8f1_row1_col4, #T_8a8f1_row1_col5, #T_8a8f1_row1_col6, #T_8a8f1_row1_col7, #T_8a8f1_row1_col8, #T_8a8f1_row1_col9, #T_8a8f1_row1_col10, #T_8a8f1_row1_col11, #T_8a8f1_row1_col12, #T_8a8f1_row1_col13, #T_8a8f1_row1_col14, #T_8a8f1_row1_col15, #T_8a8f1_row1_col16, #T_8a8f1_row1_col17, #T_8a8f1_row1_col18, #T_8a8f1_row1_col19, #T_8a8f1_row1_col20, #T_8a8f1_row1_col21, #T_8a8f1_row1_col22, #T_8a8f1_row1_col23, #T_8a8f1_row1_col24, #T_8a8f1_row1_col25, #T_8a8f1_row1_col26, #T_8a8f1_row1_col27, #T_8a8f1_row2_col0, #T_8a8f1_row2_col1, #T_8a8f1_row2_col2, #T_8a8f1_row2_col3, #T_8a8f1_row2_col4, #T_8a8f1_row2_col5, #T_8a8f1_row2_col6, #T_8a8f1_row2_col7, #T_8a8f1_row2_col8, #T_8a8f1_row2_col9, #T_8a8f1_row2_col10, #T_8a8f1_row2_col11, #T_8a8f1_row2_col12, #T_8a8f1_row2_col13, #T_8a8f1_row2_col14, #T_8a8f1_row2_col15, #T_8a8f1_row2_col16, #T_8a8f1_row2_col17, #T_8a8f1_row2_col18, #T_8a8f1_row2_col19, #T_8a8f1_row2_col20, #T_8a8f1_row2_col21, #T_8a8f1_row2_col22, #T_8a8f1_row2_col23, #T_8a8f1_row2_col24, #T_8a8f1_row2_col25, #T_8a8f1_row2_col26, #T_8a8f1_row2_col27, #T_8a8f1_row3_col0, #T_8a8f1_row3_col1, #T_8a8f1_row3_col2, #T_8a8f1_row3_col3, #T_8a8f1_row3_col4, #T_8a8f1_row3_col5, #T_8a8f1_row3_col6, #T_8a8f1_row3_col7, #T_8a8f1_row3_col8, #T_8a8f1_row3_col9, #T_8a8f1_row3_col10, #T_8a8f1_row3_col11, #T_8a8f1_row3_col12, #T_8a8f1_row3_col13, #T_8a8f1_row3_col14, #T_8a8f1_row3_col15, #T_8a8f1_row3_col16, #T_8a8f1_row3_col17, #T_8a8f1_row3_col18, #T_8a8f1_row3_col19, #T_8a8f1_row3_col20, #T_8a8f1_row3_col21, #T_8a8f1_row3_col22, #T_8a8f1_row3_col23, #T_8a8f1_row3_col24, #T_8a8f1_row3_col25, #T_8a8f1_row3_col26, #T_8a8f1_row3_col27, #T_8a8f1_row4_col0, #T_8a8f1_row4_col1, #T_8a8f1_row4_col2, #T_8a8f1_row4_col3, #T_8a8f1_row4_col4, #T_8a8f1_row4_col5, #T_8a8f1_row4_col6, #T_8a8f1_row4_col7, #T_8a8f1_row4_col8, #T_8a8f1_row4_col9, #T_8a8f1_row4_col10, #T_8a8f1_row4_col11, #T_8a8f1_row4_col12, #T_8a8f1_row4_col13, #T_8a8f1_row4_col14, #T_8a8f1_row4_col15, #T_8a8f1_row4_col16, #T_8a8f1_row4_col17, #T_8a8f1_row4_col18, #T_8a8f1_row4_col19, #T_8a8f1_row4_col20, #T_8a8f1_row4_col21, #T_8a8f1_row4_col22, #T_8a8f1_row4_col23, #T_8a8f1_row4_col24, #T_8a8f1_row4_col25, #T_8a8f1_row4_col26, #T_8a8f1_row4_col27, #T_8a8f1_row5_col0, #T_8a8f1_row5_col1, #T_8a8f1_row5_col2, #T_8a8f1_row5_col3, #T_8a8f1_row5_col4, #T_8a8f1_row5_col5, #T_8a8f1_row5_col6, #T_8a8f1_row5_col7, #T_8a8f1_row5_col8, #T_8a8f1_row5_col9, #T_8a8f1_row5_col10, #T_8a8f1_row5_col11, #T_8a8f1_row5_col12, #T_8a8f1_row5_col13, #T_8a8f1_row5_col14, #T_8a8f1_row5_col15, #T_8a8f1_row5_col16, #T_8a8f1_row5_col17, #T_8a8f1_row5_col18, #T_8a8f1_row5_col19, #T_8a8f1_row5_col20, #T_8a8f1_row5_col21, #T_8a8f1_row5_col22, #T_8a8f1_row5_col23, #T_8a8f1_row5_col24, #T_8a8f1_row5_col25, #T_8a8f1_row5_col26, #T_8a8f1_row5_col27, #T_8a8f1_row6_col0, #T_8a8f1_row6_col1, #T_8a8f1_row6_col2, #T_8a8f1_row6_col3, #T_8a8f1_row6_col4, #T_8a8f1_row6_col5, #T_8a8f1_row6_col6, #T_8a8f1_row6_col7, #T_8a8f1_row6_col8, #T_8a8f1_row6_col9, #T_8a8f1_row6_col10, #T_8a8f1_row6_col11, #T_8a8f1_row6_col12, #T_8a8f1_row6_col13, #T_8a8f1_row6_col14, #T_8a8f1_row6_col15, #T_8a8f1_row6_col16, #T_8a8f1_row6_col17, #T_8a8f1_row6_col18, #T_8a8f1_row6_col19, #T_8a8f1_row6_col20, #T_8a8f1_row6_col21, #T_8a8f1_row6_col22, #T_8a8f1_row6_col23, #T_8a8f1_row6_col24, #T_8a8f1_row6_col25, #T_8a8f1_row6_col26, #T_8a8f1_row6_col27, #T_8a8f1_row7_col0, #T_8a8f1_row7_col1, #T_8a8f1_row7_col2, #T_8a8f1_row7_col3, #T_8a8f1_row7_col4, #T_8a8f1_row7_col5, #T_8a8f1_row7_col6, #T_8a8f1_row7_col7, #T_8a8f1_row7_col8, #T_8a8f1_row7_col9, #T_8a8f1_row7_col10, #T_8a8f1_row7_col11, #T_8a8f1_row7_col12, #T_8a8f1_row7_col13, #T_8a8f1_row7_col20, #T_8a8f1_row7_col21, #T_8a8f1_row7_col22, #T_8a8f1_row7_col23, #T_8a8f1_row7_col24, #T_8a8f1_row7_col25, #T_8a8f1_row7_col26, #T_8a8f1_row7_col27, #T_8a8f1_row8_col0, #T_8a8f1_row8_col1, #T_8a8f1_row8_col20, #T_8a8f1_row8_col21, #T_8a8f1_row8_col22, #T_8a8f1_row8_col23, #T_8a8f1_row8_col24, #T_8a8f1_row8_col25, #T_8a8f1_row8_col26, #T_8a8f1_row8_col27, #T_8a8f1_row9_col0, #T_8a8f1_row9_col1, #T_8a8f1_row9_col20, #T_8a8f1_row9_col21, #T_8a8f1_row9_col22, #T_8a8f1_row9_col23, #T_8a8f1_row9_col24, #T_8a8f1_row9_col25, #T_8a8f1_row9_col26, #T_8a8f1_row9_col27, #T_8a8f1_row10_col0, #T_8a8f1_row10_col1, #T_8a8f1_row10_col2, #T_8a8f1_row10_col3, #T_8a8f1_row10_col4, #T_8a8f1_row10_col13, #T_8a8f1_row10_col14, #T_8a8f1_row10_col15, #T_8a8f1_row10_col16, #T_8a8f1_row10_col20, #T_8a8f1_row10_col21, #T_8a8f1_row10_col22, #T_8a8f1_row10_col23, #T_8a8f1_row10_col24, #T_8a8f1_row10_col25, #T_8a8f1_row10_col26, #T_8a8f1_row10_col27, #T_8a8f1_row11_col0, #T_8a8f1_row11_col1, #T_8a8f1_row11_col2, #T_8a8f1_row11_col3, #T_8a8f1_row11_col4, #T_8a8f1_row11_col5, #T_8a8f1_row11_col6, #T_8a8f1_row11_col7, #T_8a8f1_row11_col8, #T_8a8f1_row11_col9, #T_8a8f1_row11_col10, #T_8a8f1_row11_col11, #T_8a8f1_row11_col12, #T_8a8f1_row11_col13, #T_8a8f1_row11_col14, #T_8a8f1_row11_col15, #T_8a8f1_row11_col20, #T_8a8f1_row11_col21, #T_8a8f1_row11_col22, #T_8a8f1_row11_col23, #T_8a8f1_row11_col24, #T_8a8f1_row11_col25, #T_8a8f1_row11_col26, #T_8a8f1_row11_col27, #T_8a8f1_row12_col0, #T_8a8f1_row12_col1, #T_8a8f1_row12_col2, #T_8a8f1_row12_col3, #T_8a8f1_row12_col4, #T_8a8f1_row12_col5, #T_8a8f1_row12_col6, #T_8a8f1_row12_col7, #T_8a8f1_row12_col8, #T_8a8f1_row12_col9, #T_8a8f1_row12_col10, #T_8a8f1_row12_col11, #T_8a8f1_row12_col12, #T_8a8f1_row12_col13, #T_8a8f1_row12_col14, #T_8a8f1_row12_col15, #T_8a8f1_row12_col20, #T_8a8f1_row12_col21, #T_8a8f1_row12_col22, #T_8a8f1_row12_col23, #T_8a8f1_row12_col24, #T_8a8f1_row12_col25, #T_8a8f1_row12_col26, #T_8a8f1_row12_col27, #T_8a8f1_row13_col0, #T_8a8f1_row13_col1, #T_8a8f1_row13_col2, #T_8a8f1_row13_col3, #T_8a8f1_row13_col4, #T_8a8f1_row13_col5, #T_8a8f1_row13_col6, #T_8a8f1_row13_col7, #T_8a8f1_row13_col8, #T_8a8f1_row13_col9, #T_8a8f1_row13_col10, #T_8a8f1_row13_col11, #T_8a8f1_row13_col12, #T_8a8f1_row13_col13, #T_8a8f1_row13_col14, #T_8a8f1_row13_col15, #T_8a8f1_row13_col20, #T_8a8f1_row13_col21, #T_8a8f1_row13_col22, #T_8a8f1_row13_col23, #T_8a8f1_row13_col24, #T_8a8f1_row13_col25, #T_8a8f1_row13_col26, #T_8a8f1_row13_col27, #T_8a8f1_row14_col0, #T_8a8f1_row14_col1, #T_8a8f1_row14_col2, #T_8a8f1_row14_col3, #T_8a8f1_row14_col4, #T_8a8f1_row14_col5, #T_8a8f1_row14_col6, #T_8a8f1_row14_col7, #T_8a8f1_row14_col8, #T_8a8f1_row14_col9, #T_8a8f1_row14_col10, #T_8a8f1_row14_col11, #T_8a8f1_row14_col12, #T_8a8f1_row14_col13, #T_8a8f1_row14_col14, #T_8a8f1_row14_col15, #T_8a8f1_row14_col19, #T_8a8f1_row14_col20, #T_8a8f1_row14_col21, #T_8a8f1_row14_col22, #T_8a8f1_row14_col23, #T_8a8f1_row14_col24, #T_8a8f1_row14_col25, #T_8a8f1_row14_col26, #T_8a8f1_row14_col27, #T_8a8f1_row15_col0, #T_8a8f1_row15_col1, #T_8a8f1_row15_col2, #T_8a8f1_row15_col3, #T_8a8f1_row15_col4, #T_8a8f1_row15_col5, #T_8a8f1_row15_col6, #T_8a8f1_row15_col7, #T_8a8f1_row15_col8, #T_8a8f1_row15_col9, #T_8a8f1_row15_col10, #T_8a8f1_row15_col11, #T_8a8f1_row15_col12, #T_8a8f1_row15_col13, #T_8a8f1_row15_col14, #T_8a8f1_row15_col19, #T_8a8f1_row15_col20, #T_8a8f1_row15_col21, #T_8a8f1_row15_col22, #T_8a8f1_row15_col23, #T_8a8f1_row15_col24, #T_8a8f1_row15_col25, #T_8a8f1_row15_col26, #T_8a8f1_row15_col27, #T_8a8f1_row16_col0, #T_8a8f1_row16_col1, #T_8a8f1_row16_col2, #T_8a8f1_row16_col3, #T_8a8f1_row16_col4, #T_8a8f1_row16_col5, #T_8a8f1_row16_col6, #T_8a8f1_row16_col7, #T_8a8f1_row16_col8, #T_8a8f1_row16_col9, #T_8a8f1_row16_col10, #T_8a8f1_row16_col11, #T_8a8f1_row16_col12, #T_8a8f1_row16_col13, #T_8a8f1_row16_col14, #T_8a8f1_row16_col19, #T_8a8f1_row16_col20, #T_8a8f1_row16_col21, #T_8a8f1_row16_col22, #T_8a8f1_row16_col23, #T_8a8f1_row16_col24, #T_8a8f1_row16_col25, #T_8a8f1_row16_col26, #T_8a8f1_row16_col27, #T_8a8f1_row17_col0, #T_8a8f1_row17_col1, #T_8a8f1_row17_col2, #T_8a8f1_row17_col3, #T_8a8f1_row17_col4, #T_8a8f1_row17_col5, #T_8a8f1_row17_col6, #T_8a8f1_row17_col7, #T_8a8f1_row17_col8, #T_8a8f1_row17_col9, #T_8a8f1_row17_col10, #T_8a8f1_row17_col11, #T_8a8f1_row17_col12, #T_8a8f1_row17_col13, #T_8a8f1_row17_col14, #T_8a8f1_row17_col19, #T_8a8f1_row17_col20, #T_8a8f1_row17_col21, #T_8a8f1_row17_col22, #T_8a8f1_row17_col23, #T_8a8f1_row17_col24, #T_8a8f1_row17_col25, #T_8a8f1_row17_col26, #T_8a8f1_row17_col27, #T_8a8f1_row18_col0, #T_8a8f1_row18_col1, #T_8a8f1_row18_col2, #T_8a8f1_row18_col3, #T_8a8f1_row18_col4, #T_8a8f1_row18_col5, #T_8a8f1_row18_col6, #T_8a8f1_row18_col7, #T_8a8f1_row18_col8, #T_8a8f1_row18_col9, #T_8a8f1_row18_col10, #T_8a8f1_row18_col11, #T_8a8f1_row18_col12, #T_8a8f1_row18_col13, #T_8a8f1_row18_col14, #T_8a8f1_row18_col19, #T_8a8f1_row18_col20, #T_8a8f1_row18_col21, #T_8a8f1_row18_col22, #T_8a8f1_row18_col23, #T_8a8f1_row18_col24, #T_8a8f1_row18_col25, #T_8a8f1_row18_col26, #T_8a8f1_row18_col27, #T_8a8f1_row19_col0, #T_8a8f1_row19_col1, #T_8a8f1_row19_col2, #T_8a8f1_row19_col3, #T_8a8f1_row19_col4, #T_8a8f1_row19_col5, #T_8a8f1_row19_col6, #T_8a8f1_row19_col7, #T_8a8f1_row19_col8, #T_8a8f1_row19_col9, #T_8a8f1_row19_col10, #T_8a8f1_row19_col11, #T_8a8f1_row19_col12, #T_8a8f1_row19_col13, #T_8a8f1_row19_col14, #T_8a8f1_row19_col18, #T_8a8f1_row19_col19, #T_8a8f1_row19_col20, #T_8a8f1_row19_col21, #T_8a8f1_row19_col22, #T_8a8f1_row19_col23, #T_8a8f1_row19_col24, #T_8a8f1_row19_col25, #T_8a8f1_row19_col26, #T_8a8f1_row19_col27, #T_8a8f1_row20_col0, #T_8a8f1_row20_col1, #T_8a8f1_row20_col2, #T_8a8f1_row20_col3, #T_8a8f1_row20_col4, #T_8a8f1_row20_col5, #T_8a8f1_row20_col6, #T_8a8f1_row20_col7, #T_8a8f1_row20_col8, #T_8a8f1_row20_col9, #T_8a8f1_row20_col10, #T_8a8f1_row20_col11, #T_8a8f1_row20_col12, #T_8a8f1_row20_col13, #T_8a8f1_row20_col14, #T_8a8f1_row20_col18, #T_8a8f1_row20_col19, #T_8a8f1_row20_col20, #T_8a8f1_row20_col21, #T_8a8f1_row20_col22, #T_8a8f1_row20_col23, #T_8a8f1_row20_col24, #T_8a8f1_row20_col25, #T_8a8f1_row20_col26, #T_8a8f1_row20_col27, #T_8a8f1_row21_col0, #T_8a8f1_row21_col1, #T_8a8f1_row21_col2, #T_8a8f1_row21_col3, #T_8a8f1_row21_col4, #T_8a8f1_row21_col5, #T_8a8f1_row21_col6, #T_8a8f1_row21_col7, #T_8a8f1_row21_col8, #T_8a8f1_row21_col9, #T_8a8f1_row21_col10, #T_8a8f1_row21_col11, #T_8a8f1_row21_col12, #T_8a8f1_row21_col13, #T_8a8f1_row21_col14, #T_8a8f1_row21_col18, #T_8a8f1_row21_col19, #T_8a8f1_row21_col20, #T_8a8f1_row21_col21, #T_8a8f1_row21_col22, #T_8a8f1_row21_col23, #T_8a8f1_row21_col24, #T_8a8f1_row21_col25, #T_8a8f1_row21_col26, #T_8a8f1_row21_col27, #T_8a8f1_row22_col0, #T_8a8f1_row22_col1, #T_8a8f1_row22_col2, #T_8a8f1_row22_col3, #T_8a8f1_row22_col4, #T_8a8f1_row22_col5, #T_8a8f1_row22_col6, #T_8a8f1_row22_col7, #T_8a8f1_row22_col8, #T_8a8f1_row22_col9, #T_8a8f1_row22_col10, #T_8a8f1_row22_col11, #T_8a8f1_row22_col12, #T_8a8f1_row22_col13, #T_8a8f1_row22_col18, #T_8a8f1_row22_col19, #T_8a8f1_row22_col20, #T_8a8f1_row22_col21, #T_8a8f1_row22_col22, #T_8a8f1_row22_col23, #T_8a8f1_row22_col24, #T_8a8f1_row22_col25, #T_8a8f1_row22_col26, #T_8a8f1_row22_col27, #T_8a8f1_row23_col0, #T_8a8f1_row23_col1, #T_8a8f1_row23_col2, #T_8a8f1_row23_col3, #T_8a8f1_row23_col4, #T_8a8f1_row23_col5, #T_8a8f1_row23_col6, #T_8a8f1_row23_col7, #T_8a8f1_row23_col8, #T_8a8f1_row23_col9, #T_8a8f1_row23_col10, #T_8a8f1_row23_col11, #T_8a8f1_row23_col12, #T_8a8f1_row23_col13, #T_8a8f1_row23_col18, #T_8a8f1_row23_col19, #T_8a8f1_row23_col20, #T_8a8f1_row23_col21, #T_8a8f1_row23_col22, #T_8a8f1_row23_col23, #T_8a8f1_row23_col24, #T_8a8f1_row23_col25, #T_8a8f1_row23_col26, #T_8a8f1_row23_col27, #T_8a8f1_row24_col0, #T_8a8f1_row24_col1, #T_8a8f1_row24_col2, #T_8a8f1_row24_col3, #T_8a8f1_row24_col4, #T_8a8f1_row24_col5, #T_8a8f1_row24_col6, #T_8a8f1_row24_col7, #T_8a8f1_row24_col8, #T_8a8f1_row24_col9, #T_8a8f1_row24_col10, #T_8a8f1_row24_col11, #T_8a8f1_row24_col12, #T_8a8f1_row24_col13, #T_8a8f1_row24_col18, #T_8a8f1_row24_col19, #T_8a8f1_row24_col20, #T_8a8f1_row24_col21, #T_8a8f1_row24_col22, #T_8a8f1_row24_col23, #T_8a8f1_row24_col24, #T_8a8f1_row24_col25, #T_8a8f1_row24_col26, #T_8a8f1_row24_col27, #T_8a8f1_row25_col0, #T_8a8f1_row25_col1, #T_8a8f1_row25_col2, #T_8a8f1_row25_col3, #T_8a8f1_row25_col4, #T_8a8f1_row25_col5, #T_8a8f1_row25_col6, #T_8a8f1_row25_col7, #T_8a8f1_row25_col8, #T_8a8f1_row25_col9, #T_8a8f1_row25_col10, #T_8a8f1_row25_col11, #T_8a8f1_row25_col12, #T_8a8f1_row25_col13, #T_8a8f1_row25_col14, #T_8a8f1_row25_col18, #T_8a8f1_row25_col19, #T_8a8f1_row25_col20, #T_8a8f1_row25_col21, #T_8a8f1_row25_col22, #T_8a8f1_row25_col23, #T_8a8f1_row25_col24, #T_8a8f1_row25_col25, #T_8a8f1_row25_col26, #T_8a8f1_row25_col27, #T_8a8f1_row26_col0, #T_8a8f1_row26_col1, #T_8a8f1_row26_col2, #T_8a8f1_row26_col3, #T_8a8f1_row26_col4, #T_8a8f1_row26_col5, #T_8a8f1_row26_col6, #T_8a8f1_row26_col7, #T_8a8f1_row26_col8, #T_8a8f1_row26_col9, #T_8a8f1_row26_col10, #T_8a8f1_row26_col11, #T_8a8f1_row26_col12, #T_8a8f1_row26_col13, #T_8a8f1_row26_col14, #T_8a8f1_row26_col18, #T_8a8f1_row26_col19, #T_8a8f1_row26_col20, #T_8a8f1_row26_col21, #T_8a8f1_row26_col22, #T_8a8f1_row26_col23, #T_8a8f1_row26_col24, #T_8a8f1_row26_col25, #T_8a8f1_row26_col26, #T_8a8f1_row26_col27, #T_8a8f1_row27_col0, #T_8a8f1_row27_col1, #T_8a8f1_row27_col2, #T_8a8f1_row27_col3, #T_8a8f1_row27_col4, #T_8a8f1_row27_col5, #T_8a8f1_row27_col6, #T_8a8f1_row27_col7, #T_8a8f1_row27_col8, #T_8a8f1_row27_col9, #T_8a8f1_row27_col10, #T_8a8f1_row27_col11, #T_8a8f1_row27_col12, #T_8a8f1_row27_col13, #T_8a8f1_row27_col14, #T_8a8f1_row27_col15, #T_8a8f1_row27_col16, #T_8a8f1_row27_col17, #T_8a8f1_row27_col18, #T_8a8f1_row27_col19, #T_8a8f1_row27_col20, #T_8a8f1_row27_col21, #T_8a8f1_row27_col22, #T_8a8f1_row27_col23, #T_8a8f1_row27_col24, #T_8a8f1_row27_col25, #T_8a8f1_row27_col26, #T_8a8f1_row27_col27 {
  font-size: 6pt;
  background-color: #ffffff;
  color: #000000;
}
#T_8a8f1_row7_col14 {
  font-size: 6pt;
  background-color: #d8d8d8;
  color: #000000;
}
#T_8a8f1_row7_col15 {
  font-size: 6pt;
  background-color: #828282;
  color: #f1f1f1;
}
#T_8a8f1_row7_col16 {
  font-size: 6pt;
  background-color: #707070;
  color: #f1f1f1;
}
#T_8a8f1_row7_col17 {
  font-size: 6pt;
  background-color: #282828;
  color: #f1f1f1;
}
#T_8a8f1_row7_col18, #T_8a8f1_row7_col19, #T_8a8f1_row8_col2, #T_8a8f1_row8_col3, #T_8a8f1_row8_col4, #T_8a8f1_row8_col13, #T_8a8f1_row8_col14, #T_8a8f1_row9_col2, #T_8a8f1_row9_col4, #T_8a8f1_row9_col5, #T_8a8f1_row9_col6, #T_8a8f1_row9_col7, #T_8a8f1_row9_col8, #T_8a8f1_row9_col9, #T_8a8f1_row9_col10, #T_8a8f1_row9_col11, #T_8a8f1_row9_col12, #T_8a8f1_row11_col17, #T_8a8f1_row11_col18, #T_8a8f1_row15_col16, #T_8a8f1_row19_col16, #T_8a8f1_row23_col15, #T_8a8f1_row23_col16 {
  font-size: 6pt;
  background-color: #000000;
  color: #f1f1f1;
}
#T_8a8f1_row8_col5, #T_8a8f1_row22_col15 {
  font-size: 6pt;
  background-color: #101010;
  color: #f1f1f1;
}
#T_8a8f1_row8_col6 {
  font-size: 6pt;
  background-color: #5c5c5c;
  color: #f1f1f1;
}
#T_8a8f1_row8_col7, #T_8a8f1_row8_col8, #T_8a8f1_row8_col10 {
  font-size: 6pt;
  background-color: #898989;
  color: #f1f1f1;
}
#T_8a8f1_row8_col9, #T_8a8f1_row14_col18 {
  font-size: 6pt;
  background-color: #888888;
  color: #f1f1f1;
}
#T_8a8f1_row8_col11, #T_8a8f1_row13_col18 {
  font-size: 6pt;
  background-color: #2f2f2f;
  color: #f1f1f1;
}
#T_8a8f1_row8_col12 {
  font-size: 6pt;
  background-color: #1b1b1b;
  color: #f1f1f1;
}
#T_8a8f1_row8_col15, #T_8a8f1_row8_col16, #T_8a8f1_row8_col17, #T_8a8f1_row8_col18, #T_8a8f1_row8_col19, #T_8a8f1_row9_col13, #T_8a8f1_row9_col17, #T_8a8f1_row9_col18, #T_8a8f1_row9_col19, #T_8a8f1_row10_col17, #T_8a8f1_row10_col18, #T_8a8f1_row12_col17, #T_8a8f1_row12_col18, #T_8a8f1_row13_col17, #T_8a8f1_row14_col17, #T_8a8f1_row15_col17, #T_8a8f1_row16_col16, #T_8a8f1_row16_col17, #T_8a8f1_row17_col16, #T_8a8f1_row17_col17, #T_8a8f1_row18_col16, #T_8a8f1_row18_col17, #T_8a8f1_row20_col16, #T_8a8f1_row21_col16, #T_8a8f1_row22_col16, #T_8a8f1_row24_col16, #T_8a8f1_row25_col16, #T_8a8f1_row26_col16 {
  font-size: 6pt;
  background-color: #010101;
  color: #f1f1f1;
}
#T_8a8f1_row9_col3 {
  font-size: 6pt;
  background-color: #646464;
  color: #f1f1f1;
}
#T_8a8f1_row9_col14 {
  font-size: 6pt;
  background-color: #181818;
  color: #f1f1f1;
}
#T_8a8f1_row9_col15 {
  font-size: 6pt;
  background-color: #787878;
  color: #f1f1f1;
}
#T_8a8f1_row9_col16 {
  font-size: 6pt;
  background-color: #b3b3b3;
  color: #000000;
}
#T_8a8f1_row10_col5 {
  font-size: 6pt;
  background-color: #e7e7e7;
  color: #000000;
}
#T_8a8f1_row10_col6, #T_8a8f1_row10_col9, #T_8a8f1_row10_col10, #T_8a8f1_row10_col11, #T_8a8f1_row10_col12 {
  font-size: 6pt;
  background-color: #d4d4d4;
  color: #000000;
}
#T_8a8f1_row10_col7 {
  font-size: 6pt;
  background-color: #a9a9a9;
  color: #f1f1f1;
}
#T_8a8f1_row10_col8 {
  font-size: 6pt;
  background-color: #949494;
  color: #f1f1f1;
}
#T_8a8f1_row10_col19 {
  font-size: 6pt;
  background-color: #929292;
  color: #f1f1f1;
}
#T_8a8f1_row11_col16, #T_8a8f1_row22_col17, #T_8a8f1_row23_col17, #T_8a8f1_row24_col17 {
  font-size: 6pt;
  background-color: #bfbfbf;
  color: #000000;
}
#T_8a8f1_row11_col19, #T_8a8f1_row12_col19 {
  font-size: 6pt;
  background-color: #c6c6c6;
  color: #000000;
}
#T_8a8f1_row12_col16 {
  font-size: 6pt;
  background-color: #616161;
  color: #f1f1f1;
}
#T_8a8f1_row13_col16, #T_8a8f1_row14_col16, #T_8a8f1_row19_col17 {
  font-size: 6pt;
  background-color: #585858;
  color: #f1f1f1;
}
#T_8a8f1_row13_col19 {
  font-size: 6pt;
  background-color: #f7f7f7;
  color: #000000;
}
#T_8a8f1_row15_col15, #T_8a8f1_row16_col15, #T_8a8f1_row18_col18 {
  font-size: 6pt;
  background-color: #f4f4f4;
  color: #000000;
}
#T_8a8f1_row15_col18, #T_8a8f1_row16_col18 {
  font-size: 6pt;
  background-color: #a5a5a5;
  color: #f1f1f1;
}
#T_8a8f1_row17_col15 {
  font-size: 6pt;
  background-color: #b9b9b9;
  color: #000000;
}
#T_8a8f1_row17_col18 {
  font-size: 6pt;
  background-color: #afafaf;
  color: #000000;
}
#T_8a8f1_row18_col15 {
  font-size: 6pt;
  background-color: #7d7d7d;
  color: #f1f1f1;
}
#T_8a8f1_row19_col15 {
  font-size: 6pt;
  background-color: #393939;
  color: #f1f1f1;
}
#T_8a8f1_row20_col15, #T_8a8f1_row21_col15, #T_8a8f1_row25_col15 {
  font-size: 6pt;
  background-color: #3a3a3a;
  color: #f1f1f1;
}
#T_8a8f1_row20_col17 {
  font-size: 6pt;
  background-color: #5a5a5a;
  color: #f1f1f1;
}
#T_8a8f1_row21_col17 {
  font-size: 6pt;
  background-color: #adadad;
  color: #000000;
}
#T_8a8f1_row22_col14 {
  font-size: 6pt;
  background-color: #eeeeee;
  color: #000000;
}
#T_8a8f1_row23_col14 {
  font-size: 6pt;
  background-color: #e5e5e5;
  color: #000000;
}
#T_8a8f1_row24_col14 {
  font-size: 6pt;
  background-color: #ececec;
  color: #000000;
}
#T_8a8f1_row24_col15 {
  font-size: 6pt;
  background-color: #0c0c0c;
  color: #f1f1f1;
}
#T_8a8f1_row25_col17 {
  font-size: 6pt;
  background-color: #6a6a6a;
  color: #f1f1f1;
}
#T_8a8f1_row26_col15 {
  font-size: 6pt;
  background-color: #636363;
  color: #f1f1f1;
}
#T_8a8f1_row26_col17 {
  font-size: 6pt;
  background-color: #7c7c7c;
  color: #f1f1f1;
}
</style>
<table id="T_8a8f1">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_8a8f1_level0_col0" class="col_heading level0 col0" >0</th>
      <th id="T_8a8f1_level0_col1" class="col_heading level0 col1" >1</th>
      <th id="T_8a8f1_level0_col2" class="col_heading level0 col2" >2</th>
      <th id="T_8a8f1_level0_col3" class="col_heading level0 col3" >3</th>
      <th id="T_8a8f1_level0_col4" class="col_heading level0 col4" >4</th>
      <th id="T_8a8f1_level0_col5" class="col_heading level0 col5" >5</th>
      <th id="T_8a8f1_level0_col6" class="col_heading level0 col6" >6</th>
      <th id="T_8a8f1_level0_col7" class="col_heading level0 col7" >7</th>
      <th id="T_8a8f1_level0_col8" class="col_heading level0 col8" >8</th>
      <th id="T_8a8f1_level0_col9" class="col_heading level0 col9" >9</th>
      <th id="T_8a8f1_level0_col10" class="col_heading level0 col10" >10</th>
      <th id="T_8a8f1_level0_col11" class="col_heading level0 col11" >11</th>
      <th id="T_8a8f1_level0_col12" class="col_heading level0 col12" >12</th>
      <th id="T_8a8f1_level0_col13" class="col_heading level0 col13" >13</th>
      <th id="T_8a8f1_level0_col14" class="col_heading level0 col14" >14</th>
      <th id="T_8a8f1_level0_col15" class="col_heading level0 col15" >15</th>
      <th id="T_8a8f1_level0_col16" class="col_heading level0 col16" >16</th>
      <th id="T_8a8f1_level0_col17" class="col_heading level0 col17" >17</th>
      <th id="T_8a8f1_level0_col18" class="col_heading level0 col18" >18</th>
      <th id="T_8a8f1_level0_col19" class="col_heading level0 col19" >19</th>
      <th id="T_8a8f1_level0_col20" class="col_heading level0 col20" >20</th>
      <th id="T_8a8f1_level0_col21" class="col_heading level0 col21" >21</th>
      <th id="T_8a8f1_level0_col22" class="col_heading level0 col22" >22</th>
      <th id="T_8a8f1_level0_col23" class="col_heading level0 col23" >23</th>
      <th id="T_8a8f1_level0_col24" class="col_heading level0 col24" >24</th>
      <th id="T_8a8f1_level0_col25" class="col_heading level0 col25" >25</th>
      <th id="T_8a8f1_level0_col26" class="col_heading level0 col26" >26</th>
      <th id="T_8a8f1_level0_col27" class="col_heading level0 col27" >27</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_8a8f1_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_8a8f1_row0_col0" class="data row0 col0" >0</td>
      <td id="T_8a8f1_row0_col1" class="data row0 col1" >0</td>
      <td id="T_8a8f1_row0_col2" class="data row0 col2" >0</td>
      <td id="T_8a8f1_row0_col3" class="data row0 col3" >0</td>
      <td id="T_8a8f1_row0_col4" class="data row0 col4" >0</td>
      <td id="T_8a8f1_row0_col5" class="data row0 col5" >0</td>
      <td id="T_8a8f1_row0_col6" class="data row0 col6" >0</td>
      <td id="T_8a8f1_row0_col7" class="data row0 col7" >0</td>
      <td id="T_8a8f1_row0_col8" class="data row0 col8" >0</td>
      <td id="T_8a8f1_row0_col9" class="data row0 col9" >0</td>
      <td id="T_8a8f1_row0_col10" class="data row0 col10" >0</td>
      <td id="T_8a8f1_row0_col11" class="data row0 col11" >0</td>
      <td id="T_8a8f1_row0_col12" class="data row0 col12" >0</td>
      <td id="T_8a8f1_row0_col13" class="data row0 col13" >0</td>
      <td id="T_8a8f1_row0_col14" class="data row0 col14" >0</td>
      <td id="T_8a8f1_row0_col15" class="data row0 col15" >0</td>
      <td id="T_8a8f1_row0_col16" class="data row0 col16" >0</td>
      <td id="T_8a8f1_row0_col17" class="data row0 col17" >0</td>
      <td id="T_8a8f1_row0_col18" class="data row0 col18" >0</td>
      <td id="T_8a8f1_row0_col19" class="data row0 col19" >0</td>
      <td id="T_8a8f1_row0_col20" class="data row0 col20" >0</td>
      <td id="T_8a8f1_row0_col21" class="data row0 col21" >0</td>
      <td id="T_8a8f1_row0_col22" class="data row0 col22" >0</td>
      <td id="T_8a8f1_row0_col23" class="data row0 col23" >0</td>
      <td id="T_8a8f1_row0_col24" class="data row0 col24" >0</td>
      <td id="T_8a8f1_row0_col25" class="data row0 col25" >0</td>
      <td id="T_8a8f1_row0_col26" class="data row0 col26" >0</td>
      <td id="T_8a8f1_row0_col27" class="data row0 col27" >0</td>
    </tr>
    <tr>
      <th id="T_8a8f1_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_8a8f1_row1_col0" class="data row1 col0" >0</td>
      <td id="T_8a8f1_row1_col1" class="data row1 col1" >0</td>
      <td id="T_8a8f1_row1_col2" class="data row1 col2" >0</td>
      <td id="T_8a8f1_row1_col3" class="data row1 col3" >0</td>
      <td id="T_8a8f1_row1_col4" class="data row1 col4" >0</td>
      <td id="T_8a8f1_row1_col5" class="data row1 col5" >0</td>
      <td id="T_8a8f1_row1_col6" class="data row1 col6" >0</td>
      <td id="T_8a8f1_row1_col7" class="data row1 col7" >0</td>
      <td id="T_8a8f1_row1_col8" class="data row1 col8" >0</td>
      <td id="T_8a8f1_row1_col9" class="data row1 col9" >0</td>
      <td id="T_8a8f1_row1_col10" class="data row1 col10" >0</td>
      <td id="T_8a8f1_row1_col11" class="data row1 col11" >0</td>
      <td id="T_8a8f1_row1_col12" class="data row1 col12" >0</td>
      <td id="T_8a8f1_row1_col13" class="data row1 col13" >0</td>
      <td id="T_8a8f1_row1_col14" class="data row1 col14" >0</td>
      <td id="T_8a8f1_row1_col15" class="data row1 col15" >0</td>
      <td id="T_8a8f1_row1_col16" class="data row1 col16" >0</td>
      <td id="T_8a8f1_row1_col17" class="data row1 col17" >0</td>
      <td id="T_8a8f1_row1_col18" class="data row1 col18" >0</td>
      <td id="T_8a8f1_row1_col19" class="data row1 col19" >0</td>
      <td id="T_8a8f1_row1_col20" class="data row1 col20" >0</td>
      <td id="T_8a8f1_row1_col21" class="data row1 col21" >0</td>
      <td id="T_8a8f1_row1_col22" class="data row1 col22" >0</td>
      <td id="T_8a8f1_row1_col23" class="data row1 col23" >0</td>
      <td id="T_8a8f1_row1_col24" class="data row1 col24" >0</td>
      <td id="T_8a8f1_row1_col25" class="data row1 col25" >0</td>
      <td id="T_8a8f1_row1_col26" class="data row1 col26" >0</td>
      <td id="T_8a8f1_row1_col27" class="data row1 col27" >0</td>
    </tr>
    <tr>
      <th id="T_8a8f1_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_8a8f1_row2_col0" class="data row2 col0" >0</td>
      <td id="T_8a8f1_row2_col1" class="data row2 col1" >0</td>
      <td id="T_8a8f1_row2_col2" class="data row2 col2" >0</td>
      <td id="T_8a8f1_row2_col3" class="data row2 col3" >0</td>
      <td id="T_8a8f1_row2_col4" class="data row2 col4" >0</td>
      <td id="T_8a8f1_row2_col5" class="data row2 col5" >0</td>
      <td id="T_8a8f1_row2_col6" class="data row2 col6" >0</td>
      <td id="T_8a8f1_row2_col7" class="data row2 col7" >0</td>
      <td id="T_8a8f1_row2_col8" class="data row2 col8" >0</td>
      <td id="T_8a8f1_row2_col9" class="data row2 col9" >0</td>
      <td id="T_8a8f1_row2_col10" class="data row2 col10" >0</td>
      <td id="T_8a8f1_row2_col11" class="data row2 col11" >0</td>
      <td id="T_8a8f1_row2_col12" class="data row2 col12" >0</td>
      <td id="T_8a8f1_row2_col13" class="data row2 col13" >0</td>
      <td id="T_8a8f1_row2_col14" class="data row2 col14" >0</td>
      <td id="T_8a8f1_row2_col15" class="data row2 col15" >0</td>
      <td id="T_8a8f1_row2_col16" class="data row2 col16" >0</td>
      <td id="T_8a8f1_row2_col17" class="data row2 col17" >0</td>
      <td id="T_8a8f1_row2_col18" class="data row2 col18" >0</td>
      <td id="T_8a8f1_row2_col19" class="data row2 col19" >0</td>
      <td id="T_8a8f1_row2_col20" class="data row2 col20" >0</td>
      <td id="T_8a8f1_row2_col21" class="data row2 col21" >0</td>
      <td id="T_8a8f1_row2_col22" class="data row2 col22" >0</td>
      <td id="T_8a8f1_row2_col23" class="data row2 col23" >0</td>
      <td id="T_8a8f1_row2_col24" class="data row2 col24" >0</td>
      <td id="T_8a8f1_row2_col25" class="data row2 col25" >0</td>
      <td id="T_8a8f1_row2_col26" class="data row2 col26" >0</td>
      <td id="T_8a8f1_row2_col27" class="data row2 col27" >0</td>
    </tr>
    <tr>
      <th id="T_8a8f1_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_8a8f1_row3_col0" class="data row3 col0" >0</td>
      <td id="T_8a8f1_row3_col1" class="data row3 col1" >0</td>
      <td id="T_8a8f1_row3_col2" class="data row3 col2" >0</td>
      <td id="T_8a8f1_row3_col3" class="data row3 col3" >0</td>
      <td id="T_8a8f1_row3_col4" class="data row3 col4" >0</td>
      <td id="T_8a8f1_row3_col5" class="data row3 col5" >0</td>
      <td id="T_8a8f1_row3_col6" class="data row3 col6" >0</td>
      <td id="T_8a8f1_row3_col7" class="data row3 col7" >0</td>
      <td id="T_8a8f1_row3_col8" class="data row3 col8" >0</td>
      <td id="T_8a8f1_row3_col9" class="data row3 col9" >0</td>
      <td id="T_8a8f1_row3_col10" class="data row3 col10" >0</td>
      <td id="T_8a8f1_row3_col11" class="data row3 col11" >0</td>
      <td id="T_8a8f1_row3_col12" class="data row3 col12" >0</td>
      <td id="T_8a8f1_row3_col13" class="data row3 col13" >0</td>
      <td id="T_8a8f1_row3_col14" class="data row3 col14" >0</td>
      <td id="T_8a8f1_row3_col15" class="data row3 col15" >0</td>
      <td id="T_8a8f1_row3_col16" class="data row3 col16" >0</td>
      <td id="T_8a8f1_row3_col17" class="data row3 col17" >0</td>
      <td id="T_8a8f1_row3_col18" class="data row3 col18" >0</td>
      <td id="T_8a8f1_row3_col19" class="data row3 col19" >0</td>
      <td id="T_8a8f1_row3_col20" class="data row3 col20" >0</td>
      <td id="T_8a8f1_row3_col21" class="data row3 col21" >0</td>
      <td id="T_8a8f1_row3_col22" class="data row3 col22" >0</td>
      <td id="T_8a8f1_row3_col23" class="data row3 col23" >0</td>
      <td id="T_8a8f1_row3_col24" class="data row3 col24" >0</td>
      <td id="T_8a8f1_row3_col25" class="data row3 col25" >0</td>
      <td id="T_8a8f1_row3_col26" class="data row3 col26" >0</td>
      <td id="T_8a8f1_row3_col27" class="data row3 col27" >0</td>
    </tr>
    <tr>
      <th id="T_8a8f1_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_8a8f1_row4_col0" class="data row4 col0" >0</td>
      <td id="T_8a8f1_row4_col1" class="data row4 col1" >0</td>
      <td id="T_8a8f1_row4_col2" class="data row4 col2" >0</td>
      <td id="T_8a8f1_row4_col3" class="data row4 col3" >0</td>
      <td id="T_8a8f1_row4_col4" class="data row4 col4" >0</td>
      <td id="T_8a8f1_row4_col5" class="data row4 col5" >0</td>
      <td id="T_8a8f1_row4_col6" class="data row4 col6" >0</td>
      <td id="T_8a8f1_row4_col7" class="data row4 col7" >0</td>
      <td id="T_8a8f1_row4_col8" class="data row4 col8" >0</td>
      <td id="T_8a8f1_row4_col9" class="data row4 col9" >0</td>
      <td id="T_8a8f1_row4_col10" class="data row4 col10" >0</td>
      <td id="T_8a8f1_row4_col11" class="data row4 col11" >0</td>
      <td id="T_8a8f1_row4_col12" class="data row4 col12" >0</td>
      <td id="T_8a8f1_row4_col13" class="data row4 col13" >0</td>
      <td id="T_8a8f1_row4_col14" class="data row4 col14" >0</td>
      <td id="T_8a8f1_row4_col15" class="data row4 col15" >0</td>
      <td id="T_8a8f1_row4_col16" class="data row4 col16" >0</td>
      <td id="T_8a8f1_row4_col17" class="data row4 col17" >0</td>
      <td id="T_8a8f1_row4_col18" class="data row4 col18" >0</td>
      <td id="T_8a8f1_row4_col19" class="data row4 col19" >0</td>
      <td id="T_8a8f1_row4_col20" class="data row4 col20" >0</td>
      <td id="T_8a8f1_row4_col21" class="data row4 col21" >0</td>
      <td id="T_8a8f1_row4_col22" class="data row4 col22" >0</td>
      <td id="T_8a8f1_row4_col23" class="data row4 col23" >0</td>
      <td id="T_8a8f1_row4_col24" class="data row4 col24" >0</td>
      <td id="T_8a8f1_row4_col25" class="data row4 col25" >0</td>
      <td id="T_8a8f1_row4_col26" class="data row4 col26" >0</td>
      <td id="T_8a8f1_row4_col27" class="data row4 col27" >0</td>
    </tr>
    <tr>
      <th id="T_8a8f1_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_8a8f1_row5_col0" class="data row5 col0" >0</td>
      <td id="T_8a8f1_row5_col1" class="data row5 col1" >0</td>
      <td id="T_8a8f1_row5_col2" class="data row5 col2" >0</td>
      <td id="T_8a8f1_row5_col3" class="data row5 col3" >0</td>
      <td id="T_8a8f1_row5_col4" class="data row5 col4" >0</td>
      <td id="T_8a8f1_row5_col5" class="data row5 col5" >0</td>
      <td id="T_8a8f1_row5_col6" class="data row5 col6" >0</td>
      <td id="T_8a8f1_row5_col7" class="data row5 col7" >0</td>
      <td id="T_8a8f1_row5_col8" class="data row5 col8" >0</td>
      <td id="T_8a8f1_row5_col9" class="data row5 col9" >0</td>
      <td id="T_8a8f1_row5_col10" class="data row5 col10" >0</td>
      <td id="T_8a8f1_row5_col11" class="data row5 col11" >0</td>
      <td id="T_8a8f1_row5_col12" class="data row5 col12" >0</td>
      <td id="T_8a8f1_row5_col13" class="data row5 col13" >0</td>
      <td id="T_8a8f1_row5_col14" class="data row5 col14" >0</td>
      <td id="T_8a8f1_row5_col15" class="data row5 col15" >0</td>
      <td id="T_8a8f1_row5_col16" class="data row5 col16" >0</td>
      <td id="T_8a8f1_row5_col17" class="data row5 col17" >0</td>
      <td id="T_8a8f1_row5_col18" class="data row5 col18" >0</td>
      <td id="T_8a8f1_row5_col19" class="data row5 col19" >0</td>
      <td id="T_8a8f1_row5_col20" class="data row5 col20" >0</td>
      <td id="T_8a8f1_row5_col21" class="data row5 col21" >0</td>
      <td id="T_8a8f1_row5_col22" class="data row5 col22" >0</td>
      <td id="T_8a8f1_row5_col23" class="data row5 col23" >0</td>
      <td id="T_8a8f1_row5_col24" class="data row5 col24" >0</td>
      <td id="T_8a8f1_row5_col25" class="data row5 col25" >0</td>
      <td id="T_8a8f1_row5_col26" class="data row5 col26" >0</td>
      <td id="T_8a8f1_row5_col27" class="data row5 col27" >0</td>
    </tr>
    <tr>
      <th id="T_8a8f1_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_8a8f1_row6_col0" class="data row6 col0" >0</td>
      <td id="T_8a8f1_row6_col1" class="data row6 col1" >0</td>
      <td id="T_8a8f1_row6_col2" class="data row6 col2" >0</td>
      <td id="T_8a8f1_row6_col3" class="data row6 col3" >0</td>
      <td id="T_8a8f1_row6_col4" class="data row6 col4" >0</td>
      <td id="T_8a8f1_row6_col5" class="data row6 col5" >0</td>
      <td id="T_8a8f1_row6_col6" class="data row6 col6" >0</td>
      <td id="T_8a8f1_row6_col7" class="data row6 col7" >0</td>
      <td id="T_8a8f1_row6_col8" class="data row6 col8" >0</td>
      <td id="T_8a8f1_row6_col9" class="data row6 col9" >0</td>
      <td id="T_8a8f1_row6_col10" class="data row6 col10" >0</td>
      <td id="T_8a8f1_row6_col11" class="data row6 col11" >0</td>
      <td id="T_8a8f1_row6_col12" class="data row6 col12" >0</td>
      <td id="T_8a8f1_row6_col13" class="data row6 col13" >0</td>
      <td id="T_8a8f1_row6_col14" class="data row6 col14" >0</td>
      <td id="T_8a8f1_row6_col15" class="data row6 col15" >0</td>
      <td id="T_8a8f1_row6_col16" class="data row6 col16" >0</td>
      <td id="T_8a8f1_row6_col17" class="data row6 col17" >0</td>
      <td id="T_8a8f1_row6_col18" class="data row6 col18" >0</td>
      <td id="T_8a8f1_row6_col19" class="data row6 col19" >0</td>
      <td id="T_8a8f1_row6_col20" class="data row6 col20" >0</td>
      <td id="T_8a8f1_row6_col21" class="data row6 col21" >0</td>
      <td id="T_8a8f1_row6_col22" class="data row6 col22" >0</td>
      <td id="T_8a8f1_row6_col23" class="data row6 col23" >0</td>
      <td id="T_8a8f1_row6_col24" class="data row6 col24" >0</td>
      <td id="T_8a8f1_row6_col25" class="data row6 col25" >0</td>
      <td id="T_8a8f1_row6_col26" class="data row6 col26" >0</td>
      <td id="T_8a8f1_row6_col27" class="data row6 col27" >0</td>
    </tr>
    <tr>
      <th id="T_8a8f1_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_8a8f1_row7_col0" class="data row7 col0" >0</td>
      <td id="T_8a8f1_row7_col1" class="data row7 col1" >0</td>
      <td id="T_8a8f1_row7_col2" class="data row7 col2" >0</td>
      <td id="T_8a8f1_row7_col3" class="data row7 col3" >0</td>
      <td id="T_8a8f1_row7_col4" class="data row7 col4" >0</td>
      <td id="T_8a8f1_row7_col5" class="data row7 col5" >0</td>
      <td id="T_8a8f1_row7_col6" class="data row7 col6" >0</td>
      <td id="T_8a8f1_row7_col7" class="data row7 col7" >0</td>
      <td id="T_8a8f1_row7_col8" class="data row7 col8" >0</td>
      <td id="T_8a8f1_row7_col9" class="data row7 col9" >0</td>
      <td id="T_8a8f1_row7_col10" class="data row7 col10" >0</td>
      <td id="T_8a8f1_row7_col11" class="data row7 col11" >0</td>
      <td id="T_8a8f1_row7_col12" class="data row7 col12" >0</td>
      <td id="T_8a8f1_row7_col13" class="data row7 col13" >0</td>
      <td id="T_8a8f1_row7_col14" class="data row7 col14" >64</td>
      <td id="T_8a8f1_row7_col15" class="data row7 col15" >145</td>
      <td id="T_8a8f1_row7_col16" class="data row7 col16" >161</td>
      <td id="T_8a8f1_row7_col17" class="data row7 col17" >221</td>
      <td id="T_8a8f1_row7_col18" class="data row7 col18" >254</td>
      <td id="T_8a8f1_row7_col19" class="data row7 col19" >138</td>
      <td id="T_8a8f1_row7_col20" class="data row7 col20" >0</td>
      <td id="T_8a8f1_row7_col21" class="data row7 col21" >0</td>
      <td id="T_8a8f1_row7_col22" class="data row7 col22" >0</td>
      <td id="T_8a8f1_row7_col23" class="data row7 col23" >0</td>
      <td id="T_8a8f1_row7_col24" class="data row7 col24" >0</td>
      <td id="T_8a8f1_row7_col25" class="data row7 col25" >0</td>
      <td id="T_8a8f1_row7_col26" class="data row7 col26" >0</td>
      <td id="T_8a8f1_row7_col27" class="data row7 col27" >0</td>
    </tr>
    <tr>
      <th id="T_8a8f1_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_8a8f1_row8_col0" class="data row8 col0" >0</td>
      <td id="T_8a8f1_row8_col1" class="data row8 col1" >0</td>
      <td id="T_8a8f1_row8_col2" class="data row8 col2" >76</td>
      <td id="T_8a8f1_row8_col3" class="data row8 col3" >214</td>
      <td id="T_8a8f1_row8_col4" class="data row8 col4" >230</td>
      <td id="T_8a8f1_row8_col5" class="data row8 col5" >231</td>
      <td id="T_8a8f1_row8_col6" class="data row8 col6" >180</td>
      <td id="T_8a8f1_row8_col7" class="data row8 col7" >138</td>
      <td id="T_8a8f1_row8_col8" class="data row8 col8" >138</td>
      <td id="T_8a8f1_row8_col9" class="data row8 col9" >139</td>
      <td id="T_8a8f1_row8_col10" class="data row8 col10" >138</td>
      <td id="T_8a8f1_row8_col11" class="data row8 col11" >214</td>
      <td id="T_8a8f1_row8_col12" class="data row8 col12" >230</td>
      <td id="T_8a8f1_row8_col13" class="data row8 col13" >231</td>
      <td id="T_8a8f1_row8_col14" class="data row8 col14" >251</td>
      <td id="T_8a8f1_row8_col15" class="data row8 col15" >253</td>
      <td id="T_8a8f1_row8_col16" class="data row8 col16" >253</td>
      <td id="T_8a8f1_row8_col17" class="data row8 col17" >254</td>
      <td id="T_8a8f1_row8_col18" class="data row8 col18" >253</td>
      <td id="T_8a8f1_row8_col19" class="data row8 col19" >137</td>
      <td id="T_8a8f1_row8_col20" class="data row8 col20" >0</td>
      <td id="T_8a8f1_row8_col21" class="data row8 col21" >0</td>
      <td id="T_8a8f1_row8_col22" class="data row8 col22" >0</td>
      <td id="T_8a8f1_row8_col23" class="data row8 col23" >0</td>
      <td id="T_8a8f1_row8_col24" class="data row8 col24" >0</td>
      <td id="T_8a8f1_row8_col25" class="data row8 col25" >0</td>
      <td id="T_8a8f1_row8_col26" class="data row8 col26" >0</td>
      <td id="T_8a8f1_row8_col27" class="data row8 col27" >0</td>
    </tr>
    <tr>
      <th id="T_8a8f1_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_8a8f1_row9_col0" class="data row9 col0" >0</td>
      <td id="T_8a8f1_row9_col1" class="data row9 col1" >0</td>
      <td id="T_8a8f1_row9_col2" class="data row9 col2" >76</td>
      <td id="T_8a8f1_row9_col3" class="data row9 col3" >146</td>
      <td id="T_8a8f1_row9_col4" class="data row9 col4" >230</td>
      <td id="T_8a8f1_row9_col5" class="data row9 col5" >245</td>
      <td id="T_8a8f1_row9_col6" class="data row9 col6" >253</td>
      <td id="T_8a8f1_row9_col7" class="data row9 col7" >253</td>
      <td id="T_8a8f1_row9_col8" class="data row9 col8" >253</td>
      <td id="T_8a8f1_row9_col9" class="data row9 col9" >254</td>
      <td id="T_8a8f1_row9_col10" class="data row9 col10" >253</td>
      <td id="T_8a8f1_row9_col11" class="data row9 col11" >253</td>
      <td id="T_8a8f1_row9_col12" class="data row9 col12" >253</td>
      <td id="T_8a8f1_row9_col13" class="data row9 col13" >230</td>
      <td id="T_8a8f1_row9_col14" class="data row9 col14" >230</td>
      <td id="T_8a8f1_row9_col15" class="data row9 col15" >154</td>
      <td id="T_8a8f1_row9_col16" class="data row9 col16" >104</td>
      <td id="T_8a8f1_row9_col17" class="data row9 col17" >254</td>
      <td id="T_8a8f1_row9_col18" class="data row9 col18" >253</td>
      <td id="T_8a8f1_row9_col19" class="data row9 col19" >137</td>
      <td id="T_8a8f1_row9_col20" class="data row9 col20" >0</td>
      <td id="T_8a8f1_row9_col21" class="data row9 col21" >0</td>
      <td id="T_8a8f1_row9_col22" class="data row9 col22" >0</td>
      <td id="T_8a8f1_row9_col23" class="data row9 col23" >0</td>
      <td id="T_8a8f1_row9_col24" class="data row9 col24" >0</td>
      <td id="T_8a8f1_row9_col25" class="data row9 col25" >0</td>
      <td id="T_8a8f1_row9_col26" class="data row9 col26" >0</td>
      <td id="T_8a8f1_row9_col27" class="data row9 col27" >0</td>
    </tr>
    <tr>
      <th id="T_8a8f1_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_8a8f1_row10_col0" class="data row10 col0" >0</td>
      <td id="T_8a8f1_row10_col1" class="data row10 col1" >0</td>
      <td id="T_8a8f1_row10_col2" class="data row10 col2" >0</td>
      <td id="T_8a8f1_row10_col3" class="data row10 col3" >0</td>
      <td id="T_8a8f1_row10_col4" class="data row10 col4" >0</td>
      <td id="T_8a8f1_row10_col5" class="data row10 col5" >44</td>
      <td id="T_8a8f1_row10_col6" class="data row10 col6" >69</td>
      <td id="T_8a8f1_row10_col7" class="data row10 col7" >111</td>
      <td id="T_8a8f1_row10_col8" class="data row10 col8" >128</td>
      <td id="T_8a8f1_row10_col9" class="data row10 col9" >69</td>
      <td id="T_8a8f1_row10_col10" class="data row10 col10" >69</td>
      <td id="T_8a8f1_row10_col11" class="data row10 col11" >69</td>
      <td id="T_8a8f1_row10_col12" class="data row10 col12" >69</td>
      <td id="T_8a8f1_row10_col13" class="data row10 col13" >0</td>
      <td id="T_8a8f1_row10_col14" class="data row10 col14" >0</td>
      <td id="T_8a8f1_row10_col15" class="data row10 col15" >0</td>
      <td id="T_8a8f1_row10_col16" class="data row10 col16" >0</td>
      <td id="T_8a8f1_row10_col17" class="data row10 col17" >254</td>
      <td id="T_8a8f1_row10_col18" class="data row10 col18" >253</td>
      <td id="T_8a8f1_row10_col19" class="data row10 col19" >71</td>
      <td id="T_8a8f1_row10_col20" class="data row10 col20" >0</td>
      <td id="T_8a8f1_row10_col21" class="data row10 col21" >0</td>
      <td id="T_8a8f1_row10_col22" class="data row10 col22" >0</td>
      <td id="T_8a8f1_row10_col23" class="data row10 col23" >0</td>
      <td id="T_8a8f1_row10_col24" class="data row10 col24" >0</td>
      <td id="T_8a8f1_row10_col25" class="data row10 col25" >0</td>
      <td id="T_8a8f1_row10_col26" class="data row10 col26" >0</td>
      <td id="T_8a8f1_row10_col27" class="data row10 col27" >0</td>
    </tr>
    <tr>
      <th id="T_8a8f1_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_8a8f1_row11_col0" class="data row11 col0" >0</td>
      <td id="T_8a8f1_row11_col1" class="data row11 col1" >0</td>
      <td id="T_8a8f1_row11_col2" class="data row11 col2" >0</td>
      <td id="T_8a8f1_row11_col3" class="data row11 col3" >0</td>
      <td id="T_8a8f1_row11_col4" class="data row11 col4" >0</td>
      <td id="T_8a8f1_row11_col5" class="data row11 col5" >0</td>
      <td id="T_8a8f1_row11_col6" class="data row11 col6" >0</td>
      <td id="T_8a8f1_row11_col7" class="data row11 col7" >0</td>
      <td id="T_8a8f1_row11_col8" class="data row11 col8" >0</td>
      <td id="T_8a8f1_row11_col9" class="data row11 col9" >0</td>
      <td id="T_8a8f1_row11_col10" class="data row11 col10" >0</td>
      <td id="T_8a8f1_row11_col11" class="data row11 col11" >0</td>
      <td id="T_8a8f1_row11_col12" class="data row11 col12" >0</td>
      <td id="T_8a8f1_row11_col13" class="data row11 col13" >0</td>
      <td id="T_8a8f1_row11_col14" class="data row11 col14" >0</td>
      <td id="T_8a8f1_row11_col15" class="data row11 col15" >0</td>
      <td id="T_8a8f1_row11_col16" class="data row11 col16" >93</td>
      <td id="T_8a8f1_row11_col17" class="data row11 col17" >255</td>
      <td id="T_8a8f1_row11_col18" class="data row11 col18" >254</td>
      <td id="T_8a8f1_row11_col19" class="data row11 col19" >46</td>
      <td id="T_8a8f1_row11_col20" class="data row11 col20" >0</td>
      <td id="T_8a8f1_row11_col21" class="data row11 col21" >0</td>
      <td id="T_8a8f1_row11_col22" class="data row11 col22" >0</td>
      <td id="T_8a8f1_row11_col23" class="data row11 col23" >0</td>
      <td id="T_8a8f1_row11_col24" class="data row11 col24" >0</td>
      <td id="T_8a8f1_row11_col25" class="data row11 col25" >0</td>
      <td id="T_8a8f1_row11_col26" class="data row11 col26" >0</td>
      <td id="T_8a8f1_row11_col27" class="data row11 col27" >0</td>
    </tr>
    <tr>
      <th id="T_8a8f1_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_8a8f1_row12_col0" class="data row12 col0" >0</td>
      <td id="T_8a8f1_row12_col1" class="data row12 col1" >0</td>
      <td id="T_8a8f1_row12_col2" class="data row12 col2" >0</td>
      <td id="T_8a8f1_row12_col3" class="data row12 col3" >0</td>
      <td id="T_8a8f1_row12_col4" class="data row12 col4" >0</td>
      <td id="T_8a8f1_row12_col5" class="data row12 col5" >0</td>
      <td id="T_8a8f1_row12_col6" class="data row12 col6" >0</td>
      <td id="T_8a8f1_row12_col7" class="data row12 col7" >0</td>
      <td id="T_8a8f1_row12_col8" class="data row12 col8" >0</td>
      <td id="T_8a8f1_row12_col9" class="data row12 col9" >0</td>
      <td id="T_8a8f1_row12_col10" class="data row12 col10" >0</td>
      <td id="T_8a8f1_row12_col11" class="data row12 col11" >0</td>
      <td id="T_8a8f1_row12_col12" class="data row12 col12" >0</td>
      <td id="T_8a8f1_row12_col13" class="data row12 col13" >0</td>
      <td id="T_8a8f1_row12_col14" class="data row12 col14" >0</td>
      <td id="T_8a8f1_row12_col15" class="data row12 col15" >0</td>
      <td id="T_8a8f1_row12_col16" class="data row12 col16" >176</td>
      <td id="T_8a8f1_row12_col17" class="data row12 col17" >254</td>
      <td id="T_8a8f1_row12_col18" class="data row12 col18" >253</td>
      <td id="T_8a8f1_row12_col19" class="data row12 col19" >46</td>
      <td id="T_8a8f1_row12_col20" class="data row12 col20" >0</td>
      <td id="T_8a8f1_row12_col21" class="data row12 col21" >0</td>
      <td id="T_8a8f1_row12_col22" class="data row12 col22" >0</td>
      <td id="T_8a8f1_row12_col23" class="data row12 col23" >0</td>
      <td id="T_8a8f1_row12_col24" class="data row12 col24" >0</td>
      <td id="T_8a8f1_row12_col25" class="data row12 col25" >0</td>
      <td id="T_8a8f1_row12_col26" class="data row12 col26" >0</td>
      <td id="T_8a8f1_row12_col27" class="data row12 col27" >0</td>
    </tr>
    <tr>
      <th id="T_8a8f1_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_8a8f1_row13_col0" class="data row13 col0" >0</td>
      <td id="T_8a8f1_row13_col1" class="data row13 col1" >0</td>
      <td id="T_8a8f1_row13_col2" class="data row13 col2" >0</td>
      <td id="T_8a8f1_row13_col3" class="data row13 col3" >0</td>
      <td id="T_8a8f1_row13_col4" class="data row13 col4" >0</td>
      <td id="T_8a8f1_row13_col5" class="data row13 col5" >0</td>
      <td id="T_8a8f1_row13_col6" class="data row13 col6" >0</td>
      <td id="T_8a8f1_row13_col7" class="data row13 col7" >0</td>
      <td id="T_8a8f1_row13_col8" class="data row13 col8" >0</td>
      <td id="T_8a8f1_row13_col9" class="data row13 col9" >0</td>
      <td id="T_8a8f1_row13_col10" class="data row13 col10" >0</td>
      <td id="T_8a8f1_row13_col11" class="data row13 col11" >0</td>
      <td id="T_8a8f1_row13_col12" class="data row13 col12" >0</td>
      <td id="T_8a8f1_row13_col13" class="data row13 col13" >0</td>
      <td id="T_8a8f1_row13_col14" class="data row13 col14" >0</td>
      <td id="T_8a8f1_row13_col15" class="data row13 col15" >0</td>
      <td id="T_8a8f1_row13_col16" class="data row13 col16" >184</td>
      <td id="T_8a8f1_row13_col17" class="data row13 col17" >254</td>
      <td id="T_8a8f1_row13_col18" class="data row13 col18" >215</td>
      <td id="T_8a8f1_row13_col19" class="data row13 col19" >9</td>
      <td id="T_8a8f1_row13_col20" class="data row13 col20" >0</td>
      <td id="T_8a8f1_row13_col21" class="data row13 col21" >0</td>
      <td id="T_8a8f1_row13_col22" class="data row13 col22" >0</td>
      <td id="T_8a8f1_row13_col23" class="data row13 col23" >0</td>
      <td id="T_8a8f1_row13_col24" class="data row13 col24" >0</td>
      <td id="T_8a8f1_row13_col25" class="data row13 col25" >0</td>
      <td id="T_8a8f1_row13_col26" class="data row13 col26" >0</td>
      <td id="T_8a8f1_row13_col27" class="data row13 col27" >0</td>
    </tr>
    <tr>
      <th id="T_8a8f1_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_8a8f1_row14_col0" class="data row14 col0" >0</td>
      <td id="T_8a8f1_row14_col1" class="data row14 col1" >0</td>
      <td id="T_8a8f1_row14_col2" class="data row14 col2" >0</td>
      <td id="T_8a8f1_row14_col3" class="data row14 col3" >0</td>
      <td id="T_8a8f1_row14_col4" class="data row14 col4" >0</td>
      <td id="T_8a8f1_row14_col5" class="data row14 col5" >0</td>
      <td id="T_8a8f1_row14_col6" class="data row14 col6" >0</td>
      <td id="T_8a8f1_row14_col7" class="data row14 col7" >0</td>
      <td id="T_8a8f1_row14_col8" class="data row14 col8" >0</td>
      <td id="T_8a8f1_row14_col9" class="data row14 col9" >0</td>
      <td id="T_8a8f1_row14_col10" class="data row14 col10" >0</td>
      <td id="T_8a8f1_row14_col11" class="data row14 col11" >0</td>
      <td id="T_8a8f1_row14_col12" class="data row14 col12" >0</td>
      <td id="T_8a8f1_row14_col13" class="data row14 col13" >0</td>
      <td id="T_8a8f1_row14_col14" class="data row14 col14" >0</td>
      <td id="T_8a8f1_row14_col15" class="data row14 col15" >0</td>
      <td id="T_8a8f1_row14_col16" class="data row14 col16" >184</td>
      <td id="T_8a8f1_row14_col17" class="data row14 col17" >254</td>
      <td id="T_8a8f1_row14_col18" class="data row14 col18" >139</td>
      <td id="T_8a8f1_row14_col19" class="data row14 col19" >0</td>
      <td id="T_8a8f1_row14_col20" class="data row14 col20" >0</td>
      <td id="T_8a8f1_row14_col21" class="data row14 col21" >0</td>
      <td id="T_8a8f1_row14_col22" class="data row14 col22" >0</td>
      <td id="T_8a8f1_row14_col23" class="data row14 col23" >0</td>
      <td id="T_8a8f1_row14_col24" class="data row14 col24" >0</td>
      <td id="T_8a8f1_row14_col25" class="data row14 col25" >0</td>
      <td id="T_8a8f1_row14_col26" class="data row14 col26" >0</td>
      <td id="T_8a8f1_row14_col27" class="data row14 col27" >0</td>
    </tr>
    <tr>
      <th id="T_8a8f1_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_8a8f1_row15_col0" class="data row15 col0" >0</td>
      <td id="T_8a8f1_row15_col1" class="data row15 col1" >0</td>
      <td id="T_8a8f1_row15_col2" class="data row15 col2" >0</td>
      <td id="T_8a8f1_row15_col3" class="data row15 col3" >0</td>
      <td id="T_8a8f1_row15_col4" class="data row15 col4" >0</td>
      <td id="T_8a8f1_row15_col5" class="data row15 col5" >0</td>
      <td id="T_8a8f1_row15_col6" class="data row15 col6" >0</td>
      <td id="T_8a8f1_row15_col7" class="data row15 col7" >0</td>
      <td id="T_8a8f1_row15_col8" class="data row15 col8" >0</td>
      <td id="T_8a8f1_row15_col9" class="data row15 col9" >0</td>
      <td id="T_8a8f1_row15_col10" class="data row15 col10" >0</td>
      <td id="T_8a8f1_row15_col11" class="data row15 col11" >0</td>
      <td id="T_8a8f1_row15_col12" class="data row15 col12" >0</td>
      <td id="T_8a8f1_row15_col13" class="data row15 col13" >0</td>
      <td id="T_8a8f1_row15_col14" class="data row15 col14" >0</td>
      <td id="T_8a8f1_row15_col15" class="data row15 col15" >24</td>
      <td id="T_8a8f1_row15_col16" class="data row15 col16" >254</td>
      <td id="T_8a8f1_row15_col17" class="data row15 col17" >254</td>
      <td id="T_8a8f1_row15_col18" class="data row15 col18" >115</td>
      <td id="T_8a8f1_row15_col19" class="data row15 col19" >0</td>
      <td id="T_8a8f1_row15_col20" class="data row15 col20" >0</td>
      <td id="T_8a8f1_row15_col21" class="data row15 col21" >0</td>
      <td id="T_8a8f1_row15_col22" class="data row15 col22" >0</td>
      <td id="T_8a8f1_row15_col23" class="data row15 col23" >0</td>
      <td id="T_8a8f1_row15_col24" class="data row15 col24" >0</td>
      <td id="T_8a8f1_row15_col25" class="data row15 col25" >0</td>
      <td id="T_8a8f1_row15_col26" class="data row15 col26" >0</td>
      <td id="T_8a8f1_row15_col27" class="data row15 col27" >0</td>
    </tr>
    <tr>
      <th id="T_8a8f1_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_8a8f1_row16_col0" class="data row16 col0" >0</td>
      <td id="T_8a8f1_row16_col1" class="data row16 col1" >0</td>
      <td id="T_8a8f1_row16_col2" class="data row16 col2" >0</td>
      <td id="T_8a8f1_row16_col3" class="data row16 col3" >0</td>
      <td id="T_8a8f1_row16_col4" class="data row16 col4" >0</td>
      <td id="T_8a8f1_row16_col5" class="data row16 col5" >0</td>
      <td id="T_8a8f1_row16_col6" class="data row16 col6" >0</td>
      <td id="T_8a8f1_row16_col7" class="data row16 col7" >0</td>
      <td id="T_8a8f1_row16_col8" class="data row16 col8" >0</td>
      <td id="T_8a8f1_row16_col9" class="data row16 col9" >0</td>
      <td id="T_8a8f1_row16_col10" class="data row16 col10" >0</td>
      <td id="T_8a8f1_row16_col11" class="data row16 col11" >0</td>
      <td id="T_8a8f1_row16_col12" class="data row16 col12" >0</td>
      <td id="T_8a8f1_row16_col13" class="data row16 col13" >0</td>
      <td id="T_8a8f1_row16_col14" class="data row16 col14" >0</td>
      <td id="T_8a8f1_row16_col15" class="data row16 col15" >24</td>
      <td id="T_8a8f1_row16_col16" class="data row16 col16" >253</td>
      <td id="T_8a8f1_row16_col17" class="data row16 col17" >254</td>
      <td id="T_8a8f1_row16_col18" class="data row16 col18" >115</td>
      <td id="T_8a8f1_row16_col19" class="data row16 col19" >0</td>
      <td id="T_8a8f1_row16_col20" class="data row16 col20" >0</td>
      <td id="T_8a8f1_row16_col21" class="data row16 col21" >0</td>
      <td id="T_8a8f1_row16_col22" class="data row16 col22" >0</td>
      <td id="T_8a8f1_row16_col23" class="data row16 col23" >0</td>
      <td id="T_8a8f1_row16_col24" class="data row16 col24" >0</td>
      <td id="T_8a8f1_row16_col25" class="data row16 col25" >0</td>
      <td id="T_8a8f1_row16_col26" class="data row16 col26" >0</td>
      <td id="T_8a8f1_row16_col27" class="data row16 col27" >0</td>
    </tr>
    <tr>
      <th id="T_8a8f1_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_8a8f1_row17_col0" class="data row17 col0" >0</td>
      <td id="T_8a8f1_row17_col1" class="data row17 col1" >0</td>
      <td id="T_8a8f1_row17_col2" class="data row17 col2" >0</td>
      <td id="T_8a8f1_row17_col3" class="data row17 col3" >0</td>
      <td id="T_8a8f1_row17_col4" class="data row17 col4" >0</td>
      <td id="T_8a8f1_row17_col5" class="data row17 col5" >0</td>
      <td id="T_8a8f1_row17_col6" class="data row17 col6" >0</td>
      <td id="T_8a8f1_row17_col7" class="data row17 col7" >0</td>
      <td id="T_8a8f1_row17_col8" class="data row17 col8" >0</td>
      <td id="T_8a8f1_row17_col9" class="data row17 col9" >0</td>
      <td id="T_8a8f1_row17_col10" class="data row17 col10" >0</td>
      <td id="T_8a8f1_row17_col11" class="data row17 col11" >0</td>
      <td id="T_8a8f1_row17_col12" class="data row17 col12" >0</td>
      <td id="T_8a8f1_row17_col13" class="data row17 col13" >0</td>
      <td id="T_8a8f1_row17_col14" class="data row17 col14" >0</td>
      <td id="T_8a8f1_row17_col15" class="data row17 col15" >99</td>
      <td id="T_8a8f1_row17_col16" class="data row17 col16" >253</td>
      <td id="T_8a8f1_row17_col17" class="data row17 col17" >254</td>
      <td id="T_8a8f1_row17_col18" class="data row17 col18" >107</td>
      <td id="T_8a8f1_row17_col19" class="data row17 col19" >0</td>
      <td id="T_8a8f1_row17_col20" class="data row17 col20" >0</td>
      <td id="T_8a8f1_row17_col21" class="data row17 col21" >0</td>
      <td id="T_8a8f1_row17_col22" class="data row17 col22" >0</td>
      <td id="T_8a8f1_row17_col23" class="data row17 col23" >0</td>
      <td id="T_8a8f1_row17_col24" class="data row17 col24" >0</td>
      <td id="T_8a8f1_row17_col25" class="data row17 col25" >0</td>
      <td id="T_8a8f1_row17_col26" class="data row17 col26" >0</td>
      <td id="T_8a8f1_row17_col27" class="data row17 col27" >0</td>
    </tr>
    <tr>
      <th id="T_8a8f1_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_8a8f1_row18_col0" class="data row18 col0" >0</td>
      <td id="T_8a8f1_row18_col1" class="data row18 col1" >0</td>
      <td id="T_8a8f1_row18_col2" class="data row18 col2" >0</td>
      <td id="T_8a8f1_row18_col3" class="data row18 col3" >0</td>
      <td id="T_8a8f1_row18_col4" class="data row18 col4" >0</td>
      <td id="T_8a8f1_row18_col5" class="data row18 col5" >0</td>
      <td id="T_8a8f1_row18_col6" class="data row18 col6" >0</td>
      <td id="T_8a8f1_row18_col7" class="data row18 col7" >0</td>
      <td id="T_8a8f1_row18_col8" class="data row18 col8" >0</td>
      <td id="T_8a8f1_row18_col9" class="data row18 col9" >0</td>
      <td id="T_8a8f1_row18_col10" class="data row18 col10" >0</td>
      <td id="T_8a8f1_row18_col11" class="data row18 col11" >0</td>
      <td id="T_8a8f1_row18_col12" class="data row18 col12" >0</td>
      <td id="T_8a8f1_row18_col13" class="data row18 col13" >0</td>
      <td id="T_8a8f1_row18_col14" class="data row18 col14" >0</td>
      <td id="T_8a8f1_row18_col15" class="data row18 col15" >149</td>
      <td id="T_8a8f1_row18_col16" class="data row18 col16" >253</td>
      <td id="T_8a8f1_row18_col17" class="data row18 col17" >254</td>
      <td id="T_8a8f1_row18_col18" class="data row18 col18" >23</td>
      <td id="T_8a8f1_row18_col19" class="data row18 col19" >0</td>
      <td id="T_8a8f1_row18_col20" class="data row18 col20" >0</td>
      <td id="T_8a8f1_row18_col21" class="data row18 col21" >0</td>
      <td id="T_8a8f1_row18_col22" class="data row18 col22" >0</td>
      <td id="T_8a8f1_row18_col23" class="data row18 col23" >0</td>
      <td id="T_8a8f1_row18_col24" class="data row18 col24" >0</td>
      <td id="T_8a8f1_row18_col25" class="data row18 col25" >0</td>
      <td id="T_8a8f1_row18_col26" class="data row18 col26" >0</td>
      <td id="T_8a8f1_row18_col27" class="data row18 col27" >0</td>
    </tr>
    <tr>
      <th id="T_8a8f1_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_8a8f1_row19_col0" class="data row19 col0" >0</td>
      <td id="T_8a8f1_row19_col1" class="data row19 col1" >0</td>
      <td id="T_8a8f1_row19_col2" class="data row19 col2" >0</td>
      <td id="T_8a8f1_row19_col3" class="data row19 col3" >0</td>
      <td id="T_8a8f1_row19_col4" class="data row19 col4" >0</td>
      <td id="T_8a8f1_row19_col5" class="data row19 col5" >0</td>
      <td id="T_8a8f1_row19_col6" class="data row19 col6" >0</td>
      <td id="T_8a8f1_row19_col7" class="data row19 col7" >0</td>
      <td id="T_8a8f1_row19_col8" class="data row19 col8" >0</td>
      <td id="T_8a8f1_row19_col9" class="data row19 col9" >0</td>
      <td id="T_8a8f1_row19_col10" class="data row19 col10" >0</td>
      <td id="T_8a8f1_row19_col11" class="data row19 col11" >0</td>
      <td id="T_8a8f1_row19_col12" class="data row19 col12" >0</td>
      <td id="T_8a8f1_row19_col13" class="data row19 col13" >0</td>
      <td id="T_8a8f1_row19_col14" class="data row19 col14" >0</td>
      <td id="T_8a8f1_row19_col15" class="data row19 col15" >208</td>
      <td id="T_8a8f1_row19_col16" class="data row19 col16" >254</td>
      <td id="T_8a8f1_row19_col17" class="data row19 col17" >185</td>
      <td id="T_8a8f1_row19_col18" class="data row19 col18" >0</td>
      <td id="T_8a8f1_row19_col19" class="data row19 col19" >0</td>
      <td id="T_8a8f1_row19_col20" class="data row19 col20" >0</td>
      <td id="T_8a8f1_row19_col21" class="data row19 col21" >0</td>
      <td id="T_8a8f1_row19_col22" class="data row19 col22" >0</td>
      <td id="T_8a8f1_row19_col23" class="data row19 col23" >0</td>
      <td id="T_8a8f1_row19_col24" class="data row19 col24" >0</td>
      <td id="T_8a8f1_row19_col25" class="data row19 col25" >0</td>
      <td id="T_8a8f1_row19_col26" class="data row19 col26" >0</td>
      <td id="T_8a8f1_row19_col27" class="data row19 col27" >0</td>
    </tr>
    <tr>
      <th id="T_8a8f1_level0_row20" class="row_heading level0 row20" >20</th>
      <td id="T_8a8f1_row20_col0" class="data row20 col0" >0</td>
      <td id="T_8a8f1_row20_col1" class="data row20 col1" >0</td>
      <td id="T_8a8f1_row20_col2" class="data row20 col2" >0</td>
      <td id="T_8a8f1_row20_col3" class="data row20 col3" >0</td>
      <td id="T_8a8f1_row20_col4" class="data row20 col4" >0</td>
      <td id="T_8a8f1_row20_col5" class="data row20 col5" >0</td>
      <td id="T_8a8f1_row20_col6" class="data row20 col6" >0</td>
      <td id="T_8a8f1_row20_col7" class="data row20 col7" >0</td>
      <td id="T_8a8f1_row20_col8" class="data row20 col8" >0</td>
      <td id="T_8a8f1_row20_col9" class="data row20 col9" >0</td>
      <td id="T_8a8f1_row20_col10" class="data row20 col10" >0</td>
      <td id="T_8a8f1_row20_col11" class="data row20 col11" >0</td>
      <td id="T_8a8f1_row20_col12" class="data row20 col12" >0</td>
      <td id="T_8a8f1_row20_col13" class="data row20 col13" >0</td>
      <td id="T_8a8f1_row20_col14" class="data row20 col14" >0</td>
      <td id="T_8a8f1_row20_col15" class="data row20 col15" >207</td>
      <td id="T_8a8f1_row20_col16" class="data row20 col16" >253</td>
      <td id="T_8a8f1_row20_col17" class="data row20 col17" >184</td>
      <td id="T_8a8f1_row20_col18" class="data row20 col18" >0</td>
      <td id="T_8a8f1_row20_col19" class="data row20 col19" >0</td>
      <td id="T_8a8f1_row20_col20" class="data row20 col20" >0</td>
      <td id="T_8a8f1_row20_col21" class="data row20 col21" >0</td>
      <td id="T_8a8f1_row20_col22" class="data row20 col22" >0</td>
      <td id="T_8a8f1_row20_col23" class="data row20 col23" >0</td>
      <td id="T_8a8f1_row20_col24" class="data row20 col24" >0</td>
      <td id="T_8a8f1_row20_col25" class="data row20 col25" >0</td>
      <td id="T_8a8f1_row20_col26" class="data row20 col26" >0</td>
      <td id="T_8a8f1_row20_col27" class="data row20 col27" >0</td>
    </tr>
    <tr>
      <th id="T_8a8f1_level0_row21" class="row_heading level0 row21" >21</th>
      <td id="T_8a8f1_row21_col0" class="data row21 col0" >0</td>
      <td id="T_8a8f1_row21_col1" class="data row21 col1" >0</td>
      <td id="T_8a8f1_row21_col2" class="data row21 col2" >0</td>
      <td id="T_8a8f1_row21_col3" class="data row21 col3" >0</td>
      <td id="T_8a8f1_row21_col4" class="data row21 col4" >0</td>
      <td id="T_8a8f1_row21_col5" class="data row21 col5" >0</td>
      <td id="T_8a8f1_row21_col6" class="data row21 col6" >0</td>
      <td id="T_8a8f1_row21_col7" class="data row21 col7" >0</td>
      <td id="T_8a8f1_row21_col8" class="data row21 col8" >0</td>
      <td id="T_8a8f1_row21_col9" class="data row21 col9" >0</td>
      <td id="T_8a8f1_row21_col10" class="data row21 col10" >0</td>
      <td id="T_8a8f1_row21_col11" class="data row21 col11" >0</td>
      <td id="T_8a8f1_row21_col12" class="data row21 col12" >0</td>
      <td id="T_8a8f1_row21_col13" class="data row21 col13" >0</td>
      <td id="T_8a8f1_row21_col14" class="data row21 col14" >0</td>
      <td id="T_8a8f1_row21_col15" class="data row21 col15" >207</td>
      <td id="T_8a8f1_row21_col16" class="data row21 col16" >253</td>
      <td id="T_8a8f1_row21_col17" class="data row21 col17" >109</td>
      <td id="T_8a8f1_row21_col18" class="data row21 col18" >0</td>
      <td id="T_8a8f1_row21_col19" class="data row21 col19" >0</td>
      <td id="T_8a8f1_row21_col20" class="data row21 col20" >0</td>
      <td id="T_8a8f1_row21_col21" class="data row21 col21" >0</td>
      <td id="T_8a8f1_row21_col22" class="data row21 col22" >0</td>
      <td id="T_8a8f1_row21_col23" class="data row21 col23" >0</td>
      <td id="T_8a8f1_row21_col24" class="data row21 col24" >0</td>
      <td id="T_8a8f1_row21_col25" class="data row21 col25" >0</td>
      <td id="T_8a8f1_row21_col26" class="data row21 col26" >0</td>
      <td id="T_8a8f1_row21_col27" class="data row21 col27" >0</td>
    </tr>
    <tr>
      <th id="T_8a8f1_level0_row22" class="row_heading level0 row22" >22</th>
      <td id="T_8a8f1_row22_col0" class="data row22 col0" >0</td>
      <td id="T_8a8f1_row22_col1" class="data row22 col1" >0</td>
      <td id="T_8a8f1_row22_col2" class="data row22 col2" >0</td>
      <td id="T_8a8f1_row22_col3" class="data row22 col3" >0</td>
      <td id="T_8a8f1_row22_col4" class="data row22 col4" >0</td>
      <td id="T_8a8f1_row22_col5" class="data row22 col5" >0</td>
      <td id="T_8a8f1_row22_col6" class="data row22 col6" >0</td>
      <td id="T_8a8f1_row22_col7" class="data row22 col7" >0</td>
      <td id="T_8a8f1_row22_col8" class="data row22 col8" >0</td>
      <td id="T_8a8f1_row22_col9" class="data row22 col9" >0</td>
      <td id="T_8a8f1_row22_col10" class="data row22 col10" >0</td>
      <td id="T_8a8f1_row22_col11" class="data row22 col11" >0</td>
      <td id="T_8a8f1_row22_col12" class="data row22 col12" >0</td>
      <td id="T_8a8f1_row22_col13" class="data row22 col13" >0</td>
      <td id="T_8a8f1_row22_col14" class="data row22 col14" >34</td>
      <td id="T_8a8f1_row22_col15" class="data row22 col15" >240</td>
      <td id="T_8a8f1_row22_col16" class="data row22 col16" >253</td>
      <td id="T_8a8f1_row22_col17" class="data row22 col17" >93</td>
      <td id="T_8a8f1_row22_col18" class="data row22 col18" >0</td>
      <td id="T_8a8f1_row22_col19" class="data row22 col19" >0</td>
      <td id="T_8a8f1_row22_col20" class="data row22 col20" >0</td>
      <td id="T_8a8f1_row22_col21" class="data row22 col21" >0</td>
      <td id="T_8a8f1_row22_col22" class="data row22 col22" >0</td>
      <td id="T_8a8f1_row22_col23" class="data row22 col23" >0</td>
      <td id="T_8a8f1_row22_col24" class="data row22 col24" >0</td>
      <td id="T_8a8f1_row22_col25" class="data row22 col25" >0</td>
      <td id="T_8a8f1_row22_col26" class="data row22 col26" >0</td>
      <td id="T_8a8f1_row22_col27" class="data row22 col27" >0</td>
    </tr>
    <tr>
      <th id="T_8a8f1_level0_row23" class="row_heading level0 row23" >23</th>
      <td id="T_8a8f1_row23_col0" class="data row23 col0" >0</td>
      <td id="T_8a8f1_row23_col1" class="data row23 col1" >0</td>
      <td id="T_8a8f1_row23_col2" class="data row23 col2" >0</td>
      <td id="T_8a8f1_row23_col3" class="data row23 col3" >0</td>
      <td id="T_8a8f1_row23_col4" class="data row23 col4" >0</td>
      <td id="T_8a8f1_row23_col5" class="data row23 col5" >0</td>
      <td id="T_8a8f1_row23_col6" class="data row23 col6" >0</td>
      <td id="T_8a8f1_row23_col7" class="data row23 col7" >0</td>
      <td id="T_8a8f1_row23_col8" class="data row23 col8" >0</td>
      <td id="T_8a8f1_row23_col9" class="data row23 col9" >0</td>
      <td id="T_8a8f1_row23_col10" class="data row23 col10" >0</td>
      <td id="T_8a8f1_row23_col11" class="data row23 col11" >0</td>
      <td id="T_8a8f1_row23_col12" class="data row23 col12" >0</td>
      <td id="T_8a8f1_row23_col13" class="data row23 col13" >0</td>
      <td id="T_8a8f1_row23_col14" class="data row23 col14" >47</td>
      <td id="T_8a8f1_row23_col15" class="data row23 col15" >254</td>
      <td id="T_8a8f1_row23_col16" class="data row23 col16" >254</td>
      <td id="T_8a8f1_row23_col17" class="data row23 col17" >93</td>
      <td id="T_8a8f1_row23_col18" class="data row23 col18" >0</td>
      <td id="T_8a8f1_row23_col19" class="data row23 col19" >0</td>
      <td id="T_8a8f1_row23_col20" class="data row23 col20" >0</td>
      <td id="T_8a8f1_row23_col21" class="data row23 col21" >0</td>
      <td id="T_8a8f1_row23_col22" class="data row23 col22" >0</td>
      <td id="T_8a8f1_row23_col23" class="data row23 col23" >0</td>
      <td id="T_8a8f1_row23_col24" class="data row23 col24" >0</td>
      <td id="T_8a8f1_row23_col25" class="data row23 col25" >0</td>
      <td id="T_8a8f1_row23_col26" class="data row23 col26" >0</td>
      <td id="T_8a8f1_row23_col27" class="data row23 col27" >0</td>
    </tr>
    <tr>
      <th id="T_8a8f1_level0_row24" class="row_heading level0 row24" >24</th>
      <td id="T_8a8f1_row24_col0" class="data row24 col0" >0</td>
      <td id="T_8a8f1_row24_col1" class="data row24 col1" >0</td>
      <td id="T_8a8f1_row24_col2" class="data row24 col2" >0</td>
      <td id="T_8a8f1_row24_col3" class="data row24 col3" >0</td>
      <td id="T_8a8f1_row24_col4" class="data row24 col4" >0</td>
      <td id="T_8a8f1_row24_col5" class="data row24 col5" >0</td>
      <td id="T_8a8f1_row24_col6" class="data row24 col6" >0</td>
      <td id="T_8a8f1_row24_col7" class="data row24 col7" >0</td>
      <td id="T_8a8f1_row24_col8" class="data row24 col8" >0</td>
      <td id="T_8a8f1_row24_col9" class="data row24 col9" >0</td>
      <td id="T_8a8f1_row24_col10" class="data row24 col10" >0</td>
      <td id="T_8a8f1_row24_col11" class="data row24 col11" >0</td>
      <td id="T_8a8f1_row24_col12" class="data row24 col12" >0</td>
      <td id="T_8a8f1_row24_col13" class="data row24 col13" >0</td>
      <td id="T_8a8f1_row24_col14" class="data row24 col14" >38</td>
      <td id="T_8a8f1_row24_col15" class="data row24 col15" >244</td>
      <td id="T_8a8f1_row24_col16" class="data row24 col16" >253</td>
      <td id="T_8a8f1_row24_col17" class="data row24 col17" >93</td>
      <td id="T_8a8f1_row24_col18" class="data row24 col18" >0</td>
      <td id="T_8a8f1_row24_col19" class="data row24 col19" >0</td>
      <td id="T_8a8f1_row24_col20" class="data row24 col20" >0</td>
      <td id="T_8a8f1_row24_col21" class="data row24 col21" >0</td>
      <td id="T_8a8f1_row24_col22" class="data row24 col22" >0</td>
      <td id="T_8a8f1_row24_col23" class="data row24 col23" >0</td>
      <td id="T_8a8f1_row24_col24" class="data row24 col24" >0</td>
      <td id="T_8a8f1_row24_col25" class="data row24 col25" >0</td>
      <td id="T_8a8f1_row24_col26" class="data row24 col26" >0</td>
      <td id="T_8a8f1_row24_col27" class="data row24 col27" >0</td>
    </tr>
    <tr>
      <th id="T_8a8f1_level0_row25" class="row_heading level0 row25" >25</th>
      <td id="T_8a8f1_row25_col0" class="data row25 col0" >0</td>
      <td id="T_8a8f1_row25_col1" class="data row25 col1" >0</td>
      <td id="T_8a8f1_row25_col2" class="data row25 col2" >0</td>
      <td id="T_8a8f1_row25_col3" class="data row25 col3" >0</td>
      <td id="T_8a8f1_row25_col4" class="data row25 col4" >0</td>
      <td id="T_8a8f1_row25_col5" class="data row25 col5" >0</td>
      <td id="T_8a8f1_row25_col6" class="data row25 col6" >0</td>
      <td id="T_8a8f1_row25_col7" class="data row25 col7" >0</td>
      <td id="T_8a8f1_row25_col8" class="data row25 col8" >0</td>
      <td id="T_8a8f1_row25_col9" class="data row25 col9" >0</td>
      <td id="T_8a8f1_row25_col10" class="data row25 col10" >0</td>
      <td id="T_8a8f1_row25_col11" class="data row25 col11" >0</td>
      <td id="T_8a8f1_row25_col12" class="data row25 col12" >0</td>
      <td id="T_8a8f1_row25_col13" class="data row25 col13" >0</td>
      <td id="T_8a8f1_row25_col14" class="data row25 col14" >0</td>
      <td id="T_8a8f1_row25_col15" class="data row25 col15" >207</td>
      <td id="T_8a8f1_row25_col16" class="data row25 col16" >253</td>
      <td id="T_8a8f1_row25_col17" class="data row25 col17" >168</td>
      <td id="T_8a8f1_row25_col18" class="data row25 col18" >0</td>
      <td id="T_8a8f1_row25_col19" class="data row25 col19" >0</td>
      <td id="T_8a8f1_row25_col20" class="data row25 col20" >0</td>
      <td id="T_8a8f1_row25_col21" class="data row25 col21" >0</td>
      <td id="T_8a8f1_row25_col22" class="data row25 col22" >0</td>
      <td id="T_8a8f1_row25_col23" class="data row25 col23" >0</td>
      <td id="T_8a8f1_row25_col24" class="data row25 col24" >0</td>
      <td id="T_8a8f1_row25_col25" class="data row25 col25" >0</td>
      <td id="T_8a8f1_row25_col26" class="data row25 col26" >0</td>
      <td id="T_8a8f1_row25_col27" class="data row25 col27" >0</td>
    </tr>
    <tr>
      <th id="T_8a8f1_level0_row26" class="row_heading level0 row26" >26</th>
      <td id="T_8a8f1_row26_col0" class="data row26 col0" >0</td>
      <td id="T_8a8f1_row26_col1" class="data row26 col1" >0</td>
      <td id="T_8a8f1_row26_col2" class="data row26 col2" >0</td>
      <td id="T_8a8f1_row26_col3" class="data row26 col3" >0</td>
      <td id="T_8a8f1_row26_col4" class="data row26 col4" >0</td>
      <td id="T_8a8f1_row26_col5" class="data row26 col5" >0</td>
      <td id="T_8a8f1_row26_col6" class="data row26 col6" >0</td>
      <td id="T_8a8f1_row26_col7" class="data row26 col7" >0</td>
      <td id="T_8a8f1_row26_col8" class="data row26 col8" >0</td>
      <td id="T_8a8f1_row26_col9" class="data row26 col9" >0</td>
      <td id="T_8a8f1_row26_col10" class="data row26 col10" >0</td>
      <td id="T_8a8f1_row26_col11" class="data row26 col11" >0</td>
      <td id="T_8a8f1_row26_col12" class="data row26 col12" >0</td>
      <td id="T_8a8f1_row26_col13" class="data row26 col13" >0</td>
      <td id="T_8a8f1_row26_col14" class="data row26 col14" >0</td>
      <td id="T_8a8f1_row26_col15" class="data row26 col15" >174</td>
      <td id="T_8a8f1_row26_col16" class="data row26 col16" >253</td>
      <td id="T_8a8f1_row26_col17" class="data row26 col17" >151</td>
      <td id="T_8a8f1_row26_col18" class="data row26 col18" >0</td>
      <td id="T_8a8f1_row26_col19" class="data row26 col19" >0</td>
      <td id="T_8a8f1_row26_col20" class="data row26 col20" >0</td>
      <td id="T_8a8f1_row26_col21" class="data row26 col21" >0</td>
      <td id="T_8a8f1_row26_col22" class="data row26 col22" >0</td>
      <td id="T_8a8f1_row26_col23" class="data row26 col23" >0</td>
      <td id="T_8a8f1_row26_col24" class="data row26 col24" >0</td>
      <td id="T_8a8f1_row26_col25" class="data row26 col25" >0</td>
      <td id="T_8a8f1_row26_col26" class="data row26 col26" >0</td>
      <td id="T_8a8f1_row26_col27" class="data row26 col27" >0</td>
    </tr>
    <tr>
      <th id="T_8a8f1_level0_row27" class="row_heading level0 row27" >27</th>
      <td id="T_8a8f1_row27_col0" class="data row27 col0" >0</td>
      <td id="T_8a8f1_row27_col1" class="data row27 col1" >0</td>
      <td id="T_8a8f1_row27_col2" class="data row27 col2" >0</td>
      <td id="T_8a8f1_row27_col3" class="data row27 col3" >0</td>
      <td id="T_8a8f1_row27_col4" class="data row27 col4" >0</td>
      <td id="T_8a8f1_row27_col5" class="data row27 col5" >0</td>
      <td id="T_8a8f1_row27_col6" class="data row27 col6" >0</td>
      <td id="T_8a8f1_row27_col7" class="data row27 col7" >0</td>
      <td id="T_8a8f1_row27_col8" class="data row27 col8" >0</td>
      <td id="T_8a8f1_row27_col9" class="data row27 col9" >0</td>
      <td id="T_8a8f1_row27_col10" class="data row27 col10" >0</td>
      <td id="T_8a8f1_row27_col11" class="data row27 col11" >0</td>
      <td id="T_8a8f1_row27_col12" class="data row27 col12" >0</td>
      <td id="T_8a8f1_row27_col13" class="data row27 col13" >0</td>
      <td id="T_8a8f1_row27_col14" class="data row27 col14" >0</td>
      <td id="T_8a8f1_row27_col15" class="data row27 col15" >0</td>
      <td id="T_8a8f1_row27_col16" class="data row27 col16" >0</td>
      <td id="T_8a8f1_row27_col17" class="data row27 col17" >0</td>
      <td id="T_8a8f1_row27_col18" class="data row27 col18" >0</td>
      <td id="T_8a8f1_row27_col19" class="data row27 col19" >0</td>
      <td id="T_8a8f1_row27_col20" class="data row27 col20" >0</td>
      <td id="T_8a8f1_row27_col21" class="data row27 col21" >0</td>
      <td id="T_8a8f1_row27_col22" class="data row27 col22" >0</td>
      <td id="T_8a8f1_row27_col23" class="data row27 col23" >0</td>
      <td id="T_8a8f1_row27_col24" class="data row27 col24" >0</td>
      <td id="T_8a8f1_row27_col25" class="data row27 col25" >0</td>
      <td id="T_8a8f1_row27_col26" class="data row27 col26" >0</td>
      <td id="T_8a8f1_row27_col27" class="data row27 col27" >0</td>
    </tr>
  </tbody>
</table>




In the code above:
`arr[:,:,0]` is a common way to index into a 3-dimensional NumPy array (or a similar array-like object, such as a PyTorchtensor or PIL image array). ':' means _all elements_ along that axis.  Thus arr[:,:,0] selects all rows (:), all columns (:), and channel 0 (e.g., the red channel in the RGB image, the entire channel would be something like (0,0,0)).


```python
# now compare the difference in shape between with or without selection of channel 0.
print(arr[:,:,0].shape)
print(arr[:,:,:].shape)
```

    (28, 28)
    (28, 28, 3)



```python
# Now let's take a look at the unique values in the array, 
print(np.unique(arr))
```

    [  0   9  23  24  34  38  44  46  47  64  69  71  76  93  99 104 107 109
     111 115 128 137 138 139 145 146 149 151 154 161 168 174 176 180 184 185
     207 208 214 215 221 230 231 240 244 245 251 253 254 255]


These values represents the grayscale image of the digit 7 you see above.  Or how computer stores the image - with only numbers!  So the task of hand-writing recoginition of digits boils down to identify the similarity between the image and the reference digit images in numerical terms.  So we will go ahead and convert all images into numerical vectors, take the average value of 3s and 7s, and use the average vector as the representation of the digit.  This is called the _vector space model_. This will be the baseline of our model.  We then use this to predict the digit of a new image, and see if the new image is more similar to 3 or 7.  This is the simplest form (and the most intuitive) of machine learning.  A discriminate reader might notice that we have converted the NumPy array into a tensor (`img_t=tensor(arr[:,:,0])`).  This is a common practice in fastai.  Fastai is built on top of PyTorch, whose models are designed to work with torch.Tensor objects.  This means that all the layers, loss functions, and optimizers in PyTorch expect their inputs and outputs to be tensors.  Tensor supports automatic differentiation (autograd), meaning PyTorch can automatically compute graients for tensors during backpropagation, whilst their counterpart NumPy array can't.  One of the important characteristics of tensor is that it supports GPU acceleration, meaning that tensors can be moved to and operated on GPUs for faster computation (tensor.cuda()), while NumPy arrays are always on the CPU.


```python
tensor(arr).permute(2,0,1).shape
```




    torch.Size([3, 28, 28])



The code above converts the NumPy array to a PyTorch tensor and then permutes the dimensions.  The original shape is (28, 28, 3) and after permutation, it becomes (3, 28, 28).  This is the format that PyTorch expects for images: (channels, height, width).

## Baseline Images

**Stacking**
`torch.stack(seven_tensors)`: seven_tensors is a list of 2D tensors, where each tensor represents an image of digit "7".  Each tensor has a shape of [height, width] (28x28).
Stacking means that we combine all these individual 2D tensors into a single 3D tensor by adding a new dimension at the beginning.  The resutig tensor has a shape of [number_of_images, height, width].  The `.mean(0)` calculates the mean along dimension 0 (the first dimension), which is the "number_of_images" dimension.  This effectively averages all the "7" images pixel by pixel, resulting in a single 2D tensor that represents the average "7" image.


```python
# look at the 3 and 7 images
three_path = train_path/'3'
seven_path = train_path/'7'

three_tensors = [tensor(Image.open(o)) for o in three_path.ls()]
seven_tensors = [tensor(Image.open(o)) for o in seven_path.ls()]

# stack all the 3s and 7s
stacked_threes = torch.stack(three_tensors).float()/255
stacked_sevens = torch.stack(seven_tensors).float()/255

# calculate the mean of all 3s and 7s
mean3 = stacked_threes.mean(0)
mean7 = stacked_sevens.mean(0)

# show the mean images
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1) # (1, 2, 1) means that the subplot is 1 row, 2 columns, and this is the first plot (starting from upper left corner)
plt.imshow(mean3)
plt.title('Mean of 3s')
plt.subplot(1, 2, 2)
plt.imshow(mean7)
plt.title('Mean of 7s')
plt.tight_layout()
```


    
![png](/assets/images/uploads/fastai-part3_files/fastai-part3_17_0.png)
    


The code above calculates the mean of all 3s and 7s.  The mean image shows the average pixel values across all images of that digit.  This gives us a template of what a typical 3 and 7 look like.


```python
# calculate the difference between the mean 3 and mean 7
diff = mean3 - mean7
plt.imshow(diff)
```




    <matplotlib.image.AxesImage at 0x1770fc7a0>




    
![png](/assets/images/uploads/fastai-part3_files/fastai-part3_19_1.png)
    


The difference image shows where the two digits differ the most.  Bright areas indicate where 3s have higher pixel values than 7s, and dark areas indicate where 7s have higher pixel values than 3s.


```python
# calculate the similarity between each 3 and the mean 3, and each 3 and the mean 7
three_similarity = [((t - mean3)**2).mean().item() for t in stacked_threes]
three_to_seven_similarity = [((t - mean7)**2).mean().item() for t in stacked_threes]

# calculate the similarity between each 7 and the mean 7, and each 7 and the mean 3
seven_similarity = [((t - mean7)**2).mean().item() for t in stacked_sevens]
seven_to_three_similarity = [((t - mean3)**2).mean().item() for t in stacked_sevens]

# plot the similarities
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(three_similarity, label='3 to mean 3')
plt.plot(three_to_seven_similarity, label='3 to mean 7')
plt.legend()
plt.title('3s: Similarity to mean 3 vs mean 7')

plt.subplot(1, 2, 2)
plt.plot(seven_similarity, label='7 to mean 7')
plt.plot(seven_to_three_similarity, label='7 to mean 3')
plt.legend()
plt.title('7s: Similarity to mean 7 vs mean 3')
plt.tight_layout()
```


    
![png](/assets/images/uploads/fastai-part3_files/fastai-part3_21_0.png)
    




**Similarity Calculation**
`((tensor(Image.open(o)).float() - seven_avg)**2).mean()` computes the similarity between a given image `o` and the average "7" image.  In essence, this is the _mean squared error (MES)_ between each "7" images and the average "7" image.  Lower MSE indicates higher similarity.  MSE is a common metric used to measure the similarity (or differences) between predicted value and actual value.  It says that we apply the square to the difference (which may be positive or negative), the apply the square root to take the effect of square away.  But in this process, we turn the differences to only positive values.

**Plot**
- The y-axis represents the MSE.
- The x-axis represents the index of the image in the dataset.
- This plot shows that each "3" image has a lower MSE with the average "3" than with the average "7".
- **Note** we didn't scale the y-axis in order to visualize the differences more clearly.

## Training a Neural Network on fullset of MNIST

Now that we've explored the MNIST dataset and understood its structure, let's train a neural network to classify the digits. We'll use the full MNIST dataset instead of just the sample we've been working with.


```python
# Download the full MNIST dataset
path = untar_data(URLs.MNIST)
path
```




    Path('/Users/zlu/.fastai/data/mnist_png')




```python
# create a DataBlock for the MNIST dataset
mnist = DataBlock(
    blocks=(ImageBlock(cls=PILImageBW), CategoryBlock), 
    get_items=get_image_files, 
    splitter=GrandparentSplitter(train_name='training', valid_name='testing'),
    get_y=parent_label,
    batch_tfms=Normalize()
)
```

### DataBlock
A fastai DataBlock is a high-level API abstraction for data processing, like a data pipeline.  It takes the raw dataset and prepare it to train machine learning models.  There is a a lot of logic underneath this process.  First you need to specify how to get the data (file path for example).  Then you need to specify how they should be labeled; what transformations to apply (e.g., resizing, augmentataion); how to split the data (training/validation) and finally what types of inputs/outputs are involved.  So our code snippet above defines:

1. `blocks=(ImageBlock(cls=PILImageBW), CategoryBlock)` specifies that inputs are black and white images (PILImageBW) and the outputs are categories (digits 0-9).
2. `get_items=get_image_files` gets all image files from our path.
3. `splitter=GrandparentSplitter(train_name='training', valid_name='testing')` splits the data based on the grandparent directory name. In the MNIST dataset, the structure is `path/training/digit/image.jpg` and `path/testing/digit/image.jpg`.
4. `get_y=parent_label` gets the label (digit) from the parent directory name.
5. `batch_tfms=Normalize()`: This normalizes our images to have zero mean and unit variance, which helps with training.


```python
# create the DataLoaders
dls = mnist.dataloaders(path, bs=64) #bs is batch size

# show a random batch (sample) of images
dls.show_batch(max_n=9, figsize=(4,4))
```


    
![png](/assets/images/uploads/fastai-part3_files/fastai-part3_27_0.png)
    


Now let's create and train a convolutional neural network (CNN) for digit classification.  This model is designed specifically for single-channel grayscale images.  We will not get too deep into CNN for now.  We will have a separate tutorial dedicated to CNN.


```python
# create a simple custom CNN model for MNIST
class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # input: 1 channel (grayscale), Output: 16 feature maps
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # input: 16 channels, Output: 32 feature maps
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # input: 32 channels, Output: 64 feature maps
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # fully connected layers
        # after 3 pooling layers of factor 2, the 28x28 image is reduced to 3x3
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)  # 10 classes for digits 0-9
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        
        # flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x

# create a custom model
model = MnistCNN()

# create a learner with our custom CNN model
learn = Learner(dls, model, loss_func=nn.CrossEntropyLoss(), metrics=accuracy)

# print the model structure to understand it better
print("Model structure:")
print(learn.model)
```

    Model structure:
    MnistCNN(
      (conv1): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu1): ReLU()
      (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu2): ReLU()
      (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (conv3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu3): ReLU()
      (pool3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (fc1): Linear(in_features=576, out_features=128, bias=True)
      (relu4): ReLU()
      (fc2): Linear(in_features=128, out_features=10, bias=True)
    )



```python
# train the model for 1 epochs
learn.fine_tune(1)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.088422</td>
      <td>0.065185</td>
      <td>0.979500</td>
      <td>00:34</td>
    </tr>
  </tbody>
</table>




<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.030792</td>
      <td>0.024418</td>
      <td>0.991900</td>
      <td>00:33</td>
    </tr>
  </tbody>
</table>


## Model Evaluation
Let's examine the model's performance on the **validation** set:


```python
# get predictions
preds, targets = learn.get_preds()
pred_classes = preds.argmax(dim=1)

# create confusion matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(targets, pred_classes)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>








    
![png](/assets/images/uploads/fastai-part3_files/fastai-part3_32_2.png)
    


## Making Predictions

Now let's use our trained model to make predictions on some test images:


```python
# get some test images
test_files = get_image_files(path/'testing')
# take 9 random sample images
random_test_files = random.sample(test_files, 10)

test_dl = learn.dls.test_dl(random_test_files)

# make predictions
preds, _ = learn.get_preds(dl=test_dl)
pred_classes = preds.argmax(dim=1)

# Display the images and predictions
fig, axes = plt.subplots(2, 5, figsize=(6, 3))
axes = axes.flatten()

for i, (img_file, pred) in enumerate(zip(random_test_files, pred_classes)):
    img = PILImage.create(img_file)
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"Predicted: {pred.item()}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>








    
![png](/assets/images/uploads/fastai-part3_files/fastai-part3_34_2.png)
    


## Comparing with Our Template Matching Approach

Earlier, we used a template matching approach with MSE to distinguish between digits 3 and 7. Let's compare the performance of our neural network with that approach:


```python
# get all test images of 3 and 7
test_3_files = get_image_files(path/'testing'/'3')
test_7_files = get_image_files(path/'testing'/'7')

# create a test dataset with only 3s and 7s
test_files_3_7 = test_3_files[:50] + test_7_files[:50]
test_dl_3_7 = learn.dls.test_dl(test_files_3_7)

# make predictions
preds, _ = learn.get_preds(dl=test_dl_3_7)
pred_classes = preds.argmax(dim=1)

# calculate accuracy for 3s and 7s
true_labels = torch.tensor([3] * 50 + [7] * 50)
correct = (pred_classes == true_labels).float().mean()
print(f"Neural Network Accuracy on 3s and 7s: {correct.item():.4f}")
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>







    Neural Network Accuracy on 3s and 7s: 0.9900


## Visualizing Feature Maps

Let's visualize what features our CNN is learning by examining the activations of the first convolutional layer:


```python
# get a batch of images
x, y = dls.one_batch()

# get the first convolutional layer from our custom model
#gFor our custom MnistCNN model, we can directly access the conv1 attribute
conv1 = learn.model.conv1

# apply the first conv layer to get activations
with torch.no_grad():
    activations = conv1(x)

# visualize the activations for the first image
# our custom model has 16 filters in the first layer
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
axes = axes.flatten()

# show the original image
axes[0].imshow(x[0][0].cpu(), cmap='gray')
axes[0].set_title(f"Original Image: {y[0].item()}")
axes[0].axis('off')

# show the activation maps for the first 15 filters
for i in range(1, 16):
    axes[i].imshow(activations[0, i-1].detach().cpu(), cmap='viridis')
    axes[i].set_title(f"Filter {i}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# also visualize the filter weights
weights = conv1.weight.data.cpu()
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
axes = axes.flatten()

# show the original image again
axes[0].imshow(x[0][0].cpu(), cmap='gray')
axes[0].set_title(f"Original Image: {y[0].item()}")
axes[0].axis('off')

# show the weights for the first 15 filters
for i in range(1, 16):
    # Each filter has only one input channel (grayscale)
    axes[i].imshow(weights[i-1, 0], cmap='viridis')
    axes[i].set_title(f"Filter {i} Weights")
    axes[i].axis('off')

plt.tight_layout()
plt.show()
```


    
![png](/assets/images/uploads/fastai-part3_files/fastai-part3_38_0.png)
    



    
![png](/assets/images/uploads/fastai-part3_files/fastai-part3_38_1.png)
    


## Conclusion

In this notebook, we've explored the MNIST dataset in depth and trained a convolutional neural network to classify handwritten digits. We've seen how the model performs and visualized some of its internal representations.

Key takeaways:
1. Neural networks can achieve high accuracy on digit classification tasks.
2. The first layers of a CNN learn simple features like edges and textures.
3. The template matching approach we explored earlier is much simpler but less accurate than a full neural network.
4. fastai makes it easy to build, train, and interpret deep learning models.
