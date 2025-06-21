---
layout: post
title: Image Processing with R
date: 2025-06-21 09:00 +0800
tages:
  - R
  - Image Processing
---

Image processing in R, using packages like `imager`, enables powerful
analysis and manipulation of digital images. This blog demonstrates key
techniques—enhancement, denoising, and histogram equalization—using the
`imager` package, with visual results to illustrate their effects, based
on concepts from the "Image Data Analysis" document.

## Understanding Digital Images

A digital image is a 2D function
$f(x, y): \mathbb{R}^2 \rightarrow \mathbb{R}$, where $(x, y)$ are
spatial coordinates, and $f(x, y)$ represents pixel intensity. In
digital images, pixels have discrete locations and values. For 8-bit
grayscale images, intensities range from 0 (black) to 255 (white).
Images are categorized as:

- **Binary**: Pixels are 0 (black) or 1 (white).
- **Grayscale**: Pixel values in $\{0, \ldots, 255\}$.
- **Color**: Three channels (RGB), each in $\{0, \ldots, 255\}$.

## Image Processing with `imager`

The `imager` package, built on CImg, supports loading, manipulating, and
visualizing images. Below, we demonstrate key operations with code and
visual outputs using the built-in "parrots.png" image.

### 1. Image Acquisition

Load and display an image:

```r
library(imager)
file <- system.file('extdata/parrots.png', package='imager')
img <- load.image(file)
plot(img, main="Original Parrots Image")
```

**Result**:
![Original Parrots Image](/assets/images/uploads/plots/original_parrots.png)

### 2. Image Enhancement (Blurring)

Blurring enhances images for specific applications, e.g., smoothing
details:

```r
img_blurry <- isoblur(img, sigma=10)
plot(img_blurry, main="Blurred Image (sigma=10)")
```

**Result**: ![Blurred Parrots Image](/assets/images/uploads/plots/blurred_parrots.png) _Note:
The blurred image appears smoother, reducing fine details like feather
textures._

### 3. Image Denoising

Denoising removes noise while preserving structure. Add noise and apply
anisotropic blurring:

```r
img_noisy <- img + 0.5 * rnorm(prod(dim(img)))
img_denoised <- blur_anisotropic(img_noisy, ampl=1e3, sharp=0.3)
layout(t(1:2))
plot(img_noisy, main="Noisy Image")
plot(img_denoised, main="Denoised Image (Anisotropic)")
```

**Result**: ![Noisy vs Denoised](/assets/images/uploads/plots/noisy_vs_denoised_parrots.png)
_Note: The noisy image shows random speckles, while the denoised image
restores clarity, preserving edges._

### 4. Histogram Equalization

Histogram equalization enhances contrast by redistributing pixel
intensities:

```r
img_gray <- grayscale(img)
f <- ecdf(img_gray)
img_equalized <- f(img_gray) %>% as.cimg(dim=dim(img_gray))
layout(t(1:2))
plot(img_gray, main="Grayscale Image")
plot(img_equalized, main="Histogram Equalized Image")
```

**Result**: ![Grayscale vs
Equalized](/assets/images/uploads/plots/grayscale_vs_equalized_parrots.png) _Note: The
equalized image has improved contrast, making details like color
variations more distinct._

### 5. Morphological Processing

Thresholding segments objects by intensity:

```r
img_gray <- grayscale(img)
threshold(img_gray, "20%") %>% plot(main="Thresholded Image (20%)")
```

**Result**: ![Thresholded Image](/assets/images/uploads/plots/thresholded_parrots.png) _Note:
Thresholding creates a binary image, highlighting brighter regions
(e.g., white feathers) against darker ones._

### 6. Putting Everything Together

```r
library(imager)

# Set up directory for saving plots (optional: create if it doesn't exist)
output_dir <- "plots"
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}

# Image Acquisition
file <- system.file('extdata/parrots.png', package='imager')
img <- load.image(file)
plot(img, main="Original Parrots Image")
dev.copy(png, file.path(output_dir, "original_parrots.png"))
dev.off()

# Image Enhancement (Blurring)
img_blurry <- isoblur(img, sigma=10)
plot(img_blurry, main="Blurred Image (sigma=10)")
dev.copy(png, file.path(output_dir, "blurred_parrots.png"))
dev.off()

# Image Denoising
img_noisy <- img + 0.5 * rnorm(prod(dim(img)))
img_denoised <- blur_anisotropic(img_noisy, ampl=1e3, sharp=0.3)
layout(t(1:2))
plot(img_noisy, main="Noisy Image")
plot(img_denoised, main="Denoised Image (Anisotropic)")
dev.copy(png, file.path(output_dir, "noisy_vs_denoised_parrots.png"))
dev.off()

# Histogram Equalization
img_gray <- grayscale(img)
f <- ecdf(img_gray)
img_equalized <- f(img_gray) %>% as.cimg(dim=dim(img_gray))
layout(t(1:2))
plot(img_gray, main="Grayscale Image")
plot(img_equalized, main="Histogram Equalized Image")
dev.copy(png, file.path(output_dir, "grayscale_vs_equalized_parrots.png"))
dev.off()

# Morphological Processing
img_gray <- grayscale(img)
threshold(img_gray, "20%") %>% plot(main="Thresholded Image (20%)")
dev.copy(png, file.path(output_dir, "thresholded_parrots.png"))
dev.off()
```

## Applications

Image processing in R is used in: - **Automotive**: Lane detection,
obstacle warning. - **Medical**: Diagnostic imaging, surgical
assistance. - **Security**: Face recognition, surveillance. - **Media**:
Special effects, image editing.

## Conclusion

The `imager` package in R simplifies image processing tasks like
enhancement, denoising, and histogram equalization. Visual results
demonstrate how these techniques transform images, improving quality or
extracting features. Explore `imager` and `imagerExtra` for advanced
applications.

**Resources**:

- [imagerPackage](https://cran.r-project.org/web/packages/imager/imager.pdf)
- [Getting Started with
  imager](https://cran.r-project.org/web/packages/imager/vignettes/gettingstarted.html)
- [imagerExtra
  Guide](https://cran.r-project.org/web/packages/imagerExtra/vignettes/gettingstarted.html)
