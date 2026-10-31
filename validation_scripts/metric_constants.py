"""Constants shared by the validators, so every script and the docs agree."""

# Gaussian pre-blur (in px, applied to both images) before SSIM_blur. A
# sigma of 4 discounts detail finer than ~10 px (FWHM ~9.4 px).
SSIM_BLUR_SIGMA = 4.0
