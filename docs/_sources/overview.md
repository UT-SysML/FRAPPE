# FRAPPE

## Full Input, Residual Output Autoencoding with Projection Pursuit Encoder

[code](https://github.com/UT-SysML/FRAPPE)

[paper](https://danjacobellis.net/_static/FRAPPE.pdf) 

Media compression standards have reached a plateau in terms of the rate-distortion-complexity trade-off, limiting the ability to offload expensive AI perception to the cloud in applications like robotics, wearables, and remote sensing. DNN-based codecs improve compression efficiency, but at a cost: they cannot easily adapt to large changes in available bitrate, and real-time encoding requires expensive, power-hungry GPUs that prohibit use on low-cost or resource-constrained platforms. To address these limitations, we propose a novel autoencoding framework (FRAPPE) that uses the **F**ull input to predict the **R**esidual output via a **P**rojection-**P**ursuit **E**ncoder. FRAPPE's encoding objective naturally sorts latent channels by importance, allowing zero-overhead variable-rate coding. Unlike RNN-based learned codecs, whose encoder consumes the previous reconstruction's residual, or RVQ-style codecs, whose codebooks must be applied sequentially, FRAPPE's analysis path is an embarrassingly parallel DAG of independent input projections. Using FRAPPE, we build a variable-rate RGB image codec (FRAPPE-Image), and evaluate its rate-distortion-complexity trade-off against standard image codecs. At high compression ratios (~0.1 bpp) FRAPPE-Image provides higher perceptual quality than AVIF with 47x faster encoding, making it capable of real-time 1080p, 30fps CPU-only encoding.

## Installation

FRAPPE is integrated in the compressors library:

`pip install compressors`
