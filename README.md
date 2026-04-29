# FRAPPE

## Full Input, Residual Output Autoencoding with Projection Pursuit Encoder

[project page](UT-SysML.github.io/FRAPPE)
[paper](danjacobellis.net/_static/FRAPPE.pdf) 

Modern media compression standards have reached a plateau in terms of the rate-distortion-complexity trade-off. For many applications, this has severely limited the ability to offload computation to the cloud. Recent advances in DNN-based autoencoders have shown potential to break free of this plateau, but remain impractical for three reasons: (1) they cannot easily adapt to different rates; (2) they require prohibitive encoding costs to before matching compression efficiency; (3) they require power-hungry GPUs to encode in real-time. To address these issues, we propose a new type of residual autoencoder (FRAPPE) that uses the \textbf{F}ull input to predict the \textbf{R}esidual output via a \textbf{P}rojection-\textbf{P}ursuit \textbf{E}ncoder. FRAPPE's encoding objective naturally sorts latent channels by importance, allowing zero overhead variable-rate coding using a single set of encoder weights. Unlike previous residual autoencoders, which require an inherently sequential encoding workflow with the decoder in the loop, FRAPPE allows all latent channels to be encoded in parallel. At bitrates near 0.1\,bpp (compression ratio of 240:1) FRAPPE provides better perceptual quality than AVIF with 47 times faster encoding, making it capable of real-time (1080p, 30fps) CPU encoding even at the highest quality. Our code, pre-trained models, and results are available at \url{https://UT-SysML.github.io/FRAPPE}

## Installation

FRAPPE is integrated in the compressors https://danjacobellis.net/compressors library:

`pip install compressors`
