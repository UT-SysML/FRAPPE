"""FRAPPE: Full-Input Residual-Output Autoencoding with Projection Pursuit Encoders.

Inference-only port of the rae_ica prototype. Neural-network building blocks
originally from gigatorch.ops are vendored inline: quantizer primitives in
`quantize.py`, decoder primitives in `model.py`. Refresh by re-copying from
the upstream gigatorch package.
"""
