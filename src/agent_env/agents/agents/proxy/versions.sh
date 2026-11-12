# What install.sh and bake.sh share, sourced by both. A uv bump changes UV_VERSION, both checksums and UV_IMAGE_DIGEST,
# the digest of ghcr.io/astral-sh/uv:$UV_VERSION's image index, which the bake recipe pins.
UV_VERSION=0.13.1
UV_SHA256_X86_64=6c86040a8f045ab9adc9e3ec4cdafa03fd00e9a988ed2bf28971c7fc2d0debd0
UV_SHA256_AARCH64=345a8308e4d1835cdfaadf692836fd3137a6d45864773f05ad7bf59442b1ba16
UV_IMAGE_DIGEST=sha256:4a1f3666816dafadd20ceb71eac9a7d56d7ef30279f7d64048519ab901ada336
PYTHON_VERSION=3.12
BAKED=/opt/agentenv-proxy  # where bake.sh puts the proxy in an image
