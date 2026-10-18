# A prepared Docker host image for CoreWeave sandboxes: docker:dind with the tools the provider's host setup otherwise
# installs at boot, so a sandbox whose network policy blocks the Alpine package mirror still starts. Push it to a
# registry the sandbox can pull from and set `image` in [sandbox.providers.coreweave.config].
FROM public.ecr.aws/docker/library/docker:dind

RUN apk add --no-cache bash curl e2fsprogs
