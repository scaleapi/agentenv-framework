"""VM image artifact for referencing pre-built VM images."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, model_validator

from agent_env.artifact.artifact import Artifact


class VMImageArtifact(Artifact):
    """A VM image artifact that references a pre-built VM image.

    Supports ECR containerdisk images (Ubuntu/Windows) and non-ECR images
    (e.g. Orka images for macOS). At least one of ecr_url or image_name
    must be provided.

    Unlike DockerImageArtifact, this does not upload anything to S3.

    Examples:
        # ECR-based (Ubuntu/Windows)
        VMImageArtifact.put(id="cua-ubuntu", description="...", ecr_url="<account>.dkr.ecr.<region>.amazonaws.com/cua-ubuntu:1.0.4")

        # Non-ECR (macOS Orka)
        VMImageArtifact.put(id="cua-macos", description="...", image_name="tahoe-cua", os="macos")
    """

    type: Literal["vm_image"] = "vm_image"
    description: str = Field(description="Human-readable description of the VM image")
    ecr_url: Optional[str] = Field(default=None, description="Full ECR URL for the containerdisk image")
    image_name: Optional[str] = Field(default=None, description="Image name for non-ECR registries (e.g. Orka)")
    os: Optional[str] = Field(default=None, description="OS identifier (e.g. 'macos') — only needed when the sandbox API requires it")
    disk_size_gb: Optional[float] = Field(
        default=None,
        description=(
            "Root disk size in GiB for this VM image. MUST be >= the disk size the image was "
            "baked with; if it is smaller the guest cannot find its root filesystem and drops "
            "to the initramfs shell (\"Gave up waiting for root file system device\" / "
            "\"ALERT! UUID=... does not exist\"). Keep this in sync with the image build. "
            "None -> falls back to the deploying env's default."
        ),
    )
    cpu: Optional[float] = Field(default=None, description="vCPU count for the VM; None -> falls back to the deploying env's default")
    memory_mb: Optional[int] = Field(default=None, description="Memory in MiB for the VM; None -> falls back to the deploying env's default")
    sandbox_type: Optional[str] = Field(
        default=None,
        description=(
            "Sandbox backend this image deploys to (e.g. 'modal_vm', or a name from [sandbox.providers])"
        ),
    )

    @model_validator(mode="after")
    def validate_image_source(self):
        if not self.ecr_url and not self.image_name:
            raise ValueError("At least one of ecr_url or image_name must be provided")
        return self

    @property
    def image(self) -> str:
        """Return the image reference (ECR URL or image name)."""
        return self.ecr_url or self.image_name

    @classmethod
    def put(
        cls,
        id: str,
        *,
        description: str,
        ecr_url: Optional[str] = None,
        image_name: Optional[str] = None,
        os: Optional[str] = None,
        disk_size_gb: Optional[float] = None,
        cpu: Optional[float] = None,
        memory_mb: Optional[int] = None,
        sandbox_type: Optional[str] = None,
    ) -> "VMImageArtifact":
        from agent_env.artifact.store import get_artifact_store

        store = get_artifact_store()
        version = store.next_version(id)

        instance = cls(
            id=id,
            version=version,
            description=description,
            ecr_url=ecr_url,
            image_name=image_name,
            os=os,
            disk_size_gb=disk_size_gb,
            cpu=cpu,
            memory_mb=memory_mb,
            sandbox_type=sandbox_type,
        )
        return store.put_document(instance)
