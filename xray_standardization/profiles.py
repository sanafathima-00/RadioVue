"""Model-export profiles.

The two PA profiles share all policy.  Resolution is their only difference.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    target_size: int
    resize_strategy: str = "fixed_square"
    experimental: bool = False


_PROFILES = {
    "svdrr_pa_256": Profile("svdrr_pa_256", 256),
    "svdrr_pa_512": Profile("svdrr_pa_512", 512),
    "svdrr_pa_256_aspect_pad_experimental": Profile(
        "svdrr_pa_256_aspect_pad_experimental", 256, "aspect_pad", True
    ),
    "svdrr_pa_512_aspect_pad_experimental": Profile(
        "svdrr_pa_512_aspect_pad_experimental", 512, "aspect_pad", True
    ),
}


def get_profile(name: str) -> Profile:
    """Return a known profile or raise a clear error."""
    try:
        return _PROFILES[name]
    except KeyError as error:
        choices = ", ".join(sorted(_PROFILES))
        raise ValueError(f"Unknown profile '{name}'. Choose one of: {choices}") from error


def profile_names() -> list[str]:
    return sorted(_PROFILES)
