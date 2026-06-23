"""LanguageProfile registry: resolve a BCP-47 tag to its profile (R2/R3/R4).

Mirrors `providers/__init__.py` — a dict keyed by the BCP-47 primary tag plus a
`resolve(tag)` that walks the fallback chain (full tag -> drop trailing subtags
-> generic). Adding a language is one new entry in `data.py` plus one line in the
registry below; no node edits (R2). `resolve` always returns a profile (generic
for the unknown); `is_profiled` reports whether a tag matched a real entry, so
Config/preflight can gate an unprofiled target behind --allow-fallback (R4).

The module is pure (no nodes, no providers, no heavy deps), so `Config` resolves
it through a lazy accessor without an import cycle.
"""

from loro.profiles.base import LanguageProfile
from loro.profiles.data import ENGLISH, FRENCH, GENERIC, SPANISH, VIETNAMESE

__all__ = ["LanguageProfile", "resolve", "is_profiled", "registered_tags",
           "GENERIC"]

# Keyed by BCP-47 primary subtag (lowercase). Region/script subtags resolve via
# the fallback chain in `resolve` (es-MX -> es), so only base languages register.
_REGISTRY: dict[str, LanguageProfile] = {
    "en": ENGLISH,
    "vi": VIETNAMESE,
    "fr": FRENCH,
    "es": SPANISH,
}


def _chain(tag: str) -> list[str]:
    """The lookup keys for a tag, most- to least-specific: the full normalized
    tag then each prefix with a trailing subtag dropped (fr-CA -> [fr-ca, fr])."""
    parts = tag.strip().lower().replace("_", "-").split("-")
    return ["-".join(parts[:i]) for i in range(len(parts), 0, -1)]


def resolve(tag: str) -> LanguageProfile:
    """Resolve `tag` to a profile, walking the fallback chain to GENERIC. A
    registered base language wins for any of its region/script variants
    (resolve("es-MX") -> SPANISH); an unknown tag falls to GENERIC (R3)."""
    for key in _chain(tag):
        if key in _REGISTRY:
            return _REGISTRY[key]
    return GENERIC


def is_profiled(tag: str) -> bool:
    """True when `tag` resolves to a real registered profile (directly or via a
    region/script variant), False when it only resolves to GENERIC. The gate for
    the --allow-fallback policy (R4)."""
    return resolve(tag) is not GENERIC


def registered_tags() -> list[str]:
    """The registered BCP-47 base tags (for CLI help / diagnostics)."""
    return sorted(_REGISTRY)
