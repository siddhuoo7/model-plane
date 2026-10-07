"""TierResolver — maps a ComplexityTier to a concrete DeploymentConfig.

Called by the routing pipeline Stage 3.6 (tier mode) after the ML recommender
returns a predicted tier. Uses two-pass selection logic:

  Pass 1:  exact match  — deployments whose `tier` tag == requested tier
  Pass 2:  next-tier-up — if no exact match, try the next tier above
  Pass 3:  top-ranked fallback — use the first candidate from the ranked list

Deployment ranking comes from the scorer's ordered candidate list (highest score
first), so provider preferences and cost signals are already baked in.
"""

from __future__ import annotations

from model_plane.classifier.taxonomy import ComplexityTier
from model_plane.logging_setup import get_logger
from model_plane.registry.catalog import DeploymentConfig

log = get_logger(__name__)

# Tier progression order (ascending complexity)
_TIER_ORDER: list[ComplexityTier] = [
    ComplexityTier.SIMPLE,
    ComplexityTier.MEDIUM,
    ComplexityTier.COMPLEX,
    ComplexityTier.REASONING,
]


def _next_tier_up(tier: ComplexityTier) -> ComplexityTier | None:
    """Return the next tier above *tier*, or None if already at REASONING."""
    idx = _TIER_ORDER.index(tier)
    if idx + 1 < len(_TIER_ORDER):
        return _TIER_ORDER[idx + 1]
    return None


class TierResolver:
    """Resolves a predicted ``ComplexityTier`` to a ``DeploymentConfig``.

    Usage::

        resolver = TierResolver()
        dep = resolver.resolve(ComplexityTier.COMPLEX, ranked_candidates)
    """

    def resolve(
        self,
        tier: ComplexityTier,
        ranked_candidates: list[DeploymentConfig],
    ) -> DeploymentConfig | None:
        """Select a deployment for *tier* from *ranked_candidates*.

        Args:
            tier:              The target ``ComplexityTier`` recommended by ML.
            ranked_candidates: Candidates already filtered by policy and sorted by
                               scorer (best first). Must not be empty.

        Returns:
            The best matching ``DeploymentConfig``, or ``None`` if the list is empty.
        """
        if not ranked_candidates:
            return None

        # Pass 1: exact tier match
        exact = [d for d in ranked_candidates if d.tier == tier.value]
        if exact:
            log.debug(
                "tier_resolver_exact_match",
                tier=tier.value,
                selected=exact[0].name,
                candidates=[d.name for d in exact],
            )
            return exact[0]

        # Pass 2: next-tier-up fallback (never downgrade)
        up = _next_tier_up(tier)
        while up is not None:
            up_match = [d for d in ranked_candidates if d.tier == up.value]
            if up_match:
                log.debug(
                    "tier_resolver_tier_up_fallback",
                    requested_tier=tier.value,
                    fallback_tier=up.value,
                    selected=up_match[0].name,
                )
                return up_match[0]
            up = _next_tier_up(up)

        # Pass 3: top-ranked fallback (catches "small" tier in catalog or any gap)
        log.debug(
            "tier_resolver_top_ranked_fallback",
            tier=tier.value,
            selected=ranked_candidates[0].name,
        )
        return ranked_candidates[0]
