/**
 * Tier badge — colour-coded chip matching the plan's four-colour palette.
 * simple=green, medium=blue, complex=orange, reasoning=pink
 */
import React from 'react'
import { Tag } from '@carbon/react'

const TIER_TYPE: Record<string, 'green' | 'blue' | 'teal' | 'purple'> = {
  simple: 'green',
  medium: 'blue',
  complex: 'teal',
  reasoning: 'purple',
}

interface TierBadgeProps {
  tier: string
  size?: 'sm' | 'md'
}

export function TierBadge({ tier, size = 'sm' }: TierBadgeProps) {
  const type = TIER_TYPE[tier?.toLowerCase()] ?? 'blue'
  return (
    <Tag type={type} size={size}>
      {tier ?? '—'}
    </Tag>
  )
}
