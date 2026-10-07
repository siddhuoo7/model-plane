/**
 * Global error / loading boundary helpers used across all pages.
 */
import React from 'react'
import { InlineLoading, InlineNotification } from '@carbon/react'

export function LoadingState({ description = 'Loading…' }: { description?: string }) {
  return (
    <div style={{ padding: '2rem' }}>
      <InlineLoading description={description} status="active" />
    </div>
  )
}

export function ErrorState({ message }: { message: string }) {
  return (
    <div style={{ padding: '2rem' }}>
      <InlineNotification
        kind="error"
        title="Error"
        subtitle={message}
        hideCloseButton
      />
    </div>
  )
}
