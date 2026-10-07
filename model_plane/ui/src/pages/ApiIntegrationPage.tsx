/**
 * API Integration page.
 * The API key used in snippets is loaded from the first key in Settings.
 * No manual key entry needed — manage keys in the Settings page.
 */
import React from 'react'
import {
  Tabs,
  Tab,
  TabList,
  TabPanels,
  TabPanel,
  Tile,
  Tag,
  Button,
} from '@carbon/react'
import { Copy, CheckmarkFilled, Link, Locked, ArrowRight } from '@carbon/icons-react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import { NavLink } from 'react-router-dom'

const BASE_URL = typeof window !== 'undefined'
  ? `${window.location.protocol}//${window.location.hostname}${window.location.port ? ':' + window.location.port : ''}`
  : 'http://localhost:8081'

const V1    = `${BASE_URL}/v1`
const ADMIN = `${BASE_URL}/admin/api`

// ── Copy button ──────────────────────────────────────────────────────────────

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = React.useState(false)
  const copy = () => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true); setTimeout(() => setCopied(false), 2000)
    })
  }
  return (
    <button onClick={copy} title="Copy" style={{
      position: 'absolute', top: 10, right: 10,
      background: copied ? '#24a148' : '#393939',
      border: 'none', borderRadius: 4, padding: '4px 8px',
      cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4,
      color: '#fff', fontSize: '0.75rem', transition: 'background 0.2s',
    }}>
      {copied ? <CheckmarkFilled size={14} /> : <Copy size={14} />}
      {copied ? 'Copied' : 'Copy'}
    </button>
  )
}

function CodeBlock({ code }: { code: string }) {
  return (
    <div style={{ position: 'relative', marginTop: '0.5rem' }}>
      <pre style={{
        background: '#161616', color: '#f4f4f4',
        padding: '1rem 3.5rem 1rem 1rem', borderRadius: 6,
        fontSize: '0.8125rem', lineHeight: 1.65,
        overflowX: 'auto', margin: 0,
        fontFamily: "'IBM Plex Mono', 'Courier New', monospace",
        whiteSpace: 'pre',
      }}>
        <code>{code.trim()}</code>
      </pre>
      <CopyButton text={code.trim()} />
    </div>
  )
}

function EndpointRow({ method, path, desc, auth = true }: {
  method: string; path: string; desc: string; auth?: boolean
}) {
  const colors: Record<string, string> = {
    GET: '#24a148', POST: '#0f62fe', PUT: '#8a3ffc', DELETE: '#da1e28',
  }
  return (
    <div style={{
      display: 'flex', alignItems: 'flex-start', gap: '0.75rem',
      padding: '0.625rem 0', borderBottom: '1px solid #e0e0e0',
    }}>
      <span style={{
        background: colors[method] ?? '#6f6f6f', color: '#fff',
        borderRadius: 3, padding: '1px 7px', fontSize: '0.6875rem',
        fontWeight: 700, minWidth: 44, textAlign: 'center', flexShrink: 0,
        fontFamily: 'monospace', marginTop: 2,
      }}>{method}</span>
      <code style={{ fontSize: '0.8125rem', color: '#161616', flexShrink: 0, minWidth: 260, fontFamily: "'IBM Plex Mono', monospace" }}>{path}</code>
      <span style={{ fontSize: '0.8125rem', color: '#525252', flex: 1 }}>{desc}</span>
      {auth && <Locked size={14} style={{ color: '#f1c21b', flexShrink: 0, marginTop: 2 }} title="Requires Authorization: Bearer <key>" />}
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section style={{ marginBottom: '2rem' }}>
      <h3 style={{ fontSize: '1rem', fontWeight: 600, color: '#161616', marginBottom: '0.75rem' }}>{title}</h3>
      {children}
    </section>
  )
}

// ── Main page ────────────────────────────────────────────────────────────────

export default function ApiIntegrationPage() {
  // Load the first API key from settings — used to populate all snippets
  const { data: keysData } = useQuery({
    queryKey: ['api-keys'],
    queryFn: api.settings.listApiKeys,
    retry: false,
  })
  const firstKey = keysData?.keys?.[0]
  const keyLabel = firstKey ? firstKey.prefix : 'YOUR_API_KEY'
  const bearerHeader = `Authorization: Bearer ${keyLabel}`
  const hasKey = Boolean(firstKey)

  return (
    <div style={{ padding: '1.5rem 2rem', maxWidth: 920 }}>
      <h2 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '0.375rem' }}>API Integration</h2>
      <p style={{ fontSize: '0.875rem', color: '#525252', marginBottom: '1.5rem' }}>
        Model Plane exposes an OpenAI-compatible endpoint at{' '}
        <code style={{ background: '#f4f4f4', padding: '1px 6px', borderRadius: 3 }}>/v1/chat/completions</code>.
        Drop in your existing OpenAI SDK — point <code>base_url</code> here and you're done.
      </p>

      {/* Base URL */}
      <Tile style={{ marginBottom: '1.25rem', background: '#edf5ff', borderLeft: '4px solid #0f62fe', borderRadius: 6, padding: '0.875rem 1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
          <Link size={16} style={{ color: '#0f62fe', flexShrink: 0 }} />
          <span style={{ fontSize: '0.875rem', fontWeight: 600, color: '#0043ce' }}>Base URL</span>
          <code style={{ fontSize: '0.875rem', color: '#161616', background: '#d0e2ff', padding: '2px 8px', borderRadius: 4 }}>{V1}</code>
          <span style={{ fontSize: '0.8125rem', color: '#525252', marginLeft: '0.5rem' }}>
            Admin at <code style={{ background: '#f4f4f4', padding: '1px 5px', borderRadius: 3 }}>{ADMIN}</code>
          </span>
        </div>
      </Tile>

      {/* Auth / key status */}
      <div style={{
        marginBottom: '1.5rem',
        background: hasKey ? '#defbe6' : '#fff8e1',
        border: `1px solid ${hasKey ? '#a7f0ba' : '#f1c21b'}`,
        borderRadius: 8, padding: '1rem 1.25rem',
        display: 'flex', alignItems: 'center', gap: '0.875rem', flexWrap: 'wrap',
      }}>
        <Locked size={18} style={{ color: hasKey ? '#24a148' : '#f1c21b', flexShrink: 0 }} />
        <div style={{ flex: 1 }}>
          {hasKey ? (
            <>
              <p style={{ fontWeight: 600, margin: 0, fontSize: '0.9rem' }}>API key active</p>
              <p style={{ margin: '0.2rem 0 0', fontSize: '0.8125rem', color: '#525252' }}>
                Snippets below use key <code style={{ background: '#f4f4f4', padding: '1px 5px', borderRadius: 3 }}>{firstKey?.prefix}</code>{' '}
                ({firstKey?.label}). All requests require <code>Authorization: Bearer &lt;key&gt;</code>.
              </p>
            </>
          ) : (
            <>
              <p style={{ fontWeight: 600, margin: 0, fontSize: '0.9rem' }}>No API keys configured</p>
              <p style={{ margin: '0.2rem 0 0', fontSize: '0.8125rem', color: '#525252' }}>
                Auth is currently disabled (dev mode). Generate a key in Settings to enable it.
              </p>
            </>
          )}
        </div>
        <Button kind="ghost" size="sm" renderIcon={ArrowRight} as={NavLink} to="/admin/settings">
          {hasKey ? 'Manage keys' : 'Go to Settings'}
        </Button>
      </div>

      <Tabs>
        <TabList aria-label="Integration tabs">
          <Tab>Endpoints</Tab>
          <Tab>cURL</Tab>
          <Tab>Python</Tab>
          <Tab>JavaScript / TS</Tab>
        </TabList>
        <TabPanels>

          {/* Endpoints */}
          <TabPanel>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem', fontSize: '0.8125rem', color: '#6f6f6f' }}>
              <Locked size={14} style={{ color: '#f1c21b' }} />
              <span>= requires <code>Authorization: Bearer &lt;key&gt;</code> when keys are configured</span>
            </div>

            <Section title="Chat completions (OpenAI-compatible)">
              <EndpointRow method="POST" path="/v1/chat/completions"  auth desc="Submit a chat request. Routes automatically. Supports streaming." />
              <EndpointRow method="GET"  path="/v1/models"            auth={false} desc="List available deployment names." />
            </Section>
            <Section title="Monitoring">
              <EndpointRow method="GET" path="/admin/api/health"   auth={false} desc="Health check — routing mode, ML status, deployments." />
              <EndpointRow method="GET" path="/admin/api/traffic"  auth={false} desc="Aggregate metrics. ?window_hours=24" />
              <EndpointRow method="GET" path="/admin/api/requests" auth={false} desc="Ring buffer of recent requests." />
              <EndpointRow method="GET" path="/admin/api/cost"     auth={false} desc="Cost breakdown by provider / tier / task." />
            </Section>
            <Section title="Catalog management">
              <EndpointRow method="GET"    path="/admin/api/catalog"        auth={false} desc="List deployments." />
              <EndpointRow method="POST"   path="/admin/api/catalog"        auth desc="Add a deployment." />
              <EndpointRow method="PUT"    path="/admin/api/catalog/{name}" auth desc="Update a deployment." />
              <EndpointRow method="DELETE" path="/admin/api/catalog/{name}" auth desc="Delete a deployment." />
              <EndpointRow method="POST"   path="/admin/api/catalog/reload" auth desc="Hot-reload models.yaml." />
            </Section>
            <Section title="Routing config">
              <EndpointRow method="GET"  path="/admin/api/routing/config"  auth={false} desc="Read routing config." />
              <EndpointRow method="PUT"  path="/admin/api/routing/config"  auth desc="Write and persist config." />
              <EndpointRow method="POST" path="/admin/api/routing/preview" auth desc="Before/after diff against pending config." />
            </Section>
            <Section title="Classifiers">
              <EndpointRow method="GET" path="/admin/api/classifiers"        auth={false} desc="List classifiers." />
              <EndpointRow method="PUT" path="/admin/api/classifiers/{name}" auth desc="Enable/disable or set active." />
            </Section>
            <Section title="ML & Training">
              <EndpointRow method="GET"  path="/admin/api/ml/status"                  auth={false} desc="ML model status." />
              <EndpointRow method="POST" path="/admin/api/ml/retrain"                 auth desc="Trigger retrain job." />
              <EndpointRow method="GET"  path="/admin/api/ml/retrain/{job_id}/stream" auth={false} desc="SSE retrain progress." />
              <EndpointRow method="GET"  path="/admin/api/ml/training-data"           auth={false} desc="Inspect training data." />
              <EndpointRow method="POST" path="/admin/api/ml/training-data/upload"    auth desc="Upload CSV/JSONL." />
              <EndpointRow method="POST" path="/admin/api/ml/activate"                auth desc="Activate model file." />
            </Section>
            <Section title="Simulate">
              <EndpointRow method="POST" path="/admin/api/simulate" auth={false} desc="Dry-run routing — no real call made." />
            </Section>
            <Section title="Tenants & Alerts">
              <EndpointRow method="GET"    path="/admin/api/tenants"           auth={false} desc="List tenant policies." />
              <EndpointRow method="POST"   path="/admin/api/tenants"           auth desc="Create policy." />
              <EndpointRow method="PUT"    path="/admin/api/tenants/{id}"      auth desc="Update policy." />
              <EndpointRow method="DELETE" path="/admin/api/tenants/{id}"      auth desc="Delete policy." />
              <EndpointRow method="GET"    path="/admin/api/alerts"            auth={false} desc="List alerts." />
              <EndpointRow method="POST"   path="/admin/api/alerts"            auth desc="Create alert." />
              <EndpointRow method="PUT"    path="/admin/api/alerts/{alert_id}" auth desc="Update alert." />
              <EndpointRow method="DELETE" path="/admin/api/alerts/{alert_id}" auth desc="Delete alert." />
            </Section>
          </TabPanel>

          {/* cURL */}
          <TabPanel>
            <Section title="Chat completion">
              <CodeBlock code={`curl ${V1}/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "${bearerHeader}" \\
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Explain attention mechanisms in transformers"}]
  }'`} />
            </Section>
            <Section title="Streaming">
              <CodeBlock code={`curl ${V1}/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "${bearerHeader}" \\
  -d '{"model":"auto","stream":true,"messages":[{"role":"user","content":"Tell me a story"}]}'`} />
            </Section>
            <Section title="Health check (no auth)">
              <CodeBlock code={`curl ${ADMIN}/health`} />
            </Section>
            <Section title="Simulate (no auth)">
              <CodeBlock code={`curl -X POST ${ADMIN}/simulate \\
  -H "Content-Type: application/json" \\
  -d '{"messages":[{"role":"user","content":"Debug this segfault"}]}'`} />
            </Section>
          </TabPanel>

          {/* Python */}
          <TabPanel>
            <Section title="Install">
              <CodeBlock code="pip install openai" />
            </Section>
            <Section title="Client setup">
              <CodeBlock code={`from openai import OpenAI

client = OpenAI(
    base_url="${V1}",
    api_key="${keyLabel}",
)`} />
            </Section>
            <Section title="Chat completion">
              <CodeBlock code={`response = client.chat.completions.create(
    model="auto",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "Explain how transformers work."},
    ],
)
print(response.choices[0].message.content)`} />
            </Section>
            <Section title="Streaming">
              <CodeBlock code={`stream = client.chat.completions.create(
    model="auto", stream=True,
    messages=[{"role": "user", "content": "Tell me a short story."}],
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)`} />
            </Section>
          </TabPanel>

          {/* JavaScript */}
          <TabPanel>
            <Section title="Install">
              <CodeBlock code="npm install openai  # or: pnpm add openai" />
            </Section>
            <Section title="Client setup">
              <CodeBlock code={`import OpenAI from 'openai'

const client = new OpenAI({
  baseURL: '${V1}',
  apiKey: '${keyLabel}',
  dangerouslyAllowBrowser: true,  // omit for server-side Node.js
})`} />
            </Section>
            <Section title="Chat completion">
              <CodeBlock code={`const response = await client.chat.completions.create({
  model: 'auto',
  messages: [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user',   content: 'Summarise this article.' },
  ],
})
console.log(response.choices[0].message.content)`} />
            </Section>
            <Section title="Streaming">
              <CodeBlock code={`const stream = await client.chat.completions.create({
  model: 'auto', stream: true,
  messages: [{ role: 'user', content: 'Write a poem.' }],
})
for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content ?? '')
}`} />
            </Section>
          </TabPanel>

        </TabPanels>
      </Tabs>
    </div>
  )
}
