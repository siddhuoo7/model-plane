import React, { Suspense, lazy } from 'react'
import { BrowserRouter, Routes, Route, NavLink, useLocation } from 'react-router-dom'
import {
  Content,
  Header,
  HeaderName,
  HeaderGlobalBar,
  HeaderGlobalAction,
  SideNav,
  SideNavItems,
  SideNavLink,
  SkipToContent,
  Theme,
  Tag,
} from '@carbon/react'
import {
  Dashboard,
  DataTable as DataTableIcon,
  Cost,
  PlugFilled,
  RouterWifi,
  Api,
  SettingsAdjust,
  Catalog,
  TrainSpeed,
  ShieldAlert,
  TransformPipeline,
} from '@carbon/icons-react'
import '@carbon/react/index.scss'
import { SimulatorPanel } from './components/SimulatorPanel'
import { LoadingState } from './components/States'
import { AuthProvider, useAuth } from './AuthContext'

// Lazy-load pages
const DashboardPage      = lazy(() => import('./pages/DashboardPage'))
const RequestsPage       = lazy(() => import('./pages/RequestExplorerPage'))
const CostPage           = lazy(() => import('./pages/CostAnalyticsPage'))
const CatalogPage        = lazy(() => import('./pages/CatalogPage'))
const RoutingPage        = lazy(() => import('./pages/RoutingConfigPage'))
const ProvidersPage      = lazy(() => import('./pages/ProvidersPage'))
const MlPage             = lazy(() => import('./pages/MlTrainingPage'))
const ApiIntegrationPage = lazy(() => import('./pages/ApiIntegrationPage'))
const SettingsPage       = lazy(() => import('./pages/SettingsPage'))
const SecurityPage       = lazy(() => import('./pages/SecurityPage'))
const CompressionPage    = lazy(() => import('./pages/CompressionPage'))
const LoginPage          = lazy(() => import('./pages/LoginPage'))

// ── Nav helpers ───────────────────────────────────────────────────────────────

const NAV_SECTION_LABEL: React.CSSProperties = {
  fontSize: '0.625rem',
  fontWeight: 700,
  letterSpacing: '0.1em',
  textTransform: 'uppercase',
  color: 'rgba(244,244,244,0.35)',
  padding: '1.25rem 1rem 0.375rem',
  display: 'block',
  userSelect: 'none',
}

const NAV_DIVIDER: React.CSSProperties = {
  height: 1,
  background: 'rgba(255,255,255,0.07)',
  margin: '0.5rem 0',
}

function NavItem({ to, icon: Icon, label, badge }: {
  to: string; icon: React.ComponentType<any>; label: string; badge?: string
}) {
  const location = useLocation()
  const active = location.pathname === to || (to !== '/admin/' && location.pathname.startsWith(to))
  return (
    <SideNavLink
      as={NavLink}
      to={to}
      isActive={active}
      renderIcon={Icon as React.ComponentType}
      style={active ? {
        background: 'rgba(15,98,254,0.18)',
        borderLeft: '3px solid #0f62fe',
        borderRadius: '0 4px 4px 0',
      } : undefined}
    >
      <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
        <span>{label}</span>
        {badge && (
          <span style={{
            background: '#0f62fe', color: '#fff',
            borderRadius: 10, padding: '1px 7px',
            fontSize: '0.5625rem', fontWeight: 700, letterSpacing: '0.04em',
            marginLeft: 'auto', flexShrink: 0,
          }}>{badge}</span>
        )}
      </span>
    </SideNavLink>
  )
}

// ── Sidebar footer ─────────────────────────────────────────────────────────

function SidebarFooter() {
  const { user } = useAuth()
  return (
    <div style={{
      position: 'absolute', bottom: 0, left: 0, right: 0,
      padding: '0.75rem 1rem',
      borderTop: '1px solid rgba(255,255,255,0.08)',
      background: 'linear-gradient(0deg, rgba(0,0,0,0.35) 0%, transparent 100%)',
    }}>
      {user && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.625rem', marginBottom: '0.625rem' }}>
          <div style={{
            width: 30, height: 30, borderRadius: '50%',
            background: 'linear-gradient(135deg, #0f62fe 0%, #8a3ffc 100%)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            color: '#fff', fontWeight: 700, fontSize: '0.8125rem', flexShrink: 0,
            boxShadow: '0 0 0 2px rgba(15,98,254,0.4)',
          }}>
            {(user.name || user.email).charAt(0).toUpperCase()}
          </div>
          <div style={{ overflow: 'hidden', flex: 1 }}>
            <p style={{ fontSize: '0.75rem', fontWeight: 600, color: 'rgba(244,244,244,0.9)', margin: 0, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {user.name || user.email}
            </p>
            <p style={{ fontSize: '0.625rem', color: 'rgba(244,244,244,0.4)', margin: 0, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {user.email}
            </p>
          </div>
        </div>
      )}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={{ fontSize: '0.5625rem', color: 'rgba(244,244,244,0.28)', fontWeight: 600, letterSpacing: '0.08em' }}>
          MODEL PLANE · v0.1.0
        </span>
      </div>
    </div>
  )
}

// ── Authenticated shell ───────────────────────────────────────────────────────

function Shell() {
  const [simOpen, setSimOpen] = React.useState(false)

  return (
    <Theme theme="g100">
      <SkipToContent />

      <Header aria-label="Model Plane Admin">
        <HeaderName href="/admin/" prefix="">
          <span style={{
            background: 'linear-gradient(135deg, #0f62fe 0%, #8a3ffc 100%)',
            color: '#fff', borderRadius: 6,
            padding: '2px 8px', fontSize: '0.75rem', fontWeight: 700,
            letterSpacing: '0.05em', marginRight: '0.5rem',
          }}>MP</span>
          <span style={{ fontWeight: 600 }}>Model Plane</span>
          <span style={{ color: '#a8a8a8', fontWeight: 400, marginLeft: '0.375rem', fontSize: '0.875rem' }}>Admin</span>
        </HeaderName>

        <HeaderGlobalBar>
          <NavLink to="/admin/api-integration" style={{ display: 'flex', alignItems: 'center' }}>
            <HeaderGlobalAction aria-label="API Integration" tooltipAlignment="end">
              <Api size={20} />
            </HeaderGlobalAction>
          </NavLink>
          <NavLink to="/admin/settings" style={{ display: 'flex', alignItems: 'center' }}>
            <HeaderGlobalAction aria-label="Settings" tooltipAlignment="end">
              <SettingsAdjust size={20} />
            </HeaderGlobalAction>
          </NavLink>
          <HeaderGlobalAction
            aria-label={simOpen ? 'Close simulator' : 'Open simulator'}
            tooltipAlignment="end"
            onClick={() => setSimOpen(v => !v)}
          >
            <RouterWifi size={20} />
          </HeaderGlobalAction>
        </HeaderGlobalBar>
      </Header>

      <SideNav
        aria-label="Navigation"
        expanded isFixedNav isChildOfHeader={false}
        addFocusListeners={false} addMouseListeners={false}
        style={{ paddingBottom: 100, overflowY: 'auto' }}
      >
        <SideNavItems>
          {/* Monitoring */}
          <span style={NAV_SECTION_LABEL}>Monitoring</span>
          <NavItem to="/admin/"         icon={Dashboard}     label="Dashboard" />
          <NavItem to="/admin/requests" icon={DataTableIcon} label="Requests" />
          <NavItem to="/admin/cost"     icon={Cost}          label="Cost Analytics" />

          <div style={NAV_DIVIDER} />

          {/* Configuration */}
          <span style={NAV_SECTION_LABEL}>Configuration</span>
          <NavItem to="/admin/catalog"     icon={Catalog}           label="Model Catalog" />
          <NavItem to="/admin/routing"     icon={RouterWifi}        label="Routing &amp; Classification" />
          <NavItem to="/admin/compression" icon={TransformPipeline} label="Compression" />
          <NavItem to="/admin/providers"   icon={PlugFilled}        label="Providers" />
          <NavItem to="/admin/ml"          icon={TrainSpeed}        label="ML &amp; Training" />

          <div style={NAV_DIVIDER} />

          {/* Security */}
          <span style={NAV_SECTION_LABEL}>Security</span>
          <NavItem to="/admin/security" icon={ShieldAlert} label="Security &amp; Governance" />

          <div style={NAV_DIVIDER} />

          {/* Developer */}
          <span style={NAV_SECTION_LABEL}>Developer</span>
          <NavItem to="/admin/api-integration" icon={Api}           label="API Integration" />
          <NavItem to="/admin/settings"        icon={SettingsAdjust} label="Settings" />
        </SideNavItems>

        <SidebarFooter />
      </SideNav>

      <Theme theme="white">
        <Content style={{ marginLeft: 256, background: '#f4f4f4', minHeight: '100vh' }}>
          <div style={{ display: 'flex', minHeight: '100vh' }}>
            <div style={{ flex: 1, overflow: 'auto' }}>
              <Suspense fallback={<LoadingState />}>
                <Routes>
                  <Route path="/admin/"                element={<DashboardPage />} />
                  <Route path="/admin/requests"        element={<RequestsPage />} />
                  <Route path="/admin/cost"    element={<CostPage />} />
                  <Route path="/admin/catalog" element={<CatalogPage />} />
                  <Route path="/admin/routing"         element={<RoutingPage />} />
                  <Route path="/admin/providers"       element={<ProvidersPage />} />
                  <Route path="/admin/ml"              element={<MlPage />} />
                  <Route path="/admin/api-integration" element={<ApiIntegrationPage />} />
                  <Route path="/admin/settings"        element={<SettingsPage />} />
                  <Route path="/admin/security"        element={<SecurityPage />} />
                  <Route path="/admin/compression"     element={<CompressionPage />} />
                  <Route path="*"                      element={<DashboardPage />} />
                </Routes>
              </Suspense>
            </div>
            {simOpen && (
              <div style={{
                borderLeft: '1px solid #e0e0e0',
                minWidth: 380, overflowY: 'auto',
                background: '#fff',
                boxShadow: '-4px 0 12px rgba(0,0,0,0.06)',
              }}>
                <SimulatorPanel />
              </div>
            )}
          </div>
        </Content>
      </Theme>
    </Theme>
  )
}

// ── Auth gate ─────────────────────────────────────────────────────────────────

function AuthGate() {
  const { token, loading } = useAuth()

  if (loading) {
    return (
      <div style={{ display: 'flex', height: '100vh', alignItems: 'center', justifyContent: 'center', background: '#161616' }}>
        <LoadingState />
      </div>
    )
  }

  if (!token) {
    return (
      <Suspense fallback={null}>
        <Routes>
          <Route path="*" element={<LoginPage />} />
        </Routes>
      </Suspense>
    )
  }

  return <Shell />
}

// ── Root ─────────────────────────────────────────────────────────────────────

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AuthGate />
      </AuthProvider>
    </BrowserRouter>
  )
}
