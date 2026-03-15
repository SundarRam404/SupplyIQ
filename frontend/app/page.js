'use client'

import { useState } from 'react'
import {
  Activity,
  ArrowRight,
  BadgeAlert,
  BarChart3,
  CheckCircle2,
  FileText,
  ShieldCheck,
  Sparkles,
  TrendingUp,
  UploadCloud,
  WandSparkles,
} from 'lucide-react'
import HeroIllustration from '../components/HeroIllustration'
import MarkdownCard from '../components/MarkdownCard'

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL

const defaultScenario = 'Demand increase (custom %)'

const tabs = [
  { id: 'forecast', label: 'Demand Forecasting' },
  { id: 'risk', label: 'Risk Detection' },
  { id: 'contract', label: 'Contract Intelligence' },
  { id: 'decision', label: 'Decision Engine' },
  { id: 'scenario', label: 'Scenario Simulator' },
]

const featureCards = [
  {
    icon: BarChart3,
    title: 'Demand Forecasting',
    text: 'Forecast future demand, visualize trend direction, and surface inventory implications in a polished business-ready view.',
  },
  {
    icon: BadgeAlert,
    title: 'Supply Risk Detection',
    text: 'Flag anomalies, volatility spikes, and operational irregularities before they silently damage procurement decisions.',
  },
  {
    icon: FileText,
    title: 'Contract Intelligence',
    text: 'Turn supplier PDFs into structured intelligence covering payment terms, obligations, risks, and operational clauses.',
  },
  {
    icon: WandSparkles,
    title: 'Executive Recommendations',
    text: 'Convert fragmented analysis into clear leadership-ready actions with strong AI-supported business reasoning.',
  },
]

const valuePills = [
  'Built around real supply chain workflows',
  'Portfolio-ready premium product UI',
  'Interactive AI workspace for live demos',
  'Designed to impress recruiters instantly',
]

const commandMetrics = [
  { label: 'AI Modules', value: '5' },
  { label: 'Input Types', value: 'CSV + PDF' },
  { label: 'Decision Layer', value: 'Live' },
]

const pipelineSteps = [
  'Upload ERP demand data and supplier contract documents.',
  'Run forecasting, anomaly detection, contract analysis, and scenario testing.',
  'Present charts, risk signals, and executive guidance in one command center.',
]

async function apiCall(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, options)
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.detail || 'Request failed')
  return data
}

export default function HomePage() {
  const [activeTab, setActiveTab] = useState('forecast')
  const [dashboardHtml, setDashboardHtml] = useState('')
  const [csvStatus, setCsvStatus] = useState('Upload ERP demand data to unlock forecasting and risk insights.')
  const [pdfStatus, setPdfStatus] = useState('Upload a supplier contract to activate document intelligence.')
  const [forecast, setForecast] = useState(null)
  const [anomalies, setAnomalies] = useState(null)
  const [contractReport, setContractReport] = useState('')
  const [contractQuestion, setContractQuestion] = useState('What payment terms does this contract define?')
  const [contractAnswer, setContractAnswer] = useState('')
  const [decisionReport, setDecisionReport] = useState('')
  const [scenarioName, setScenarioName] = useState(defaultScenario)
  const [customPercent, setCustomPercent] = useState(30)
  const [customScenarioText, setCustomScenarioText] = useState('')
  const [scenarioReport, setScenarioReport] = useState('')
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')

  const updateDashboard = (html) => {
    if (html) setDashboardHtml(html)
  }

  const handleCsvUpload = async (file) => {
    if (!file) return
    setBusy('Uploading ERP data...')
    setError('')
    try {
      const formData = new FormData()
      formData.append('file', file)
      const data = await apiCall('/upload/csv', { method: 'POST', body: formData })
      setCsvStatus(data.status)
      updateDashboard(data.dashboard_html)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy('')
    }
  }

  const handlePdfUpload = async (file) => {
    if (!file) return
    setBusy('Indexing contract PDF...')
    setError('')
    try {
      const formData = new FormData()
      formData.append('file', file)
      const data = await apiCall('/upload/pdf', { method: 'POST', body: formData })
      setPdfStatus(data.status)
      updateDashboard(data.dashboard_html)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy('')
    }
  }

  const runForecast = async () => {
    setBusy('Generating forecast...')
    setError('')
    try {
      const data = await apiCall('/forecast', { method: 'POST' })
      setForecast(data)
      updateDashboard(data.dashboard_html)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy('')
    }
  }

  const runAnomalies = async () => {
    setBusy('Scanning for supply anomalies...')
    setError('')
    try {
      const data = await apiCall('/anomalies', { method: 'POST' })
      setAnomalies(data)
      updateDashboard(data.dashboard_html)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy('')
    }
  }

  const runContractReport = async () => {
    setBusy('Generating contract intelligence...')
    setError('')
    try {
      const data = await apiCall('/contract-report', { method: 'POST' })
      setContractReport(data.report)
      updateDashboard(data.dashboard_html)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy('')
    }
  }

  const askContract = async () => {
    setBusy('Answering contract question...')
    setError('')
    try {
      const data = await apiCall('/ask-contract', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: contractQuestion }),
      })
      setContractAnswer(data.answer)
      updateDashboard(data.dashboard_html)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy('')
    }
  }

  const runDecision = async () => {
    setBusy('Building executive recommendation...')
    setError('')
    try {
      const data = await apiCall('/decision', { method: 'POST' })
      setDecisionReport(data.report)
      updateDashboard(data.dashboard_html)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy('')
    }
  }

  const runScenario = async () => {
    setBusy('Simulating scenario...')
    setError('')
    try {
      const data = await apiCall('/scenario', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scenario_name: scenarioName,
          custom_percent: Number(customPercent),
          custom_scenario_text: customScenarioText,
        }),
      })
      setScenarioReport(data.report)
      updateDashboard(data.dashboard_html)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy('')
    }
  }

  return (
    <main className="page-shell">
      <div className="ambient ambient-1" />
      <div className="ambient ambient-2" />
      <div className="ambient ambient-3" />

      <section className="hero section-card">
        <div className="hero-copy">
          <div className="eyebrow"><Sparkles size={16} /> SupplyIQ · AI Supply Chain Intelligence</div>
          <h1>From raw supply data to boardroom-ready decisions.</h1>
          <p>
            SupplyIQ is an intelligent command center for forecasting demand, detecting risk,
            understanding contracts, simulating disruptions, and generating executive-grade recommendations.
          </p>

          <div className="hero-actions">
            <a href="#workspace" className="primary-btn">Launch Command Center <ArrowRight size={18} /></a>
            <a href="#intelligence" className="secondary-btn">Explore Capabilities</a>
          </div>

          <div className="value-pill-row">
            {valuePills.map((item) => (
              <span key={item} className="value-pill">{item}</span>
            ))}
          </div>
        </div>

        <div className="hero-art-stack">
          <div className="hero-art-panel section-card">
            <HeroIllustration />
          </div>
          <div className="command-mini-card">
            <div className="mini-kicker">Command Summary</div>
            <div className="mini-stats">
              {commandMetrics.map((item) => (
                <div key={item.label} className="mini-stat">
                  <span>{item.label}</span>
                  <strong>{item.value}</strong>
                </div>
              ))}
            </div>
            <div className="signal-row">
              <span className="signal positive"><TrendingUp size={14} /> Demand signal active</span>
              <span className="signal neutral"><ShieldCheck size={14} /> Contract AI ready</span>
            </div>
          </div>
        </div>
      </section>

      <section id="intelligence" className="feature-zone">
        <div className="section-heading">
          <div className="section-label">Core Intelligence Layer</div>
          <h2>Everything important sits in one polished AI workspace</h2>
          <p>No filler sections. No developer text. Just the product, the pipeline, and the results that matter.</p>
        </div>

        <div className="feature-grid">
          {featureCards.map(({ icon: Icon, title, text }) => (
            <article className="feature-card" key={title}>
              <div className="icon-wrap"><Icon size={20} /></div>
              <h3>{title}</h3>
              <p>{text}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="story-grid">
        <article className="section-card story-card">
          <div className="section-label">How It Works</div>
          <h2>Simple flow, premium experience</h2>
          <div className="pipeline-list">
            {pipelineSteps.map((step, index) => (
              <div key={step} className="pipeline-item">
                <div className="pipeline-index">0{index + 1}</div>
                <p>{step}</p>
              </div>
            ))}
          </div>
        </article>

        <article className="section-card story-card accent-card">
          <div className="section-label">Why It Feels Strong</div>
          <h2>Built like a serious product, not a class submission</h2>
          <div className="why-list">
            <div><CheckCircle2 size={18} /> Multi-module AI flow in a single interface</div>
            <div><CheckCircle2 size={18} /> Clean visual hierarchy for recruiter demos</div>
            <div><CheckCircle2 size={18} /> Real inputs, real outputs, real product direction</div>
            <div><CheckCircle2 size={18} /> Strong enough for portfolio, internship, and interview walkthroughs</div>
          </div>
        </article>
      </section>

      <section className="section-card dashboard-shell">
        <div className="section-heading compact-left">
          <div>
            <div className="section-label">Executive Snapshot</div>
            <h2>Live analytics dashboard</h2>
          </div>
          <div className="runtime-pill">{busy || 'System ready for demo'}</div>
        </div>
        <div className="dashboard-box" dangerouslySetInnerHTML={{ __html: dashboardHtml || '<p class="placeholder">Upload data or run a module to populate the live dashboard.</p>' }} />
      </section>

      <section className="upload-band">
        <div className="upload-card section-card">
          <div className="card-heading"><UploadCloud size={18} /> ERP Demand Data</div>
          <h3>Upload operational demand data</h3>
          <p>Use your ERP-style CSV file to power forecasting and anomaly detection.</p>
          <label className="file-input">
            <input type="file" accept=".csv" onChange={(e) => handleCsvUpload(e.target.files?.[0])} />
            Choose CSV File
          </label>
          <div className="status-box">{csvStatus}</div>
        </div>

        <div className="upload-card section-card">
          <div className="card-heading"><FileText size={18} /> Supplier Contract PDF</div>
          <h3>Upload supplier agreements</h3>
          <p>Index a contract PDF to enable contract report generation and custom document Q&A.</p>
          <label className="file-input">
            <input type="file" accept=".pdf" onChange={(e) => handlePdfUpload(e.target.files?.[0])} />
            Choose PDF File
          </label>
          <div className="status-box">{pdfStatus}</div>
        </div>
      </section>

      <section id="workspace" className="section-card workspace-shell">
        <div className="workspace-header">
          <div>
            <div className="section-label">AI Command Center</div>
            <h2>Run each intelligence layer from one workspace</h2>
          </div>
          <div className="runtime-pill">{busy || 'Awaiting input'}</div>
        </div>

        <div className="tab-row">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              className={activeTab === tab.id ? 'tab active' : 'tab'}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {error ? <div className="error-banner">{error}</div> : null}

        {activeTab === 'forecast' && (
          <div className="tab-panel">
            <div className="panel-actions">
              <button className="primary-btn" onClick={runForecast}><BarChart3 size={18} /> Run 30-Day Forecast</button>
            </div>
            <div className="results-grid wide">
              <div className="visual-card">
                {forecast?.plot_base64 ? (
                  <img src={`data:image/png;base64,${forecast.plot_base64}`} alt="Forecast plot" className="forecast-image" />
                ) : (
                  <div className="empty-state">Forecast chart will appear here after running the model.</div>
                )}
              </div>
              <div className="report-card">
                {forecast?.summary ? <MarkdownCard content={forecast.summary} /> : <div className="empty-state">Forecast insight will appear here.</div>}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'risk' && (
          <div className="tab-panel">
            <div className="panel-actions">
              <button className="primary-btn" onClick={runAnomalies}><BadgeAlert size={18} /> Run Demand Risk Detection</button>
            </div>
            <div className="report-card">
              {anomalies?.summary ? <MarkdownCard content={anomalies.summary} /> : <div className="empty-state">Anomaly report will appear here.</div>}
            </div>
          </div>
        )}

        {activeTab === 'contract' && (
          <div className="tab-panel">
            <div className="panel-actions split">
              <button className="primary-btn" onClick={runContractReport}><ShieldCheck size={18} /> Generate Contract Report</button>
              <div className="ask-box">
                <input value={contractQuestion} onChange={(e) => setContractQuestion(e.target.value)} placeholder="Ask a custom contract question" />
                <button className="secondary-btn" onClick={askContract}>Ask Contract AI</button>
              </div>
            </div>
            <div className="results-grid">
              <div className="report-card">
                {contractReport ? <MarkdownCard content={contractReport} /> : <div className="empty-state">Contract intelligence report will appear here.</div>}
              </div>
              <div className="report-card">
                {contractAnswer ? <MarkdownCard content={contractAnswer} /> : <div className="empty-state">Custom contract answers will appear here.</div>}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'decision' && (
          <div className="tab-panel">
            <div className="panel-actions">
              <button className="primary-btn" onClick={runDecision}><Activity size={18} /> Generate SupplyIQ Decision Report</button>
            </div>
            <div className="report-card">
              {decisionReport ? <MarkdownCard content={decisionReport} /> : <div className="empty-state">Final executive recommendation will appear here.</div>}
            </div>
          </div>
        )}

        {activeTab === 'scenario' && (
          <div className="tab-panel">
            <div className="scenario-controls">
              <select value={scenarioName} onChange={(e) => setScenarioName(e.target.value)}>
                <option>Demand increase (custom %)</option>
                <option>Supplier delay increases by 10 days</option>
                <option>Component price rises by 15%</option>
                <option>Forecast error causes over-ordering</option>
                <option>Custom scenario (write your own below)</option>
              </select>
              <textarea
                value={customScenarioText}
                onChange={(e) => setCustomScenarioText(e.target.value)}
                placeholder="Example: Supplier delay increases by 18 days and demand rises by 40% next month."
              />
              <button className="primary-btn" onClick={runScenario}>Run Scenario Simulation</button>
            </div>
            <div className="report-card">
              {scenarioReport ? <MarkdownCard content={scenarioReport} /> : <div className="empty-state">Scenario simulation output will appear here.</div>}
            </div>
          </div>
        )
}
