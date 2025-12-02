import { useEffect, useMemo, useState } from 'react'
import { NavLink } from 'react-router-dom'
import {
  previewColumns,
  uploadDataset,
  fetchDatasets,
  downloadDataset,
  deleteDataset,
  fetchTrainingStatus,
  fetchActiveTrainingRuns,
} from '../lib/api'
import { useSession } from '../context/SessionContext'

const initialPayload = {
  task_type: 'classification',
  target_col: '',
  tuning: 'false',
}

const Workspace = () => {
  const { token, profile } = useSession()
  const [datasetFile, setDatasetFile] = useState(null)
  const [form, setForm] = useState(initialPayload)
  const [columns, setColumns] = useState([])
  const [status, setStatus] = useState(null)
  const [catalogue, setCatalogue] = useState({ classification: [], regression: [] })
  const [loadingCatalogue, setLoadingCatalogue] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [columnLoading, setColumnLoading] = useState(false)
  const [activeRun, setActiveRun] = useState(null)
  const [statusFeed, setStatusFeed] = useState([])
  const [statusError, setStatusError] = useState(null)
  const [currentStatus, setCurrentStatus] = useState(null)
  const [timeElapsed, setTimeElapsed] = useState(0)

  const updateForm = (field, value) => setForm((prev) => ({ ...prev, [field]: value }))

  const formatElapsedTime = (seconds) => {
    if (seconds < 60) return `${seconds}s`
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    if (mins < 60) return `${mins}m ${secs}s`
    const hours = Math.floor(mins / 60)
    const remainingMins = mins % 60
    return `${hours}h ${remainingMins}m`
  }

  const loadCatalogue = () => {
    if (!token) return
    setLoadingCatalogue(true)
    fetchDatasets(token)
      .then(setCatalogue)
      .catch(() => setCatalogue({ classification: [], regression: [] }))
      .finally(() => setLoadingCatalogue(false))
  }

  useEffect(() => {
    loadCatalogue()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token])

  // Restore active runs on mount
  useEffect(() => {
    if (!token) return

    const restoreActiveRuns = async () => {
      try {
        // Fetch active runs from backend
        const response = await fetchActiveTrainingRuns(token)
        const backendRuns = response?.active_runs || []

        // Also check localStorage for any runs
        const storedRun = localStorage.getItem('smartml_active_run')
        
        if (backendRuns.length > 0) {
          // Use the first active run from backend
          const run = backendRuns[0]
          setActiveRun({
            datasetId: run.dataset_id,
            name: run.name,
          })
        } else if (storedRun) {
          // Fallback to localStorage if backend has no active runs
          try {
            const parsed = JSON.parse(storedRun)
            setActiveRun(parsed)
          } catch (e) {
            localStorage.removeItem('smartml_active_run')
          }
        }
      } catch (error) {
        console.error('Failed to restore active runs:', error)
      }
    }

    restoreActiveRuns()
  }, [token])


  // Auto-dismiss success alerts after 5 seconds
  useEffect(() => {
    if (status?.type === 'success') {
      const timer = setTimeout(() => {
        setStatus(null)
      }, 5000)
      return () => clearTimeout(timer)
    }
  }, [status])

  const hydrateColumns = async (file) => {
    if (!token || !file) return
    setColumnLoading(true)
    try {
      const response = await previewColumns(token, file)
      setColumns(response)
      setForm((prev) => ({
        ...prev,
        target_col: response?.[0] ?? '',
      }))
    } catch (error) {
      setColumns([])
      setForm((prev) => ({ ...prev, target_col: '' }))
      setStatus({ type: 'error', message: error.message })
    } finally {
      setColumnLoading(false)
    }
  }

  const handleFileChange = async (event) => {
    const file = event.target.files?.[0]
    setDatasetFile(file || null)
    setColumns([])
    setForm((prev) => ({ ...prev, target_col: '' }))
    if (file) {
      await hydrateColumns(file)
    }
  }

  const handleUpload = async (event) => {
    event.preventDefault()
    if (!datasetFile) {
      setStatus({ type: 'error', message: 'Attach a dataset before submitting.' })
      return
    }
    setSubmitting(true)
    setStatus(null)
    try {
      const response = await uploadDataset(token, { ...form, file: datasetFile })
      
      // Set activeRun immediately to start polling
      if (response?.status?.dataset_id) {
        const newRun = {
          datasetId: response.status.dataset_id,
          name: response.dataset?.original_name ?? 'dataset',
        }
        setActiveRun(newRun)
        // Persist to localStorage
        localStorage.setItem('smartml_active_run', JSON.stringify(newRun))
        setStatusFeed([])
        setStatusError(null)
        setSubmitting(false) // Reset button immediately
      }
      
      setStatus({
        type: 'success',
        message: 'Dataset uploaded successfully. Training started.',
      })
      setDatasetFile(null)
      setColumns([])
      setForm(initialPayload)
      loadCatalogue()
    } catch (error) {
      setStatus({ type: 'error', message: error.message })
      setSubmitting(false)
    }
  }

  const handleDeleteDataset = async (filePath) => {
    if (!token) return
    try {
      await deleteDataset(token, filePath)
      setStatus({ type: 'success', message: 'Dataset deleted.' })
      loadCatalogue()
    } catch (error) {
      setStatus({ type: 'error', message: error.message })
    }
  }

  useEffect(() => {
    if (!token || !activeRun?.datasetId) return
    let cancelled = false

    const statusPollRef = { current: null }

    const poll = async () => {
      try {
        const res = await fetchTrainingStatus(token, activeRun.datasetId)
        if (!cancelled) {
          setStatusFeed(res?.history ?? [])
          setCurrentStatus(res?.current ?? null)
          setStatusError(null)
          if (res?.current?.state === 'completed' || res?.current?.state === 'error') {
            setActiveRun((prev) => (prev ? { ...prev, terminalState: res.current.state } : prev))
            // Clean up localStorage when training completes
            localStorage.removeItem('smartml_active_run')
            return true
          }
        }
      } catch (error) {
        if (!cancelled) {
          setStatusError(error.message)
        }
      }
      return false
    }

    const kickoff = async () => {
      const done = await poll()
      if (done) return
      const interval = setInterval(async () => {
        const finished = await poll()
        if (finished) {
          clearInterval(interval)
        }
      }, 4000)
      statusPollRef.current = interval
    }

    kickoff()

    return () => {
      cancelled = true
      if (statusPollRef.current) {
        clearInterval(statusPollRef.current)
      }
    }
  }, [token, activeRun?.datasetId])

  // Track elapsed time in current stage
  useEffect(() => {
    if (!currentStatus?.timestamp) {
      setTimeElapsed(0)
      return
    }

    // Don't update timer if in terminal state
    if (currentStatus.state === 'completed' || currentStatus.state === 'error') {
      const start = new Date(currentStatus.timestamp)
      const now = new Date()
      const diff = Math.floor((now - start) / 1000)
      setTimeElapsed(diff)
      return
    }

    const updateElapsed = () => {
      const start = new Date(currentStatus.timestamp)
      const now = new Date()
      const diff = Math.floor((now - start) / 1000)
      setTimeElapsed(diff)
    }

    updateElapsed()
    const interval = setInterval(updateElapsed, 1000)

    return () => clearInterval(interval)
  }, [currentStatus?.timestamp, currentStatus?.state])

  const formattedStatus = useMemo(
    () =>
      statusFeed
        .slice()
        .reverse()
        .map((event, idx) => ({
          ...event,
          id: `${event.phase}-${event.timestamp}-${idx}`,
          time: new Date(event.timestamp || Date.now()).toLocaleTimeString(),
        })),
    [statusFeed]
  )

  if (!token) {
    return (
      <section className="page workspace">
        <header>
          <h1>Workspace</h1>
          <p className="lead">Authenticate to stream datasets to FastAPI.</p>
        </header>
        <div className="card muted">
          <p>Bring your JWT token by logging in first. Don’t have one yet? Create an account below.</p>
          <NavLink className="btn primary" to="/auth">
            Login / Sign up
          </NavLink>
        </div>
      </section>
    )
  }
  return (
    <section className="page workspace">
      <header>
        <p className="eyebrow">Pipelines</p>
        <h1>Upload datasets & orchestrate training</h1>
        <p className="lead">
          Hi {profile?.fname ?? 'there'}, ship CSV datasets straight to <code>/users/send_dataset</code>. We automatically
          persist the file under your user namespace before triggering the AutoML user worker.
        </p>
      </header>

      {status && (
        <div className={`alert ${status.type}`}>
          <span>{status.message}</span>
        </div>
      )}

      {activeRun && (
        <div className="card status-feed">
          <div className="form-heading">
            <h3>Live training status</h3>
            <p>Tracking: {activeRun.name}</p>
            <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
              {activeRun.terminalState && (
                <span className={`badge ${activeRun.terminalState === 'completed' ? 'success' : 'error'}`}>
                  {activeRun.terminalState === 'completed' ? 'Finished' : 'Failed'}
                </span>
              )}
              {activeRun.terminalState === 'completed' && (
                <NavLink to="/models" className="btn primary" style={{ fontSize: '0.875rem', padding: '0.375rem 0.75rem' }}>
                  Go to Models →
                </NavLink>
              )}
            </div>
          </div>
          {statusError && <p className="muted">Status temporarily unavailable: {statusError}</p>}
          
          {currentStatus && (
            <div className="current-stage">
              <div className="current-stage-header">
                <span className={`current-stage-badge ${currentStatus.state}`}>
                  {currentStatus.state === 'running' && (
                    <svg className="stage-spinner" width="16" height="16" viewBox="0 0 24 24" fill="none">
                      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeDasharray="31.4 31.4" />
                    </svg>
                  )}
                  {currentStatus.state === 'completed' && (
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                      <path d="M20 6L9 17l-5-5" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  )}
                  {currentStatus.state === 'error' && (
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                      <path d="M6 18L18 6M6 6l12 12" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
                    </svg>
                  )}
                  <span className="phase-name">{currentStatus.phase}</span>
                </span>
                <span className="current-stage-timer">⏱ {formatElapsedTime(timeElapsed)}</span>
              </div>
              <p className="current-stage-message">{currentStatus.message}</p>
            </div>
          )}

          {formattedStatus.length ? (
            <>
              <h4 className="history-heading">History</h4>
              <ol className="status-timeline">
                {formattedStatus.map((event) => (
                  <li key={event.id}>
                    <div className="status-header">
                      <strong>{event.phase}</strong>
                      <span>{event.time}</span>
                    </div>
                    <p>{event.message}</p>
                  </li>
                ))}
              </ol>
            </>
          ) : !currentStatus ? (
            <p className="muted">Waiting for the first update…</p>
          ) : null}
        </div>
      )}

      <form onSubmit={handleUpload} className="card form">
        <div className="form-heading">
          <h3>Dataset intake</h3>
          <p>We store the artifact and immediately kick off preprocessing, meta-feature extraction and model search.</p>
        </div>
        <label className="file-input">
          <span>Dataset (CSV)</span>
          <input type="file" accept=".csv" onChange={handleFileChange} />
        </label>
        {columnLoading && <span className="badge">Detecting columns…</span>}

        <div className="form-grid">
          <label>
            Task type
            <select value={form.task_type} onChange={(e) => updateForm('task_type', e.target.value)}>
              <option value="classification">Classification</option>
              <option value="regression">Regression</option>
            </select>
          </label>
          <label>
            Target column
            <select
              value={form.target_col}
              onChange={(e) => updateForm('target_col', e.target.value)}
              disabled={!columns.length}
            >
              <option value="">{columns.length ? 'Select a target column' : 'Upload a dataset first'}</option>
              {columns.map((column) => (
                <option key={column} value={column}>
                  {column}
                </option>
              ))}
            </select>
          </label>
          <label>
            Hyperparameter tuning
            <select value={form.tuning} onChange={(e) => updateForm('tuning', e.target.value)}>
              <option value="true">Enabled</option>
              <option value="false">Disabled</option>
            </select>
          </label>
        </div>

        <div className="actions">
          <button type="submit" className="btn primary" disabled={submitting || !form.target_col}>
            {submitting ? 'Uploading…' : 'Send dataset'}
          </button>
        </div>
      </form>

      <section className="catalogue">
        <header>
          <h2>Your dataset catalogue</h2>
          {loadingCatalogue && <span className="badge">Refreshing…</span>}
        </header>
        <div className="grid two">
          {['classification', 'regression'].map((type) => (
            <div key={type} className="card">
              <h3>{type}</h3>
              {catalogue[type]?.length ? (
                <ul>
                  {catalogue[type].map((file) => (
                    <li key={file.download_url} className="catalogue-item">
                      <span>{file.name}</span>
                      <div className="catalogue-actions">
                        <button
                          type="button"
                          className="icon-btn"
                          onClick={() => downloadDataset(token, file.download_url, file.name)}
                          aria-label={`Download ${file.name}`}
                        >
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path
                              d="M12 4v10m0 0 4-4m-4 4-4-4m-4 9h16"
                              stroke="currentColor"
                              strokeWidth="1.6"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                            />
                          </svg>
                        </button>
                        <button
                          type="button"
                          className="icon-btn danger"
                          onClick={() => handleDeleteDataset(file.path)}
                          aria-label={`Delete ${file.name}`}
                        >
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path
                              d="M18 6l-1 14H7L6 6m3 0V4h6v2m-9 0h12"
                              stroke="currentColor"
                              strokeWidth="1.6"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                            />
                          </svg>
                        </button>
                      </div>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="muted">No datasets yet.</p>
              )}
            </div>
          ))}
        </div>
      </section>
    </section>
  )
}

export default Workspace

