import { useEffect, useMemo, useState } from "react";
import { NavLink } from "react-router-dom";
import { motion } from "framer-motion";
import {
  previewColumns,
  uploadDataset,
  fetchDatasets,
  downloadDataset,
  deleteDataset,
  fetchTrainingStatus,
  fetchActiveTrainingRuns,
} from "../lib/api";
import { useSession } from "../context/SessionContext";

const initialPayload = {
  task_type: "classification",
  target_col: "",
  tuning: "false",
};

const fadeUp = {
  initial: { opacity: 0, y: 24 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, margin: "-50px" },
  transition: { duration: 0.45, ease: [0.16, 1, 0.3, 1] },
};

const Workspace = () => {
  const { token, profile } = useSession();
  const [datasetFile, setDatasetFile] = useState(null);
  const [form, setForm] = useState(initialPayload);
  const [columns, setColumns] = useState([]);
  const [status, setStatus] = useState(null);
  const [catalogue, setCatalogue] = useState({
    classification: [],
    regression: [],
  });
  const [loadingCatalogue, setLoadingCatalogue] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [columnLoading, setColumnLoading] = useState(false);
  const [activeRun, setActiveRun] = useState(null);
  const [statusFeed, setStatusFeed] = useState([]);
  const [statusError, setStatusError] = useState(null);
  const [currentStatus, setCurrentStatus] = useState(null);
  const [timeElapsed, setTimeElapsed] = useState(0);
  const [showTracking, setShowTracking] = useState(false);

  const updateForm = (field, value) =>
    setForm((prev) => ({ ...prev, [field]: value }));

  const formatElapsedTime = (seconds) => {
    if (seconds < 60) return `${seconds}s`;
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    if (mins < 60) return `${mins}m ${secs}s`;
    const hours = Math.floor(mins / 60);
    const remainingMins = mins % 60;
    return `${hours}h ${remainingMins}m`;
  };

  const loadCatalogue = () => {
    if (!token) return;
    setLoadingCatalogue(true);
    fetchDatasets(token)
      .then(setCatalogue)
      .catch(() => setCatalogue({ classification: [], regression: [] }))
      .finally(() => setLoadingCatalogue(false));
  };

  useEffect(() => {
    loadCatalogue();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  /* Restore active runs */
  useEffect(() => {
    if (!token) return;
    const restoreActiveRuns = async () => {
      try {
        const response = await fetchActiveTrainingRuns(token);
        const backendRuns = response?.active_runs || [];
        const storedRun = localStorage.getItem("smartml_active_run");

        if (backendRuns.length > 0) {
          const run = backendRuns[0];
          setActiveRun({ datasetId: run.dataset_id, name: run.name });
        } else if (storedRun) {
          try {
            setActiveRun(JSON.parse(storedRun));
          } catch {
            localStorage.removeItem("smartml_active_run");
          }
        }
      } catch (error) {
        console.error("Failed to restore active runs:", error);
      }
    };
    restoreActiveRuns();
  }, [token]);

  /* Auto dismiss success alerts */
  useEffect(() => {
    if (status?.type === "success") {
      const timer = setTimeout(() => setStatus(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [status]);

  const hydrateColumns = async (file) => {
    if (!token || !file) return;
    setColumnLoading(true);
    try {
      const response = await previewColumns(token, file);
      setColumns(response);
      setForm((prev) => ({ ...prev, target_col: response?.[0] ?? "" }));
    } catch (error) {
      setColumns([]);
      setForm((prev) => ({ ...prev, target_col: "" }));
      setStatus({ type: "error", message: error.message });
    } finally {
      setColumnLoading(false);
    }
  };

  const handleFileChange = async (event) => {
    const file = event.target.files?.[0];
    setDatasetFile(file || null);
    setColumns([]);
    setForm((prev) => ({ ...prev, target_col: "" }));
    if (file) await hydrateColumns(file);
  };

  const handleUpload = async (event) => {
    event.preventDefault();
    if (!datasetFile) {
      setStatus({ type: "error", message: "Attach a dataset before submitting." });
      return;
    }
    setSubmitting(true);
    setStatus(null);
    try {
      const response = await uploadDataset(token, { ...form, file: datasetFile });
      if (response?.status?.dataset_id) {
        const newRun = {
          datasetId: response.status.dataset_id,
          name: response.dataset?.original_name ?? "dataset",
        };
        setActiveRun(newRun);
        localStorage.setItem("smartml_active_run", JSON.stringify(newRun));
        setStatusFeed([]);
        setStatusError(null);
        setSubmitting(false);
      }
      setStatus({ type: "success", message: "Dataset uploaded successfully. Training started." });
      setDatasetFile(null);
      setColumns([]);
      setForm(initialPayload);
      loadCatalogue();
    } catch (error) {
      setStatus({ type: "error", message: error.message });
      setSubmitting(false);
    }
  };

  const handleDeleteDataset = async (filePath) => {
    if (!token) return;
    try {
      await deleteDataset(token, filePath);
      setStatus({ type: "success", message: "Dataset deleted." });
      loadCatalogue();
    } catch (error) {
      setStatus({ type: "error", message: error.message });
    }
  };

  /* Polling for training status */
  useEffect(() => {
    if (!token || !activeRun?.datasetId) return;
    let cancelled = false;
    const statusPollRef = { current: null };

    const poll = async () => {
      try {
        const res = await fetchTrainingStatus(token, activeRun.datasetId);
        if (!cancelled) {
          setStatusFeed(res?.history ?? []);
          setCurrentStatus(res?.current ?? null);
          setStatusError(null);
          if (res?.current?.state === "completed" || res?.current?.state === "error") {
            setActiveRun((prev) =>
              prev ? { ...prev, terminalState: res.current.state } : prev
            );
            localStorage.removeItem("smartml_active_run");
            return true;
          }
        }
      } catch (error) {
        if (!cancelled) setStatusError(error.message);
      }
      return false;
    };

    const kickoff = async () => {
      const done = await poll();
      if (done) return;
      const interval = setInterval(async () => {
        const finished = await poll();
        if (finished) clearInterval(interval);
      }, 4000);
      statusPollRef.current = interval;
    };

    kickoff();
    return () => {
      cancelled = true;
      if (statusPollRef.current) clearInterval(statusPollRef.current);
    };
  }, [token, activeRun?.datasetId]);

  /* Elapsed time */
  useEffect(() => {
    if (!currentStatus?.timestamp) {
      setTimeElapsed(0);
      return;
    }
    if (currentStatus.state === "completed" || currentStatus.state === "error") {
      const diff = Math.floor((new Date() - new Date(currentStatus.timestamp)) / 1000);
      setTimeElapsed(diff);
      return;
    }
    const updateElapsed = () => {
      const diff = Math.floor((new Date() - new Date(currentStatus.timestamp)) / 1000);
      setTimeElapsed(diff);
    };
    updateElapsed();
    const interval = setInterval(updateElapsed, 1000);
    return () => clearInterval(interval);
  }, [currentStatus?.timestamp, currentStatus?.state]);

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
  );

  /* ---- Unauthenticated ---- */
  if (!token) {
    return (
      <section className="page workspace">
        <motion.header {...fadeUp}>
          <p className="eyebrow">Your Workspace</p>
          <h1>Ready to Build Your First Model?</h1>
          <p className="lead">
            Sign in to unlock automated machine learning. Upload datasets, train
            models, and download production-ready AI in minutes.
          </p>
        </motion.header>
        <motion.div className="card" style={{ textAlign: "center", padding: "3rem 2rem" }} {...fadeUp}>
          <div style={{ fontSize: "2.5rem", marginBottom: "1rem" }}>🔐</div>
          <h3 style={{ marginBottom: "0.75rem" }}>Authentication Required</h3>
          <p style={{ marginBottom: "1.5rem" }}>
            Please sign in to access your personal workspace and start building
            models.
          </p>
          <NavLink className="btn primary" to="/auth">
            Sign In to Continue
          </NavLink>
        </motion.div>
      </section>
    );
  }

  /* ---- Authenticated ---- */
  return (
    <section className="page workspace">
      <motion.header {...fadeUp}>
        <p className="eyebrow">Your Workspace</p>
        <h1>Build &amp; Train Your Models</h1>
        <p className="lead">
          Welcome back, {profile?.fname ?? "there"}! Upload your dataset and
          watch as our AI automatically processes, trains, and delivers a
          production-ready model.
        </p>
      </motion.header>

      {status && (
        <div className={`alert ${status.type}`}>
          <i
            className={`fas fa-${
              status.type === "success" ? "check-circle" : "triangle-exclamation"
            }`}
          />
          <span>{status.message}</span>
        </div>
      )}

      {/* Training status */}
      {activeRun && (
        <motion.div className="training-status-container" {...fadeUp}>
          <div className="training-status-header">
            <div className="status-info">
              <h3>
                <i className="fas fa-robot" /> Training: {activeRun.name}
              </h3>
              <div className="status-badges">
                {activeRun.terminalState ? (
                  <span
                    className={`badge ${
                      activeRun.terminalState === "completed" ? "success" : "error"
                    }`}
                  >
                    <i
                      className={`fas fa-${
                        activeRun.terminalState === "completed" ? "check" : "times"
                      }`}
                    />
                    {activeRun.terminalState === "completed" ? "Completed" : "Failed"}
                  </span>
                ) : (
                  currentStatus && (
                    <span className="badge running">
                      <i className="fas fa-spinner fa-spin" />
                      {currentStatus.phase}
                    </span>
                  )
                )}
                {currentStatus && !activeRun.terminalState && (
                  <span className="badge info">
                    <i className="far fa-clock" />
                    {formatElapsedTime(timeElapsed)}
                  </span>
                )}
              </div>
            </div>
            <div className="status-actions">
              {activeRun.terminalState === "completed" && (
                <NavLink to="/models" className="btn primary btn-sm">
                  <i className="fas fa-eye" /> View Model
                </NavLink>
              )}
              <button
                type="button"
                className="btn ghost btn-sm"
                onClick={() => setShowTracking(!showTracking)}
              >
                <i className={`fas fa-chevron-${showTracking ? "up" : "down"}`} />
                {showTracking ? "Hide" : "Show"} Tracking
              </button>
            </div>
          </div>

          {statusError && (
            <p className="muted" style={{ padding: "1rem", margin: 0 }}>
              <i className="fas fa-triangle-exclamation" /> Status temporarily
              unavailable: {statusError}
            </p>
          )}

          {showTracking && (
            <div className="tracking-details">
              {currentStatus && (
                <div className="current-stage-detail">
                  <p className="current-message">
                    <i className="fas fa-info-circle" />
                    {currentStatus.message}
                  </p>
                </div>
              )}
              {formattedStatus.length ? (
                <div className="tracking-timeline">
                  <h4 className="timeline-heading">
                    <i className="fas fa-list-check" /> Progress Timeline
                  </h4>
                  <div className="timeline-wrapper">
                    {formattedStatus.map((event, index) => {
                      const isActive =
                        index === 0 && currentStatus?.phase === event.phase;
                      const isCompleted =
                        event.state === "completed" || (!isActive && index !== 0);
                      const isError = event.state === "error";
                      return (
                        <div
                          key={event.id}
                          className={`timeline-item ${isActive ? "active" : ""} ${
                            isCompleted ? "completed" : ""
                          } ${isError ? "error" : ""}`}
                        >
                          <div className="timeline-marker">
                            <i
                              className={`fas fa-${
                                isError
                                  ? "times"
                                  : isCompleted
                                  ? "check"
                                  : "circle"
                              }`}
                            />
                          </div>
                          <div className="timeline-content">
                            <div className="timeline-header">
                              <span className="timeline-phase">{event.phase}</span>
                              <span className="timeline-time">{event.time}</span>
                            </div>
                            <p className="timeline-message">{event.message}</p>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ) : !currentStatus ? (
                <p className="muted" style={{ padding: "1rem", textAlign: "center" }}>
                  <i className="fas fa-hourglass-start" /> Waiting for training to
                  begin…
                </p>
              ) : null}
            </div>
          )}
        </motion.div>
      )}

      {/* Upload form */}
      <motion.form onSubmit={handleUpload} className="card form-card" {...fadeUp}>
        <div className="form-heading">
          <h3>
            <i className="fas fa-cloud-arrow-up" /> Upload Your Dataset
          </h3>
          <p>
            Start by uploading a CSV file. We'll analyze it, train the optimal
            model, and have it ready for download in minutes.
          </p>
        </div>

        <label className="file-input">
          <span>
            <i className="fas fa-file-csv" />{" "}
            {datasetFile ? datasetFile.name : "Choose CSV File"}
          </span>
          <input type="file" accept=".csv" onChange={handleFileChange} />
        </label>

        {columnLoading && (
          <span className="badge">
            <i className="fas fa-spinner fa-spin" /> Analyzing columns…
          </span>
        )}

        <div className="form-grid">
          <label>
            What type of prediction?
            <select
              value={form.task_type}
              onChange={(e) => updateForm("task_type", e.target.value)}
            >
              <option value="classification">Classification (Categories)</option>
              <option value="regression">Regression (Numbers)</option>
            </select>
          </label>
          <label>
            Target column (what to predict)
            <select
              value={form.target_col}
              onChange={(e) => updateForm("target_col", e.target.value)}
              disabled={!columns.length}
            >
              <option value="">
                {columns.length ? "Choose target column" : "Upload file first"}
              </option>
              {columns.map((col) => (
                <option key={col} value={col}>
                  {col}
                </option>
              ))}
            </select>
          </label>
          <label>
            Optimize performance?
            <select
              value={form.tuning}
              onChange={(e) => updateForm("tuning", e.target.value)}
            >
              <option value="true">Yes (Recommended)</option>
              <option value="false">No (Faster)</option>
            </select>
          </label>
        </div>

        <div className="actions">
          <button
            type="submit"
            className="btn primary"
            disabled={submitting || !form.target_col}
          >
            {submitting ? (
              <>
                <i className="fas fa-spinner fa-spin" /> Uploading…
              </>
            ) : (
              <>
                <i className="fas fa-rocket" /> Start Training
              </>
            )}
          </button>
        </div>
      </motion.form>

      {/* Dataset catalogue */}
      <motion.section className="catalogue" {...fadeUp}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "1.5rem" }}>
          <h2>
            <i className="fas fa-database" style={{ color: "var(--amber-400)", marginRight: "0.5rem" }} />
            Your Datasets
          </h2>
          {loadingCatalogue && (
            <span className="badge">
              <i className="fas fa-spinner fa-spin" /> Refreshing…
            </span>
          )}
        </div>

        <div className="grid-2">
          {["classification", "regression"].map((type) => (
            <div key={type} className="card">
              <h3 style={{ textTransform: "capitalize", marginBottom: "1rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <i
                  className={`fas fa-${
                    type === "classification" ? "tag" : "chart-line"
                  }`}
                  style={{ color: type === "classification" ? "var(--amber-400)" : "var(--teal-400)" }}
                />
                {type}
              </h3>
              {catalogue[type]?.length ? (
                <div className="card-scroll">
                  <ul style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                    {catalogue[type].map((file) => (
                      <li key={file.download_url} className="catalogue-item">
                        <span>{file.name}</span>
                        <div className="catalogue-actions">
                          <button
                            type="button"
                            className="icon-btn"
                            onClick={() =>
                              downloadDataset(token, file.download_url, file.name)
                            }
                            aria-label={`Download ${file.name}`}
                            title="Download dataset"
                          >
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                              <path d="M12 4v10m0 0 4-4m-4 4-4-4m-4 9h16" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                            </svg>
                          </button>
                          <button
                            type="button"
                            className="icon-btn danger"
                            onClick={() => handleDeleteDataset(file.path || file.download_url)}
                            aria-label={`Delete ${file.name}`}
                            title="Delete dataset"
                          >
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                              <path d="M18 6l-1 14H7L6 6m3 0V4h6v2m-9 0h12" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                            </svg>
                          </button>
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : (
                <p className="muted" style={{ padding: "2rem", textAlign: "center" }}>
                  No {type} datasets yet. Upload your first one above!
                </p>
              )}
            </div>
          ))}
        </div>
      </motion.section>
    </section>
  );
};

export default Workspace;
