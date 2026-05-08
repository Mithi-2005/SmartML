import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  fetchBundles,
  fetchModels,
  downloadBundle,
  deleteBundle,
} from "../lib/api";
import { useSession } from "../context/SessionContext";

const EMPTY_GROUPS = { classification: [], regression: [] };
const UUID_PREFIX = /^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}[_-]*/i;
const MotionHeader = motion.header;
const MotionDiv = motion.div;

const fadeUp = {
  initial: { opacity: 0, y: 24 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, margin: "-50px" },
  transition: { duration: 0.45, ease: [0.16, 1, 0.3, 1] },
};

const Models = () => {
  const { token } = useSession();
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [bundles, setBundles] = useState(EMPTY_GROUPS);
  const [models, setModels] = useState(EMPTY_GROUPS);
  const [activeTab, setActiveTab] = useState("classification");
  const [searchTerm, setSearchTerm] = useState("");
  const [expandedCards, setExpandedCards] = useState({});
  const [selectedPackageInfo, setSelectedPackageInfo] = useState(null);

  useEffect(() => {
    if (!token) return;
    const hydrate = async () => {
      setLoading(true);
      try {
        const [bundleResp, modelResp] = await Promise.all([
          fetchBundles(token),
          fetchModels(token),
        ]);
        setBundles(bundleResp ?? EMPTY_GROUPS);
        setModels(modelResp ?? EMPTY_GROUPS);
      } catch (error) {
        setStatus({ type: "error", message: error.message });
      } finally {
        setLoading(false);
      }
    };
    hydrate();
  }, [token]);

  const refreshData = async () => {
    if (!token) return;
    const [bundleResp, modelResp] = await Promise.all([
      fetchBundles(token),
      fetchModels(token),
    ]);
    setBundles(bundleResp ?? EMPTY_GROUPS);
    setModels(modelResp ?? EMPTY_GROUPS);
  };

  const handleDeleteBundle = async (bundlePath) => {
    if (!token) return;
    try {
      await deleteBundle(token, bundlePath);
      setStatus({ type: "success", message: "Package deleted." });
      await refreshData();
    } catch (error) {
      setStatus({ type: "error", message: error.message });
    }
  };

  useEffect(() => {
    if (status?.type === "success") {
      const timer = setTimeout(() => setStatus(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [status]);

  const formatMetric = (metric) => {
    if (metric === null || metric === undefined) return "N/A";
    const n = Number(metric);
    if (Number.isNaN(n)) return String(metric);
    if (n <= 1 && n >= 0) return `${(n * 100).toFixed(2)}%`;
    return n.toFixed(3);
  };

  const getMetricText = (item) => {
    if (!item) return "N/A";
    if (item.human_metric) return item.human_metric;
    if (item.metric_name || item.metric_value !== undefined) {
      const label = item.metric_name ?? "Metric";
      return `${label}: ${formatMetric(item.metric_value)}`;
    }
    return "N/A";
  };

  const getDisplayName = (bundle) => {
    if (bundle?.display_name) return bundle.display_name;
    const name = bundle?.name || "";
    return name
      .replace(/\.zip$/i, "")
      .replace(UUID_PREFIX, "")
      .replace(/[_-]+/g, " ")
      .trim();
  };

  const normalizeValue = (value = "") =>
    String(value)
      .replace(/\.zip$/i, "")
      .replace(/\.meta\.json$/i, "")
      .replace(/\.pkl$/i, "")
      .replace(UUID_PREFIX, "")
      .replace(/[_-]+/g, " ")
      .replace(/\s+/g, " ")
      .trim()
      .toLowerCase();

  const findRelatedModel = (bundle, group) => {
    const bundleName = normalizeValue(bundle?.name || bundle?.display_name || "");

    return (models[group] || []).find((model) => {
      const candidates = [
        model.model_label,
        model.name,
        model.path,
        model.path?.split(/[\\/]/).pop(),
      ]
        .filter(Boolean)
        .map(normalizeValue);

      return candidates.some(
        (candidate) =>
          candidate === bundleName ||
          candidate.includes(bundleName) ||
          bundleName.includes(candidate)
      );
    });
  };

  const buildLimeLabel = (item) => {
    if (Array.isArray(item)) {
      return `${item[0]}${item[1] !== undefined ? `: ${item[1]}` : ""}`;
    }
    if (typeof item === "object" && item !== null) {
      const feature = item.feature || item.name || "Feature";
      const detail = item.weight ?? item.value ?? item.importance;
      return `${feature}${detail !== undefined ? `: ${detail}` : ""}`;
    }
    return String(item);
  };

  const toggleExpand = (key) =>
    setExpandedCards((prev) => ({ ...prev, [key]: !prev[key] }));

  const getFinalModelUsed = (bundle, model) =>
    model?.model_label || bundle?.model_name || "Not available";

  const openPackageInfo = (bundle, group) => {
    setSelectedPackageInfo({
      bundle,
      model: findRelatedModel(bundle, group),
    });
  };

  if (!token) {
    return (
      <section className="page models">
        <MotionHeader {...fadeUp}>
          <p className="eyebrow">Packages</p>
          <h1>Your Deployment Packages</h1>
          <p className="lead">
            Sign in to view, search, and download your packaged machine learning outputs.
          </p>
        </MotionHeader>
        <MotionDiv className="card" style={{ textAlign: "center", padding: "3rem 2rem" }} {...fadeUp}>
          <div style={{ fontSize: "2.5rem", marginBottom: "1rem" }}>
            <i className="fas fa-lock" />
          </div>
          <h3 style={{ marginBottom: "0.75rem" }}>Authentication Required</h3>
          <p>Please sign in to access your deployment packages.</p>
        </MotionDiv>
      </section>
    );
  }

  const filteredBundles = (bundles[activeTab] || []).filter((bundle) =>
    getDisplayName(bundle).toLowerCase().includes(searchTerm.toLowerCase().trim())
  );

  const renderBundleList = (items = [], key, group) => {
    if (!items.length) {
      return (
        <p className="muted" style={{ padding: "2rem", textAlign: "center" }}>
          <i className="fas fa-info-circle" /> No matching packages found.
        </p>
      );
    }

    const isExpanded = expandedCards[key];
    const needsScroll = items.length > 3;

    return (
      <>
        <div className={`card-scroll ${isExpanded ? "expanded" : ""}`}>
          <ul className="model-list">
            {items.map((bundle) => {
              const displayName = getDisplayName(bundle);
              const relatedModel = findRelatedModel(bundle, group);

              return (
                <li key={bundle.path} className="model-row">
                  <div className="model-meta">
                    <strong>{displayName}</strong>
                    <small>Final model used: {getFinalModelUsed(bundle, relatedModel)}</small>
                    <small>
                      Updated {new Date(bundle.modified_ts * 1000).toLocaleString()} · {(bundle.size_bytes / (1024 * 1024)).toFixed(2)} MB
                    </small>
                  </div>
                  <div className="model-actions">
                    <button
                      type="button"
                      className="icon-btn"
                      aria-label={`View info for ${displayName}`}
                      onClick={(event) => {
                        event.stopPropagation();
                        openPackageInfo(bundle, group);
                      }}
                    >
                      <i className="fas fa-circle-info" />
                    </button>
                    <button
                      type="button"
                      className="icon-btn"
                      aria-label={`Download package ${displayName}`}
                      onClick={() =>
                        downloadBundle(token, bundle.download_url, `${displayName}.zip`)
                      }
                    >
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 4v10m0 0 4-4m-4 4-4-4m-4 9h16" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                    </button>
                    <button
                      type="button"
                      className="icon-btn danger"
                      aria-label={`Delete package ${displayName}`}
                      onClick={() => handleDeleteBundle(bundle.path)}
                    >
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M18 6l-1 14H7L6 6m3 0V4h6v2m-9 0h12" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                    </button>
                  </div>
                </li>
              );
            })}
          </ul>
        </div>
        {needsScroll && (
          <button
            type="button"
            className="view-more-btn"
            onClick={() => toggleExpand(key)}
          >
            <i className={`fas fa-chevron-${isExpanded ? "up" : "down"}`} />
            {isExpanded ? "Show Less" : `View All (${items.length})`}
          </button>
        )}
      </>
    );
  };

  return (
    <section className="page models">
      <MotionHeader {...fadeUp}>
        <p className="eyebrow">Packages</p>
        <h1>Your Deployment Packages</h1>
        <p className="lead">
          Browse packages only, switch between classification and regression, search quickly, and open full LIME details from each package card.
        </p>
      </MotionHeader>

      {status && (
        <div className={`alert ${status.type}`}>
          <i
            className={`fas fa-${
              status.type === "success" ? "check-circle" : "triangle-exclamation"
            }`}
          />
          {status.message}
        </div>
      )}

      {loading && (
        <div style={{ textAlign: "center" }}>
          <span className="badge">
            <i className="fas fa-spinner fa-spin" /> Loading packages...
          </span>
        </div>
      )}

      <MotionDiv {...fadeUp}>
        <div className="models-tabs">
          <button
            type="button"
            className={`models-tab-btn ${activeTab === "classification" ? "active" : ""}`}
            onClick={() => setActiveTab("classification")}
          >
            <i className="fas fa-bullseye" /> Classification
          </button>
          <button
            type="button"
            className={`models-tab-btn ${activeTab === "regression" ? "active" : ""}`}
            onClick={() => setActiveTab("regression")}
          >
            <i className="fas fa-chart-line" /> Regression
          </button>
        </div>
      </MotionDiv>

      <MotionDiv {...fadeUp}>
        <div className="packages-header-row" style={{ marginBottom: "1.5rem" }}>
          <div>
            <h2 style={{ fontSize: "1.25rem", marginBottom: "0.5rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <i
                className={`fas fa-${activeTab === "classification" ? "bullseye" : "chart-line"}`}
                style={{ color: activeTab === "classification" ? "var(--amber-400)" : "var(--teal-400)" }}
              />
              {activeTab === "classification" ? "Classification Packages" : "Regression Packages"}
            </h2>
            <p style={{ color: "var(--text-secondary)", fontSize: "0.9375rem" }}>
              Open the info button on any package to view model accuracy, reason, and all available LIME information.
            </p>
          </div>
          <label className="packages-search">
            <i className="fas fa-search" />
            <input
              type="search"
              placeholder={`Search ${activeTab} packages`}
              value={searchTerm}
              onChange={(event) => setSearchTerm(event.target.value)}
            />
          </label>
        </div>
        <div className="card">
          {renderBundleList(filteredBundles, `bundles-${activeTab}`, activeTab)}
        </div>
      </MotionDiv>

      {selectedPackageInfo && (
        <div
          className="reason-overlay"
          role="dialog"
          aria-modal="true"
          onClick={() => setSelectedPackageInfo(null)}
        >
          <div className="card reason-card" onClick={(event) => event.stopPropagation()}>
            <div className="reason-card-head">
              <div>
                <p className="eyebrow">Package Info</p>
                <h3>{getDisplayName(selectedPackageInfo.bundle)}</h3>
              </div>
              <button
                type="button"
                className="icon-btn"
                aria-label="Close package info"
                onClick={() => setSelectedPackageInfo(null)}
              >
                <i className="fas fa-xmark" />
              </button>
            </div>
            <div className="reason-card-body">
              <p>
                <strong>Final model used:</strong> {getFinalModelUsed(selectedPackageInfo.bundle, selectedPackageInfo.model)}
              </p>
              <p>
                <strong>Package size:</strong> {(selectedPackageInfo.bundle.size_bytes / (1024 * 1024)).toFixed(2)} MB
              </p>
              <p>
                <strong>Last updated:</strong> {new Date(selectedPackageInfo.bundle.modified_ts * 1000).toLocaleString()}
              </p>
              <p>
                <strong>Accuracy:</strong> {getMetricText(selectedPackageInfo.model || selectedPackageInfo.bundle)}
              </p>
              <p>
                <strong>Model reason:</strong> {selectedPackageInfo.model?.model_reason || selectedPackageInfo.bundle.model_reason || "Not available"}
              </p>
              <div style={{ marginTop: "1rem" }}>
                <small><strong>All LIME Information:</strong></small>
                {(selectedPackageInfo.model?.explanations || selectedPackageInfo.bundle.explanations)?.length ? (
                  <div className="explanations" style={{ marginTop: "0.75rem" }}>
                    {(selectedPackageInfo.model?.explanations || selectedPackageInfo.bundle.explanations).map((item, index) => (
                      <span
                        key={`${selectedPackageInfo.bundle.path}-${index}`}
                        className="pill"
                      >
                        {buildLimeLabel(item)}
                      </span>
                    ))}
                  </div>
                ) : (
                  <p className="muted" style={{ marginTop: "0.5rem" }}>
                    LIME information not available.
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </section>
  );
};

export default Models;
