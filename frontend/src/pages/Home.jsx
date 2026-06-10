import { motion } from "framer-motion";
import { NavLink } from "react-router-dom";
import { useSession } from "../context/SessionContext";

/* ---- Data ---- */
const stats = [
  { label: "Models Trained", value: "500+", icon: "fa-robot" },
  { label: "Data Processed", value: "2.4M+", icon: "fa-chart-bar" },
  { label: "Average Time", value: "<7 min", icon: "fa-bolt" },
  { label: "Accuracy Rate", value: "75%", icon: "fa-bullseye" },
];

const features = [
  {
    title: "Upload & Analyze",
    body: "Simply upload your dataset and let our AI do the heavy lifting. No coding required — just pure insights.",
    icon: "fa-cloud-arrow-up",
  },
  {
    title: "Smart Model Selection",
    body: "Our meta-learning engine automatically selects the best ML algorithm for your specific data patterns.",
    icon: "fa-brain",
  },
  {
    title: "Instant Training",
    body: "Watch your models train in real-time with live progress updates. Production-ready models in minutes.",
    icon: "fa-bolt",
  },
  {
    title: "Download & Deploy",
    body: "Get trained models with all preprocessing pipelines packaged and ready for deployment.",
    icon: "fa-box-open",
  },
  {
    title: "Visual Insights",
    body: "Understand model decisions with explainable AI features and feature importance visualizations.",
    icon: "fa-chart-line",
  },
  {
    title: "Secure & Private",
    body: "Your data stays yours. All processing happens in your secure workspace with enterprise-grade security.",
    icon: "fa-shield-halved",
  },
];

const steps = [
  {
    title: "Upload Your Data",
    body: "Support for CSV files with automatic column detection, validation, and smart preprocessing.",
  },
  {
    title: "AI Selects the Best Model",
    body: "Meta-learning analyzes your data characteristics, selects the optimal algorithm, and tunes hyperparameters.",
  },
  {
    title: "Download & Deploy",
    body: "Robust cross-validation ensures quality. Download production-ready models with pipelines included.",
  },
];

/* ---- Animation helpers ---- */
const fadeUp = {
  initial: { opacity: 0, y: 30 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, margin: "-60px" },
  transition: { duration: 0.5, ease: [0.16, 1, 0.3, 1] },
};

const stagger = {
  whileInView: { transition: { staggerChildren: 0.08 } },
  viewport: { once: true, margin: "-60px" },
};

const childFadeUp = {
  initial: { opacity: 0, y: 24 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true },
  transition: { duration: 0.45, ease: [0.16, 1, 0.3, 1] },
};

/* ---- Component ---- */
const Home = () => {
  const { profile } = useSession();

  return (
    <section className="page home">
      {/* Hero */}
      <div className="hero">
        <div className="hero-dots" aria-hidden="true">
          <span /><span /><span /><span /><span /><span />
        </div>

        <motion.p
          className="eyebrow"
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          SmartML AutoML Platform
        </motion.p>

        <motion.h1
          initial={{ opacity: 0, y: 28 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1, duration: 0.55 }}
        >
          Machine Learning Made Simple for Everyone
        </motion.h1>

        <motion.p
          className="lead"
          initial={{ opacity: 0, y: 28 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.55 }}
        >
          {profile?.fname ? `Welcome back, ${profile.fname}! ` : ""}
          Transform your data into powerful predictions without writing a single
          line of code. SmartML automates the entire pipeline — from
          preprocessing to deployment.
        </motion.p>

        <motion.div
          className="hero-cta"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.35, duration: 0.5 }}
        >
          <NavLink to="/workspace" className="btn primary">
            {profile ? "Go to Workspace" : "Get Started Free"}
          </NavLink>
          <NavLink
            to={profile ? "/models" : "/auth"}
            className="btn ghost"
          >
            {profile ? "View Models" : "Learn More"}
          </NavLink>
        </motion.div>
      </div>

      {/* Stats strip */}
      <motion.div className="stat-strip" {...fadeUp} transition={{ delay: 0.1, duration: 0.6 }}>
        {stats.map((s, i) => (
          <motion.div
            key={s.label}
            className="stat-item"
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 + i * 0.08, duration: 0.4 }}
          >
            <div className="stat-icon">
              <i className={`fas ${s.icon}`} />
            </div>
            <span className="stat-value">{s.value}</span>
            <span className="stat-label">{s.label}</span>
          </motion.div>
        ))}
      </motion.div>

      {/* Features heading */}
      <motion.div {...fadeUp}>
        <h2 className="section-heading">Everything You Need in One Platform</h2>
        <p className="section-subheading">
          From data upload to model deployment, we've automated the entire
          workflow so you can focus on what matters.
        </p>
        <div className="section-divider" />
      </motion.div>

      {/* Feature grid */}
      <motion.div className="grid-3" {...stagger}>
        {features.map((f) => (
          <motion.div key={f.title} className="feature-card" {...childFadeUp}>
            <div className="feature-icon">
              <i className={`fas ${f.icon}`} />
            </div>
            <h3>{f.title}</h3>
            <p>{f.body}</p>
          </motion.div>
        ))}
      </motion.div>

      {/* How it works */}
      <motion.div className="pipeline-section" {...fadeUp}>
        <h2 className="section-heading">How It Works</h2>
        <p className="section-subheading">
          Three simple steps powered by advanced meta-learning algorithms.
        </p>
        <div className="section-divider" />
      </motion.div>

      <motion.div className="steps-row" {...stagger}>
        {steps.map((s) => (
          <motion.div key={s.title} className="step-card" {...childFadeUp}>
            <h4>{s.title}</h4>
            <p>{s.body}</p>
          </motion.div>
        ))}
      </motion.div>

      {/* CTA Banner */}
      <motion.div className="cta-banner" {...fadeUp}>
        <h2>Ready to Transform Your Data?</h2>
        <p>
          Join hundreds of users already leveraging the power of automated
          machine learning.
        </p>
        <div className="cta-actions">
          <NavLink to="/workspace" className="btn primary">
            Start Building Now
          </NavLink>
          {!profile && (
            <NavLink to="/auth" className="btn ghost">
              Create Free Account
            </NavLink>
          )}
        </div>
      </motion.div>
    </section>
  );
};

export default Home;
