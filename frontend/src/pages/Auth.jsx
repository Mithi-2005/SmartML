import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { loginUser, registerUser } from "../lib/api";
import { useSession } from "../context/SessionContext";

const fadeUp = {
  initial: { opacity: 0, y: 20 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, margin: "-40px" },
  transition: { duration: 0.45, ease: [0.16, 1, 0.3, 1] },
};

const authFeatures = [
  { icon: "fa-cloud-arrow-up", text: "Upload datasets and get instant analysis" },
  { icon: "fa-brain", text: "AI-powered model selection and training" },
  { icon: "fa-box-open", text: "Download production-ready model packages" },
  { icon: "fa-shield-halved", text: "Enterprise-grade security for your data" },
];

const Auth = () => {
  const { setToken, profile } = useSession();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState("signin");
  const [loginForm, setLoginForm] = useState({ email: "", password: "" });
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [registerForm, setRegisterForm] = useState({
    fname: "",
    lname: "",
    username: "",
    email: "",
    password: "",
    cpassword: "",
  });
  const [registerStatus, setRegisterStatus] = useState(null);
  const [registerLoading, setRegisterLoading] = useState(false);

  const updateLogin = (field, value) =>
    setLoginForm((prev) => ({ ...prev, [field]: value }));
  const updateRegister = (field, value) =>
    setRegisterForm((prev) => ({ ...prev, [field]: value }));

  const handleLogin = async (event) => {
    event.preventDefault();
    setStatus(null);
    setLoading(true);
    try {
      const response = await loginUser(loginForm);
      setToken(response.token);
      setStatus({ type: "success", message: "Logged in successfully." });
      navigate("/");
    } catch (error) {
      setStatus({ type: "error", message: error.message });
    } finally {
      setLoading(false);
    }
  };

  const handleRegister = async (event) => {
    event.preventDefault();
    setRegisterStatus(null);
    setRegisterLoading(true);
    try {
      const response = await registerUser(registerForm);
      setRegisterStatus({
        type: "success",
        message: response.msg || "Account created. You can sign in now.",
      });
      setRegisterForm({
        fname: "",
        lname: "",
        username: "",
        email: "",
        password: "",
        cpassword: "",
      });
    } catch (error) {
      setRegisterStatus({ type: "error", message: error.message });
    } finally {
      setRegisterLoading(false);
    }
  };

  /* ---- Already authenticated ---- */
  if (profile) {
    return (
      <section className="page auth">
        <motion.header {...fadeUp}>
          <p className="eyebrow">Already Authenticated</p>
          <h1>Welcome Back, {profile.fname || profile.username}!</h1>
          <p className="lead">
            You're already signed in. Ready to build amazing ML models?
          </p>
        </motion.header>

        <motion.div
          className="card"
          style={{ textAlign: "center", padding: "2.5rem 2rem" }}
          {...fadeUp}
        >
          <div
            style={{
              width: 72,
              height: 72,
              margin: "0 auto 1.25rem",
              background: "linear-gradient(135deg, var(--amber-500), var(--teal-500))",
              borderRadius: "50%",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: "1.75rem",
              color: "#ffffff",
            }}
          >
            <i className="fas fa-user-check" />
          </div>
          <h3 style={{ marginBottom: "0.5rem" }}>You're All Set!</h3>
          <p style={{ marginBottom: "1.5rem", fontSize: "0.9375rem" }}>
            Signed in as{" "}
            <strong style={{ color: "var(--amber-600)" }}>
              {profile.email}
            </strong>
          </p>
          <div
            style={{
              display: "flex",
              gap: "0.75rem",
              justifyContent: "center",
              flexWrap: "wrap",
            }}
          >
            <button
              onClick={() => navigate("/workspace")}
              className="btn primary"
            >
              <i className="fas fa-rocket" /> Go to Workspace
            </button>
            <button
              onClick={() => navigate("/")}
              className="btn secondary"
            >
              <i className="fas fa-house" /> Back to Home
            </button>
          </div>
        </motion.div>
      </section>
    );
  }

  /* ---- Sign in / Sign up ---- */
  return (
    <section className="page auth">
      <div className="auth-split">
        {/* Left: brand panel */}
        <motion.div
          className="auth-brand-panel"
          initial={{ opacity: 0, x: -30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.55, ease: [0.16, 1, 0.3, 1] }}
        >
          <h2>Build AI Models Without Writing Code</h2>
          <p>
            SmartML automates the entire machine learning pipeline — from data
            preprocessing to model deployment — so you can focus on insights.
          </p>
          <div className="auth-features">
            {authFeatures.map((f) => (
              <div key={f.icon} className="auth-feature-item">
                <i className={`fas ${f.icon}`} />
                <span>{f.text}</span>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Right: form panel */}
        <motion.div
          className="auth-form-panel"
          initial={{ opacity: 0, x: 30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.55, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
        >
          <div className="auth-container">
            <div className="auth-tabs">
              <button
                type="button"
                className={`auth-tab ${activeTab === "signin" ? "active" : ""}`}
                onClick={() => setActiveTab("signin")}
              >
                <i className="fas fa-key" /> Sign In
              </button>
              <button
                type="button"
                className={`auth-tab ${activeTab === "signup" ? "active" : ""}`}
                onClick={() => setActiveTab("signup")}
              >
                <i className="fas fa-user-plus" /> Create Account
              </button>
            </div>

            <AnimatePresence mode="wait">
              {activeTab === "signin" && (
                <motion.form
                  key="signin"
                  onSubmit={handleLogin}
                  className="auth-form"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.25 }}
                >
                  {status && (
                    <div className={`alert ${status.type}`}>
                      <i
                        className={`fas fa-${
                          status.type === "success"
                            ? "check-circle"
                            : "triangle-exclamation"
                        }`}
                      />
                      <span>{status.message}</span>
                    </div>
                  )}
                  <label>
                    Email Address
                    <input
                      type="email"
                      placeholder="you@example.com"
                      value={loginForm.email}
                      onChange={(e) => updateLogin("email", e.target.value)}
                      required
                    />
                  </label>
                  <label>
                    Password
                    <input
                      type="password"
                      placeholder="Enter your password"
                      value={loginForm.password}
                      onChange={(e) => updateLogin("password", e.target.value)}
                      required
                    />
                  </label>
                  <button className="btn primary full" disabled={loading}>
                    {loading ? (
                      <>
                        <i className="fas fa-spinner fa-spin" /> Signing In…
                      </>
                    ) : (
                      <>
                        <i className="fas fa-arrow-right-to-bracket" /> Sign In
                      </>
                    )}
                  </button>
                </motion.form>
              )}

              {activeTab === "signup" && (
                <motion.form
                  key="signup"
                  onSubmit={handleRegister}
                  className="auth-form"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.25 }}
                >
                  {registerStatus && (
                    <div className={`alert ${registerStatus.type}`}>
                      <i
                        className={`fas fa-${
                          registerStatus.type === "success"
                            ? "check-circle"
                            : "triangle-exclamation"
                        }`}
                      />
                      <span>{registerStatus.message}</span>
                    </div>
                  )}
                  <div className="form-grid">
                    <label>
                      First Name
                      <input
                        type="text"
                        placeholder="John"
                        value={registerForm.fname}
                        onChange={(e) =>
                          updateRegister("fname", e.target.value)
                        }
                        required
                      />
                    </label>
                    <label>
                      Last Name
                      <input
                        type="text"
                        placeholder="Doe"
                        value={registerForm.lname}
                        onChange={(e) =>
                          updateRegister("lname", e.target.value)
                        }
                        required
                      />
                    </label>
                  </div>
                  <label>
                    Username
                    <input
                      type="text"
                      placeholder="johndoe"
                      value={registerForm.username}
                      onChange={(e) =>
                        updateRegister("username", e.target.value)
                      }
                      required
                    />
                  </label>
                  <label>
                    Email Address
                    <input
                      type="email"
                      placeholder="you@example.com"
                      value={registerForm.email}
                      onChange={(e) =>
                        updateRegister("email", e.target.value)
                      }
                      required
                    />
                  </label>
                  <div className="form-grid">
                    <label>
                      Password
                      <input
                        type="password"
                        placeholder="Create password"
                        value={registerForm.password}
                        onChange={(e) =>
                          updateRegister("password", e.target.value)
                        }
                        required
                      />
                    </label>
                    <label>
                      Confirm Password
                      <input
                        type="password"
                        placeholder="Confirm password"
                        value={registerForm.cpassword}
                        onChange={(e) =>
                          updateRegister("cpassword", e.target.value)
                        }
                        required
                      />
                    </label>
                  </div>
                  <button
                    className="btn primary full"
                    disabled={registerLoading}
                  >
                    {registerLoading ? (
                      <>
                        <i className="fas fa-spinner fa-spin" /> Creating
                        Account…
                      </>
                    ) : (
                      <>
                        <i className="fas fa-rocket" /> Create Free Account
                      </>
                    )}
                  </button>
                </motion.form>
              )}
            </AnimatePresence>
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default Auth;
