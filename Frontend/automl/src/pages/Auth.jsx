import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { loginUser, registerUser } from "../lib/api";
import { useSession } from "../context/SessionContext";

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

  const updateLogin = (field, value) => {
    setLoginForm((prev) => ({ ...prev, [field]: value }));
  };

  const updateRegister = (field, value) => {
    setRegisterForm((prev) => ({ ...prev, [field]: value }));
  };

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

  return (
    <section className="page auth">
      <header>
        <p className="eyebrow">Account Access</p>
        <h1>Welcome to SmartML</h1>
        <p className="lead">
          Sign in to access your personal AutoML workspace or create a new
          account to get started with automated machine learning.
        </p>
      </header>

      <div className="auth-container">
        <div className="auth-tabs">
          <button
            type="button"
            className={`auth-tab ${activeTab === "signin" ? "active" : ""}`}
            onClick={() => setActiveTab("signin")}
          >
            🔑 Sign In
          </button>
          <button
            type="button"
            className={`auth-tab ${activeTab === "signup" ? "active" : ""}`}
            onClick={() => setActiveTab("signup")}
          >
            ✨ Create Account
          </button>
        </div>

        {activeTab === "signin" && (
          <form onSubmit={handleLogin} className="auth-form">
            {status && (
              <div className={`alert ${status.type}`}>
                {status.type === "success" && "✓ "}
                {status.type === "error" && "⚠ "}
                <span>{status.message}</span>
              </div>
            )}
            {profile && (
              <div className="alert info">
                ✓ Currently signed in as {profile.email}
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
              {loading ? "⏳ Signing In..." : "→ Sign In"}
            </button>
          </form>
        )}

        {activeTab === "signup" && (
          <form onSubmit={handleRegister} className="auth-form">
            {registerStatus && (
              <div className={`alert ${registerStatus.type}`}>
                {registerStatus.type === "success" && "✓ "}
                {registerStatus.type === "error" && "⚠ "}
                {registerStatus.message}
              </div>
            )}
            <div className="form-grid">
              <label>
                First Name
                <input
                  type="text"
                  placeholder="John"
                  value={registerForm.fname}
                  onChange={(e) => updateRegister("fname", e.target.value)}
                  required
                />
              </label>
              <label>
                Last Name
                <input
                  type="text"
                  placeholder="Doe"
                  value={registerForm.lname}
                  onChange={(e) => updateRegister("lname", e.target.value)}
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
                onChange={(e) => updateRegister("username", e.target.value)}
                required
              />
            </label>
            <label>
              Email Address
              <input
                type="email"
                placeholder="you@example.com"
                value={registerForm.email}
                onChange={(e) => updateRegister("email", e.target.value)}
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
                  onChange={(e) => updateRegister("password", e.target.value)}
                  required
                />
              </label>
              <label>
                Confirm Password
                <input
                  type="password"
                  placeholder="Confirm password"
                  value={registerForm.cpassword}
                  onChange={(e) => updateRegister("cpassword", e.target.value)}
                  required
                />
              </label>
            </div>
            <button className="btn primary full" disabled={registerLoading}>
              {registerLoading
                ? "⏳ Creating Account..."
                : "🚀 Create Free Account"}
            </button>
          </form>
        )}
      </div>

      <div
        className="card"
        style={{
          textAlign: "center",
          padding: "2rem",
          background:
            "linear-gradient(135deg, rgba(14, 165, 233, 0.08), rgba(34, 197, 94, 0.08))",
          border: "1px solid rgba(14, 165, 233, 0.2)",
        }}
      >
        <h3 style={{ marginBottom: "0.75rem" }}>🎯 Why SmartML?</h3>
        <p style={{ color: "var(--gray-400)" }}>
          Join hundreds of users building AI models without code. Our platform
          automates the entire ML pipeline, from data preprocessing to model
          deployment.
        </p>
      </div>
    </section>
  );
};

export default Auth;
