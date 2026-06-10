import { useState, useEffect } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { useSession } from "../context/SessionContext";

const links = [
  { path: "/", label: "Home", icon: "fa-house" },
  { path: "/workspace", label: "Workspace", icon: "fa-flask" },
  { path: "/models", label: "Models", icon: "fa-cubes" },
];

const Navigation = () => {
  const { profile, logout } = useSession();
  const navigate = useNavigate();
  const [menuOpen, setMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  const closeMenu = () => setMenuOpen(false);

  const handleLogout = () => {
    logout();
    closeMenu();
    navigate("/auth");
  };

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header className={`nav-shell ${scrolled ? "scrolled" : ""}`}>
      <NavLink to="/" className="brand" onClick={closeMenu}>
        <span className="brand-emblem" aria-hidden="true">
          <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path
              d="M16 20L32 12L48 20V44L32 52L16 44V20Z"
              stroke="currentColor"
              strokeWidth="2.5"
            />
            <path
              d="M16 20L32 30L48 20M32 30V52"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
            />
            <circle cx="16" cy="20" r="4" fill="currentColor" />
            <circle cx="32" cy="12" r="4" fill="currentColor" />
            <circle cx="48" cy="20" r="4" fill="currentColor" />
            <circle cx="32" cy="30" r="4" fill="currentColor" />
            <circle cx="32" cy="52" r="4" fill="currentColor" />
          </svg>
        </span>
        <span className="brand-copy">
          <span className="brand-mark">SmartML</span>
          <span className="brand-subtitle">Adaptive AutoML System</span>
        </span>
      </NavLink>

      <button
        type="button"
        className={`nav-toggle ${menuOpen ? "active" : ""}`}
        onClick={() => setMenuOpen((c) => !c)}
        aria-label="Toggle navigation"
        aria-expanded={menuOpen}
      >
        <span />
        <span />
        <span />
      </button>

      <nav className={menuOpen ? "open" : ""}>
        {links.map((link) => (
          <NavLink
            key={link.path}
            to={link.path}
            className={({ isActive }) =>
              `nav-link${isActive ? " active" : ""}`
            }
            onClick={closeMenu}
          >
            <i className={`fas ${link.icon}`} />
            {link.label}
          </NavLink>
        ))}
        {profile ? (
          <>
            <NavLink
              to="/auth"
              className="nav-link nav-cta"
              onClick={closeMenu}
            >
              <i className="fas fa-user-gear" />
              {profile.fname}
            </NavLink>
            <button
              type="button"
              className="nav-link nav-logout"
              onClick={handleLogout}
            >
              <i className="fas fa-arrow-right-from-bracket" />
              Sign Out
            </button>
          </>
        ) : (
          <NavLink
            to="/auth"
            className="nav-link nav-cta"
            onClick={closeMenu}
          >
            <i className="fas fa-arrow-right-to-bracket" />
            Sign In
          </NavLink>
        )}
      </nav>
    </header>
  );
};

export default Navigation;
