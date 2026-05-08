import { useEffect, useState } from "react";
import Navigation from "./Navigation";
import SiteFooter from "./SiteFooter";

const Layout = ({ children }) => {
  const [booting, setBooting] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => setBooting(false), 1600);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className={`app-shell ${booting ? "app-shell-lock" : ""}`}>
      {/* Loader */}
      <div className={`app-loader ${booting ? "is-visible" : ""}`}>
        <div className="loader-shapes" aria-hidden="true">
          <span className="loader-shape-a" />
          <span className="loader-shape-b" />
          <span className="loader-shape-c" />
        </div>
        <div className="loader-text">
          <p className="loader-kicker">SmartML</p>
          <h2>Preparing your workspace…</h2>
        </div>
      </div>

      {/* Background decoration */}
      <div className="bg-mesh" aria-hidden="true" />
      <div className="bg-grain" aria-hidden="true" />
      <div className="bg-grid-pattern" aria-hidden="true" />

      {/* Main frame */}
      <div className="app-frame">
        <Navigation />
        <main className="site-main">{children}</main>
        <SiteFooter />
      </div>
    </div>
  );
};

export default Layout;
