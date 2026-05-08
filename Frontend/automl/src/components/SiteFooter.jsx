import { motion } from "framer-motion";

const SiteFooter = () => (
  <motion.footer
    className="site-footer"
    initial={{ opacity: 0, y: 20 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true, margin: "-50px" }}
    transition={{ duration: 0.5 }}
  >
    <div className="footer-inner">
      <div className="footer-summary">
        <p className="brand-mark">SmartML</p>
        <p className="footer-note">
          Automated model building powered by meta-learning — from raw data to
          deployment-ready predictions.
        </p>
        <small className="footer-copy">
          © {new Date().getFullYear()} SmartML. Built for faster
          experimentation.
        </small>
      </div>

      <div className="footer-links">
        <a href="mailto:support@smartml.ai">
          <i className="fas fa-envelope" />
          Support
        </a>
        <a
          href="https://fastapi.tiangolo.com/"
          target="_blank"
          rel="noreferrer"
        >
          <i className="fas fa-server" />
          FastAPI
        </a>
        <a href="https://react.dev/" target="_blank" rel="noreferrer">
          <i className="fab fa-react" />
          React
        </a>
      </div>
    </div>
  </motion.footer>
);

export default SiteFooter;
