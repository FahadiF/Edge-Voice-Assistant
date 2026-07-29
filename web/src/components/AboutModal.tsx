import { useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { Globe, GitBranch, Bug, ExternalLink } from "lucide-react";
import { system } from "../api/endpoints";
import { EvaLogo } from "./EvaLogo";
import "./about.css";

export function AboutModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null);
  const health = useQuery({ queryKey: ["system-health"], queryFn: system.health });

  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (open && !dialog.open) {
      dialog.showModal();
    } else if (!open && dialog.open) {
      dialog.close();
    }
  }, [open]);

  const onDialogClick = (e: React.MouseEvent<HTMLDialogElement>) => {
    if (e.target === ref.current) onClose();
  };

  const versionText = health.isError
    ? "Version unavailable"
    : (health.data?.version ? `v${health.data.version}` : "Loading...");

  return (
    <dialog ref={ref} onCancel={onClose} onClick={onDialogClick} className="about-modal" aria-label="About Edge Voice Assistant">
      <div className="about-content" onClick={(e) => e.stopPropagation()}>
        <div className="about-header">
          <EvaLogo size={48} />
          <h2>Edge Voice Assistant</h2>
          <span className="about-version">{versionText}</span>
        </div>
        
        <div className="about-body">
          <p className="about-desc">
            A privacy-focused, local-first voice AI application. Everything runs directly on your hardware.
          </p>

          <div className="about-links">
            <a href="https://www.fahadibnefahian.com/" target="_blank" rel="noreferrer" className="about-link-item">
              <Globe size={18} className="about-link-icon" />
              <span>Developer Website</span>
              <ExternalLink size={14} className="about-link-ext" />
            </a>
            <a href="https://github.com/FahadiF/Edge-Voice-Assistant" target="_blank" rel="noreferrer" className="about-link-item">
              <GitBranch size={18} className="about-link-icon" />
              <span>GitHub Repository</span>
              <ExternalLink size={14} className="about-link-ext" />
            </a>
            <a href="https://github.com/FahadiF/Edge-Voice-Assistant/issues" target="_blank" rel="noreferrer" className="about-link-item">
              <Bug size={18} className="about-link-icon" />
              <span>Report an Issue</span>
              <ExternalLink size={14} className="about-link-ext" />
            </a>
          </div>

          <div className="about-acknowledgements">
            <h3>Acknowledgements</h3>
            <p>
              Sincere gratitude to my thesis supervisor, <a href="https://github.com/jboutell" target="_blank" rel="noreferrer">Jani Boutellier</a>, for his guidance and support throughout the research that inspired this project.
            </p>
          </div>
        </div>

        <div className="about-footer">
          <button className="primary" onClick={onClose}>Close</button>
        </div>
      </div>
    </dialog>
  );
}
