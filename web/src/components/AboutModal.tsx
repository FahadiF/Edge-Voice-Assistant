import { useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { system } from "../api/endpoints";
import { EvaLogo } from "./Layout";
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

  const version = health.data?.version ?? "Loading...";

  return (
    <dialog ref={ref} onCancel={onClose} onClick={onDialogClick} className="about-modal" aria-label="About Edge Voice Assistant">
      <div className="about-content" onClick={(e) => e.stopPropagation()}>
        <div className="about-header">
          <EvaLogo size={48} />
          <h2>Edge Voice Assistant</h2>
          <span className="about-version">v{version}</span>
        </div>
        
        <div className="about-body">
          <p>
            A privacy-focused, local-first voice AI application. Everything runs directly on your hardware.
          </p>
          <div className="about-links">
            <a href="https://github.com/FahadiF/Edge-Voice-Assistant" target="_blank" rel="noreferrer">
              GitHub Repository
            </a>
            <a href="https://github.com/FahadiF/Edge-Voice-Assistant/issues" target="_blank" rel="noreferrer">
              Report an Issue
            </a>
          </div>
        </div>

        <div className="about-footer">
          <button className="primary" onClick={onClose}>Close</button>
        </div>
      </div>
    </dialog>
  );
}
