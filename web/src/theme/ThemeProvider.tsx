/**
 * Applies `settings.ui.theme` (dark/light/system) as a `data-theme`
 * attribute on <html>, and `settings.ui.reduced_motion` + `scale`.
 * "system" tracks `prefers-color-scheme` live.
 */

import { createContext, useContext, useEffect, useState } from "react";
import type { ReactNode } from "react";
import type { UISettings } from "../api/types";

type ThemeChoice = UISettings["theme"];

interface ThemeContextValue {
  theme: ThemeChoice;
  setTheme: (theme: ThemeChoice) => void;
}

const ThemeContext = createContext<ThemeContextValue>({
  theme: "system",
  setTheme: () => {},
});

function resolve(theme: ThemeChoice): "dark" | "light" {
  if (theme !== "system") return theme;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function applyUiSettings(ui: Partial<UISettings>): void {
  const root = document.documentElement;
  if (ui.theme) root.dataset.theme = resolve(ui.theme);
  if (ui.reduced_motion !== undefined) {
    root.dataset.reducedMotion = String(ui.reduced_motion);
  }
  if (ui.scale !== undefined && ui.scale > 0) {
    root.style.fontSize = `${ui.scale * 100}%`;
  }
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<ThemeChoice>("system");

  useEffect(() => {
    document.documentElement.dataset.theme = resolve(theme);
    if (theme !== "system") return;
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => {
      document.documentElement.dataset.theme = resolve("system");
    };
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, [theme]);

  return <ThemeContext.Provider value={{ theme, setTheme }}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeContextValue {
  return useContext(ThemeContext);
}
