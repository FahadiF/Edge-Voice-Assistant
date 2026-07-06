/**
 * Assistant-message Markdown renderer (ADR-024).
 *
 * GFM (tables, strikethrough, task lists) via remark-gfm; raw HTML in model
 * output is NOT rendered (react-markdown's default — no rehype-raw): LLM
 * output is untrusted input. Fenced code blocks get a copy button. Syntax
 * highlighting is deliberately deferred (grammar bundles are heavy for an
 * offline app — ADR-024).
 */

import { useState } from "react";
import type { ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./markdown.css";

function CodeBlock({ children }: { children: ReactNode }) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    // The <code> child's text is the block's source.
    const text = extractText(children);
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard unavailable (non-secure context) — silently ignore.
    }
  };

  return (
    <div className="md-codeblock">
      <button
        className="md-copy"
        onClick={copy}
        aria-label={copied ? "Copied" : "Copy code"}
        title="Copy code"
      >
        {copied ? "✓ copied" : "copy"}
      </button>
      <pre>{children}</pre>
    </div>
  );
}

function extractText(node: ReactNode): string {
  if (typeof node === "string") return node;
  if (Array.isArray(node)) return node.map(extractText).join("");
  if (node && typeof node === "object" && "props" in node) {
    return extractText((node as { props: { children?: ReactNode } }).props.children);
  }
  return "";
}

export function Markdown({ children }: { children: string }) {
  return (
    <div className="markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          pre: ({ children }) => <CodeBlock>{children}</CodeBlock>,
          a: ({ href, children }) => (
            // Local-first app: external links open in a new tab, never
            // navigate the SPA away.
            <a href={href} target="_blank" rel="noreferrer noopener">
              {children}
            </a>
          ),
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}
