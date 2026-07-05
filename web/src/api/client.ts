/**
 * Minimal fetch wrapper for the platform API (ADR-017).
 *
 * Relative base URL: in production the SPA is served by the same FastAPI
 * process; in dev, Vite proxies `/api` to the backend. Every backend error
 * carries `{detail, error_type}` — surfaced as ApiError so callers branch
 * on `error_type`, never on prose.
 */

export const API_BASE = "/api/v1";

export class ApiError extends Error {
  readonly status: number;
  readonly errorType: string;
  readonly detail: unknown;

  constructor(status: number, errorType: string, detail: unknown) {
    super(typeof detail === "string" ? detail : JSON.stringify(detail));
    this.status = status;
    this.errorType = errorType;
    this.detail = detail;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!response.ok) {
    let errorType = "unknown";
    let detail: unknown = response.statusText;
    try {
      const body = await response.json();
      errorType = body.error_type ?? "unknown";
      detail = body.detail ?? body;
    } catch {
      // non-JSON error body; keep statusText
    }
    throw new ApiError(response.status, errorType, detail);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) }),
  put: <T>(path: string, body: unknown) =>
    request<T>(path, { method: "PUT", body: JSON.stringify(body) }),
  patch: <T>(path: string, body: unknown) =>
    request<T>(path, { method: "PATCH", body: JSON.stringify(body) }),
  delete: <T>(path: string) => request<T>(path, { method: "DELETE" }),
  /** POST that returns raw bytes (voice preview PCM). */
  postBinary: async (path: string, body?: unknown): Promise<ArrayBuffer> => {
    const response = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (!response.ok) {
      throw new ApiError(response.status, "unknown", response.statusText);
    }
    return response.arrayBuffer();
  },
};
