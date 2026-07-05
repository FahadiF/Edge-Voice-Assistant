import { afterEach, describe, expect, it, vi } from "vitest";
import { api, ApiError } from "./client";

function mockFetch(status: number, body: unknown, ok = status < 400) {
  return vi.fn().mockResolvedValue({
    ok,
    status,
    statusText: "error",
    json: async () => body,
  } as Response);
}

describe("api client", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("returns parsed JSON on success", async () => {
    vi.stubGlobal("fetch", mockFetch(200, { hello: "world" }));
    const result = await api.get<{ hello: string }>("/health");
    expect(result).toEqual({ hello: "world" });
  });

  it("throws ApiError with error_type and detail on failure", async () => {
    vi.stubGlobal(
      "fetch",
      mockFetch(404, { detail: "not found", error_type: "MemoryNotFoundError" }, false),
    );
    await expect(api.get("/memory/turns/999")).rejects.toMatchObject({
      status: 404,
      errorType: "MemoryNotFoundError",
      detail: "not found",
    });
  });

  it("ApiError instances are recognizable via instanceof", async () => {
    vi.stubGlobal("fetch", mockFetch(500, { detail: "boom", error_type: "unknown" }, false));
    try {
      await api.get("/health");
      expect.unreachable();
    } catch (e) {
      expect(e).toBeInstanceOf(ApiError);
    }
  });

  it("falls back gracefully when the error body isn't JSON", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 502,
        statusText: "Bad Gateway",
        json: async () => {
          throw new Error("not json");
        },
      } as unknown as Response),
    );
    await expect(api.get("/health")).rejects.toMatchObject({ status: 502, errorType: "unknown" });
  });

  it("sends the request body as JSON on post/patch", async () => {
    const fetchMock = mockFetch(200, {});
    vi.stubGlobal("fetch", fetchMock);
    await api.patch("/settings", { ui: { theme: "dark" } });
    const [, init] = fetchMock.mock.calls[0];
    expect(init.method).toBe("PATCH");
    expect(JSON.parse(init.body)).toEqual({ ui: { theme: "dark" } });
  });
});
