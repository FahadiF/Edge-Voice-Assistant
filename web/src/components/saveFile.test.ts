/** saveTextFile tests (M6.2 desktop polish): the export must use a NATIVE
 * Save-As on desktop (pywebview bridge) and report the path, and fall back to
 * a browser download otherwise — so a user always knows where the file went. */

import { afterEach, describe, expect, it, vi } from "vitest";
import { saveTextFile } from "./common";

afterEach(() => {
  vi.unstubAllGlobals();
  delete (window as { pywebview?: unknown }).pywebview;
});

describe("saveTextFile", () => {
  it("uses the native Save-As bridge on desktop and returns the chosen path", async () => {
    const save = vi.fn(() => Promise.resolve({ status: "saved" as const, path: "C:\\logs\\eva.txt" }));
    (window as { pywebview?: unknown }).pywebview = { api: { save_text_file: save } };

    const result = await saveTextFile("hello", "eva.txt");

    expect(save).toHaveBeenCalledWith("eva.txt", "hello");
    expect(result).toEqual({ outcome: "saved", path: "C:\\logs\\eva.txt" });
  });

  it("reports a cancelled native dialog without error", async () => {
    const save = vi.fn(() => Promise.resolve({ status: "cancelled" as const }));
    (window as { pywebview?: unknown }).pywebview = { api: { save_text_file: save } };

    expect(await saveTextFile("x", "eva.txt")).toEqual({ outcome: "cancelled" });
  });

  it("surfaces a native write error", async () => {
    const save = vi.fn(() =>
      Promise.resolve({ status: "error" as const, message: "disk full" }),
    );
    (window as { pywebview?: unknown }).pywebview = { api: { save_text_file: save } };

    expect(await saveTextFile("x", "eva.txt")).toEqual({ outcome: "error", message: "disk full" });
  });

  it("falls back to a browser download when no native bridge is present", async () => {
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    const createUrl = vi.fn(() => "blob:fake");
    vi.stubGlobal("URL", { createObjectURL: createUrl, revokeObjectURL: vi.fn() });

    const result = await saveTextFile("body", "eva.txt");

    expect(clickSpy).toHaveBeenCalled();
    expect(createUrl).toHaveBeenCalled();
    expect(result).toEqual({ outcome: "saved" }); // no path — browser owns the location
  });
});
