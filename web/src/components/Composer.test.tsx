/** Composer tests (M5.3): typed send via /conversation/say, keyboard
 * behavior, attachment placeholders, disabled-when-stopped. */

import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Composer } from "./Composer";
import { ToastHost } from "./common";

function renderComposer(engineRunning = true) {
  const queryClient = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <Composer engineRunning={engineRunning} />
      <ToastHost />
    </QueryClientProvider>,
  );
}

function mockSayOk() {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: async () => ({ status: "accepted" }),
  } as Response);
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => vi.unstubAllGlobals());

describe("Composer", () => {
  it("sends the trimmed text to /conversation/say on click", async () => {
    const fetchMock = mockSayOk();
    renderComposer();
    fireEvent.change(screen.getByLabelText("Message"), { target: { value: "  hello  " } });
    fireEvent.click(screen.getByLabelText("Send message"));
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/v1/conversation/say");
    expect(JSON.parse(init.body)).toEqual({ text: "hello" });
  });

  it("Enter sends; Shift+Enter does not", async () => {
    const fetchMock = mockSayOk();
    renderComposer();
    const input = screen.getByLabelText("Message");
    fireEvent.change(input, { target: { value: "line one" } });
    fireEvent.keyDown(input, { key: "Enter", shiftKey: true });
    expect(fetchMock).not.toHaveBeenCalled();
    fireEvent.keyDown(input, { key: "Enter" });
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
  });

  it("clears the input after a successful send", async () => {
    mockSayOk();
    renderComposer();
    const input = screen.getByLabelText("Message") as HTMLTextAreaElement;
    fireEvent.change(input, { target: { value: "hello" } });
    fireEvent.keyDown(input, { key: "Enter" });
    await waitFor(() => expect(input.value).toBe(""));
  });

  it("is disabled when the engine is stopped", () => {
    renderComposer(false);
    expect(screen.getByLabelText("Message")).toBeDisabled();
    expect(screen.getByLabelText("Send message")).toBeDisabled();
  });

  it("send stays disabled for empty text even when running", () => {
    renderComposer();
    expect(screen.getByLabelText("Send message")).toBeDisabled();
  });

  it("the + menu offers Vision-coming-soon placeholder actions", () => {
    renderComposer();
    fireEvent.click(screen.getByLabelText("Add attachment"));
    expect(screen.getByRole("menu")).toBeInTheDocument();
    fireEvent.click(screen.getByText(/Attach image/));
    // Action is a placeholder: a toast explains, no crash, menu closes.
    expect(screen.queryByRole("menu")).toBeNull();
    expect(screen.getByText(/Vision support \(coming soon\)/i)).toBeInTheDocument();
  });

  it("dropped files become removable placeholder chips", () => {
    renderComposer();
    const composer = screen.getByLabelText("Message").closest(".composer")!;
    const file = new File(["x"], "photo.png", { type: "image/png" });
    fireEvent.drop(composer, { dataTransfer: { files: [file] } });
    expect(screen.getByText(/photo\.png/)).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText("Remove photo.png"));
    expect(screen.queryByText(/photo\.png/)).toBeNull();
  });

  it("mic button offers to start the engine when stopped", () => {
    renderComposer(false);
    expect(screen.getByLabelText("Start the engine")).toBeInTheDocument();
  });

  it("mic button starts the engine on click when stopped", async () => {
    const fetchMock = mockSayOk(); // any ok JSON response works for /engine/start
    renderComposer(false);
    fireEvent.click(screen.getByLabelText("Start the engine"));
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    expect(fetchMock.mock.calls[0][0]).toBe("/api/v1/engine/start");
  });
});
