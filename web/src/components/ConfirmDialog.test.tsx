import { beforeAll, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { ConfirmDialog } from "./common";

beforeAll(() => {
  HTMLDialogElement.prototype.showModal = vi.fn(function (this: HTMLDialogElement) {
    this.open = true;
  });
  HTMLDialogElement.prototype.close = vi.fn(function (this: HTMLDialogElement) {
    this.open = false;
  });
});

describe("ConfirmDialog", () => {
  it("prevents double submission while onConfirm is in flight", async () => {
    let resolveConfirm!: () => void;
    const onConfirm = vi.fn(() => new Promise<void>((r) => { resolveConfirm = r; }));
    const onCancel = vi.fn();

    render(
      <ConfirmDialog
        open={true}
        title="Delete item?"
        body="Are you sure?"
        onConfirm={onConfirm}
        onCancel={onCancel}
      />,
    );

    const confirmBtn = screen.getByRole("button", { name: "Confirm" });
    expect(confirmBtn).not.toBeDisabled();

    // First click
    fireEvent.click(confirmBtn);
    expect(onConfirm).toHaveBeenCalledTimes(1);
    expect(confirmBtn).toBeDisabled();

    // Second click while in flight
    fireEvent.click(confirmBtn);
    expect(onConfirm).toHaveBeenCalledTimes(1);

    // Resolve promise
    resolveConfirm();
  });

  it("re-enables button if onConfirm throws an error", async () => {
    const onConfirm = vi.fn().mockRejectedValueOnce(new Error("Network error"));
    const onCancel = vi.fn();

    render(
      <ConfirmDialog
        open={true}
        title="Delete item?"
        body="Are you sure?"
        onConfirm={onConfirm}
        onCancel={onCancel}
      />,
    );

    const confirmBtn = screen.getByRole("button", { name: "Confirm" });
    fireEvent.click(confirmBtn);

    expect(onConfirm).toHaveBeenCalledTimes(1);
    await vi.waitFor(() => {
      expect(confirmBtn).not.toBeDisabled();
    });
  });

  it("enforces requireText condition", () => {
    const onConfirm = vi.fn();
    const onCancel = vi.fn();

    render(
      <ConfirmDialog
        open={true}
        title="Delete item?"
        body="Are you sure?"
        requireText="DELETE"
        onConfirm={onConfirm}
        onCancel={onCancel}
      />,
    );

    const confirmBtn = screen.getByRole("button", { name: "Confirm" });
    expect(confirmBtn).toBeDisabled();

    const input = screen.getByRole("textbox");
    fireEvent.change(input, { target: { value: "DELETE" } });
    expect(confirmBtn).not.toBeDisabled();
  });
});
