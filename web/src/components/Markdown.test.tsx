/** Markdown rendering regression tests (ADR-024): every construct the
 * milestone brief names must render as real elements — and raw HTML in
 * model output must NOT become live DOM. */

import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { Markdown } from "./Markdown";

describe("Markdown renderer", () => {
  it("renders bold as <strong>", () => {
    const { container } = render(<Markdown>{"My name is **Edge Voice Assistant**."}</Markdown>);
    expect(container.querySelector("strong")).toHaveTextContent("Edge Voice Assistant");
  });

  it("renders italic as <em>", () => {
    const { container } = render(<Markdown>{"This is *subtle*."}</Markdown>);
    expect(container.querySelector("em")).toHaveTextContent("subtle");
  });

  it("renders headings", () => {
    render(<Markdown>{"## Setup steps"}</Markdown>);
    expect(screen.getByRole("heading", { level: 2 })).toHaveTextContent("Setup steps");
  });

  it("renders inline code", () => {
    const { container } = render(<Markdown>{"Run `eva doctor` now."}</Markdown>);
    expect(container.querySelector("code")).toHaveTextContent("eva doctor");
  });

  it("renders fenced code blocks inside a copyable block", () => {
    const { container } = render(<Markdown>{"```python\nx = 1\n```"}</Markdown>);
    expect(container.querySelector("pre code")).toHaveTextContent("x = 1");
    expect(screen.getByRole("button", { name: /copy/i })).toBeInTheDocument();
  });

  it("renders blockquotes", () => {
    const { container } = render(<Markdown>{"> quoted"}</Markdown>);
    expect(container.querySelector("blockquote")).toHaveTextContent("quoted");
  });

  it("renders numbered lists", () => {
    const { container } = render(<Markdown>{"1. first\n2. second"}</Markdown>);
    const items = container.querySelectorAll("ol li");
    expect(items).toHaveLength(2);
    expect(items[0]).toHaveTextContent("first");
  });

  it("renders bullet lists", () => {
    const { container } = render(<Markdown>{"- one\n- two"}</Markdown>);
    expect(container.querySelectorAll("ul li")).toHaveLength(2);
  });

  it("renders GFM tables", () => {
    const table = "| Name | Size |\n|------|------|\n| Qwen | 4B |";
    const { container } = render(<Markdown>{table}</Markdown>);
    expect(container.querySelector("table")).toBeInTheDocument();
    expect(container.querySelectorAll("td")).toHaveLength(2);
  });

  it("renders links that open in a new tab", () => {
    render(<Markdown>{"[the docs](https://example.com)"}</Markdown>);
    const link = screen.getByRole("link", { name: "the docs" });
    expect(link).toHaveAttribute("href", "https://example.com");
    expect(link).toHaveAttribute("target", "_blank");
    expect(link.getAttribute("rel")).toContain("noreferrer");
  });

  it("does NOT render raw HTML from model output", () => {
    const { container } = render(
      <Markdown>{'before <img src=x onerror="alert(1)"> after'}</Markdown>,
    );
    expect(container.querySelector("img")).toBeNull();
  });

  it("renders GFM strikethrough", () => {
    const { container } = render(<Markdown>{"~~wrong~~ right"}</Markdown>);
    expect(container.querySelector("del")).toHaveTextContent("wrong");
  });
});
