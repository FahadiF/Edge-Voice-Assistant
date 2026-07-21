/**
 * Regression guard for a measured bug: `animation-duration: 0.01ms !important`
 * does NOT stop an `infinite` animation — it makes it loop ~100,000 times/sec,
 * which renders as rapid, erratic flicker (confirmed via getComputedStyle in a
 * live browser: duration collapsed to 1e-05s while animation-iteration-count
 * stayed `infinite`). This is what caused the reported "status indicator
 * blinks rapidly" bug under `prefers-reduced-motion: reduce`. The fix uses
 * `animation: none !important`, which genuinely stops the animation. jsdom
 * doesn't evaluate media queries or computed animation state, so this test
 * asserts the CSS source text directly rather than rendering it.
 */

import { describe, expect, it } from "vitest";
import rawCss from "./tokens.css?raw";

// Strip comments so the explanatory comment describing the bug (which quotes
// the broken pattern verbatim) doesn't trip the regression check below.
const css = rawCss.replace(/\/\*[\s\S]*?\*\//g, "");

describe("reduced-motion CSS override", () => {
  it("never uses the broken 0.01ms-duration-only pattern for infinite animations", () => {
    // The bug pattern: setting only animation-duration (leaving animation-name
    // and iteration-count untouched) inside a reduced-motion block/selector.
    expect(css).not.toMatch(/animation-duration:\s*0\.01ms/);
  });

  it("stops animations outright (animation: none) under prefers-reduced-motion", () => {
    const mediaBlock = css.match(/@media \(prefers-reduced-motion: reduce\) \{[\s\S]*?\n\}/);
    expect(mediaBlock).not.toBeNull();
    expect(mediaBlock![0]).toMatch(/animation:\s*none\s*!important/);
  });

  it("stops animations outright (animation: none) for the app's own reduced-motion setting", () => {
    const attrBlock = css.match(/\[data-reduced-motion="true"\][\s\S]*?\n\}/);
    expect(attrBlock).not.toBeNull();
    expect(attrBlock![0]).toMatch(/animation:\s*none\s*!important/);
  });
});
