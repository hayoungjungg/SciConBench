/* Minimal SVG charting for the SciConBench dashboard.
 *
 * Hand-rolled rather than pulled from a CDN: the site is a static bundle on a
 * university host, so it should render with zero third-party requests. */

(function (global) {
  "use strict";

  const NS = "http://www.w3.org/2000/svg";

  function el(name, attrs, text) {
    const node = document.createElementNS(NS, name);
    for (const key in attrs || {}) {
      if (attrs[key] !== null && attrs[key] !== undefined) {
        node.setAttribute(key, attrs[key]);
      }
    }
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function niceTicks(min, max, count) {
    const span = max - min || 1;
    const raw = span / count;
    const magnitude = Math.pow(10, Math.floor(Math.log10(raw)));
    const normalized = raw / magnitude;
    const step =
      (normalized >= 5 ? 10 : normalized >= 2 ? 5 : normalized >= 1 ? 2 : 1) * magnitude;
    const ticks = [];
    for (let t = Math.ceil(min / step) * step; t <= max + step / 1000; t += step) {
      ticks.push(Number(t.toFixed(10)));
    }
    return ticks;
  }

  /* Tooltip shared by every chart on the page. */
  let tip;
  let activeTipNode = null;

  function hideTip() {
    if (tip) tip.style.opacity = "0";
    activeTipNode = null;
  }

  function tooltip() {
    if (!tip) {
      tip = document.createElement("div");
      Object.assign(tip.style, {
        position: "fixed",
        zIndex: "100",
        pointerEvents: "none",
        opacity: "0",
        transition: "opacity 0.12s",
        padding: "7px 10px",
        borderRadius: "6px",
        background: "#ffffff",
        border: "1px solid #dfe3e8",
        boxShadow: "0 4px 14px rgba(16,24,40,0.12)",
        font: "12px/1.5 -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        color: "#1c1f23",
        whiteSpace: "nowrap",
      });
      document.body.appendChild(tip);
    }
    return tip;
  }

  function bindTip(node, html) {
    node.addEventListener("mouseenter", (event) => {
      const t = tooltip();
      activeTipNode = node;
      t.innerHTML = html;
      t.style.opacity = "1";
      move(event);
    });
    node.addEventListener("mousemove", move);
    node.addEventListener("mouseleave", hideTip);
    function move(event) {
      const t = tooltip();
      const box = t.getBoundingClientRect();
      let x = event.clientX + 14;
      if (x + box.width > window.innerWidth - 12) x = event.clientX - box.width - 14;
      t.style.left = x + "px";
      t.style.top = Math.max(12, event.clientY - box.height - 12) + "px";
    }
  }

  // A chart redraw can remove the hovered SVG node before `mouseleave` fires.
  // Hide the shared tooltip whenever the pointer is no longer over its owner.
  document.addEventListener("mousemove", (event) => {
    if (activeTipNode && !activeTipNode.contains(event.target)) hideTip();
  }, true);
  window.addEventListener("blur", hideTip);
  window.addEventListener("scroll", hideTip, true);

  /* ---------------------------------------------------------------- *
   * Provider "logomarks" — small abstract glyphs drawn in a 16x16 box
   * centered at the origin, used to badge each chart point instead of a
   * plain dot. Deliberately generic/abstract rather than reproductions
   * of real trademarks.
   * ---------------------------------------------------------------- */
  const ICON_DRAWERS = {
    openai: (g) => {
      for (let i = 0; i < 6; i++) {
        const a = (Math.PI / 3) * i;
        g.appendChild(
          el("circle", { cx: Math.cos(a) * 4.6, cy: Math.sin(a) * 4.6, r: 2.5, fill: "#fff" })
        );
      }
    },
    anthropic: (g) => {
      for (let i = 0; i < 6; i++) {
        const a = (Math.PI / 3) * i;
        g.appendChild(
          el("line", {
            x1: 0, y1: 0, x2: Math.cos(a) * 6.6, y2: Math.sin(a) * 6.6,
            stroke: "#fff", "stroke-width": 2, "stroke-linecap": "round",
          })
        );
      }
    },
    gemini: (g) => {
      g.appendChild(
        el("path", {
          d: "M0 -7 C1.4 -1.8 1.8 -1.4 7 0 C1.8 1.4 1.4 1.8 0 7 C-1.4 1.8 -1.8 1.4 -7 0 C-1.8 -1.4 -1.4 -1.8 0 -7 Z",
          fill: "#fff",
        })
      );
    },
    azure: (g) => {
      g.appendChild(
        el("path", {
          d: "M-7,0 C-7,-4.5 -3,-4.5 0,0 C3,4.5 7,4.5 7,0 C7,-4.5 3,-4.5 0,0 C-3,4.5 -7,4.5 -7,0",
          fill: "none", stroke: "#fff", "stroke-width": 1.9, "stroke-linecap": "round",
        })
      );
    },
    openrouter: (g) => {
      const pts = [[0, -6.5], [-5.8, 3.8], [5.8, 3.8]];
      pts.forEach(([x1, y1], i) => {
        const [x2, y2] = pts[(i + 1) % pts.length];
        g.appendChild(
          el("line", { x1, y1, x2, y2, stroke: "#fff", "stroke-width": 1.4, opacity: 0.85 })
        );
      });
      pts.forEach(([cx, cy]) => g.appendChild(el("circle", { cx, cy, r: 1.7, fill: "#fff" })));
    },
    perplexity: (g) => {
      for (let i = 0; i < 4; i++) {
        const a = (Math.PI / 4) * i;
        g.appendChild(
          el("line", {
            x1: -Math.cos(a) * 6.4, y1: -Math.sin(a) * 6.4,
            x2: Math.cos(a) * 6.4, y2: Math.sin(a) * 6.4,
            stroke: "#fff", "stroke-width": 1.6, "stroke-linecap": "round",
          })
        );
      }
      g.appendChild(el("circle", { cx: 0, cy: 0, r: 1.5, fill: "#fff" }));
    },
    deepseek: (g) => {
      g.appendChild(
        el("path", {
          d: "M-6.5,-1.5 Q-3.5,-6.5 0,-1.5 Q3.5,3.5 6.5,-1.5",
          fill: "none", stroke: "#fff", "stroke-width": 1.9, "stroke-linecap": "round",
        })
      );
      g.appendChild(el("circle", { cx: 0, cy: 4.6, r: 1.3, fill: "#fff" }));
    },
    xai: (g) => {
      g.appendChild(el("line", { x1: -5.6, y1: -5.6, x2: 5.6, y2: 5.6, stroke: "#fff", "stroke-width": 2.2, "stroke-linecap": "round" }));
      g.appendChild(el("line", { x1: -5.6, y1: 5.6, x2: 5.6, y2: -5.6, stroke: "#fff", "stroke-width": 2.2, "stroke-linecap": "round" }));
    },
    meta: (g) => {
      g.appendChild(el("circle", { cx: -3.1, cy: 0, r: 4.6, fill: "none", stroke: "#fff", "stroke-width": 1.7 }));
      g.appendChild(el("circle", { cx: 3.1, cy: 0, r: 4.6, fill: "none", stroke: "#fff", "stroke-width": 1.7 }));
    },
    mistral: (g) => {
      [-4.2, 0, 4.2].forEach((x, i) => {
        g.appendChild(
          el("rect", {
            x: x - 1.1, y: -6 + i * 0.6, width: 2.2, height: 12 - i * 1.2,
            rx: 1, fill: "#fff", opacity: 0.55 + i * 0.22,
          })
        );
      });
    },
    qwen: (g) => {
      g.appendChild(
        el("path", {
          d: "M-6.4,2.4 a3,3 0 0 1 0.4,-5.9 a3.6,3.6 0 0 1 6.7,-1.3 a3.2,3.2 0 0 1 2.7,6.4 a2,2 0 0 1 -0.2,0.8 Z",
          fill: "#fff",
        })
      );
    },
    kimi: (g) => {
      g.appendChild(el("circle", { cx: -2.2, cy: 0, r: 5.2, fill: "none", stroke: "#fff", "stroke-width": 1.7 }));
      g.appendChild(el("line", { x1: 1.2, y1: -4.6, x2: 6.4, y2: -4.6, stroke: "#fff", "stroke-width": 1.9, "stroke-linecap": "round" }));
      g.appendChild(el("line", { x1: 1.2, y1: 0, x2: 6.8, y2: 0, stroke: "#fff", "stroke-width": 1.9, "stroke-linecap": "round" }));
      g.appendChild(el("line", { x1: 1.2, y1: 4.6, x2: 5.6, y2: 4.6, stroke: "#fff", "stroke-width": 1.9, "stroke-linecap": "round" }));
    },
    glm: (g) => {
      g.appendChild(
        el("path", {
          d: "M-6.5,-5 L-6.5,5 M-6.5,-5 L-1,-5 M-6.5,0 L-2.2,0 M1.5,-5 L1.5,5 M1.5,-5 L6.8,-5 M1.5,0 L5.6,0 M1.5,5 L6.8,5",
          fill: "none", stroke: "#fff", "stroke-width": 1.5, "stroke-linecap": "round", "stroke-linejoin": "round",
        })
      );
    },
    minimax: (g) => {
      [-4.6, 0, 4.6].forEach((cx) => g.appendChild(el("circle", { cx, cy: 0, r: 2.1, fill: "#fff" })));
    },
  };

  function drawFallbackIcon(g, letter) {
    g.appendChild(
      el(
        "text",
        {
          x: 0, y: 3.4, "text-anchor": "middle",
          "font-size": 9.5, "font-weight": 700, fill: "#fff",
          "font-family": "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        },
        letter
      )
    );
  }

  function drawIcon(g, key, fallbackLetter) {
    const drawer = ICON_DRAWERS[(key || "").toLowerCase()];
    if (drawer) drawer(g);
    else drawFallbackIcon(g, (fallbackLetter || "?").slice(0, 1).toUpperCase());
  }

  /* Real brand marks, vendored locally (no third-party requests). Used when
   * available; providers without an asset here fall back to ICON_DRAWERS. */
  const ICON_IMAGES = {
    openai: "assets/logos/openai.png",
    anthropic: "assets/logos/anthropic.jpg",
    gemini: "assets/logos/gemini.png",
    perplexity: "assets/logos/perplexity.png",
    deepseek: "assets/logos/deepseek.svg",
    kimi: "assets/logos/kimi.svg",
    qwen: "assets/logos/qwen.svg",
    glm: "assets/logos/glm.svg",
    ai2: "assets/logos/ai2.jpg",
    minimax: "assets/logos/minimax.svg",
  };

  // Model name / date, a divider, then the plotted metric and a sample-size
  // line. Live points carry a `panels` reviews breakdown (core vs. each
  // rolling cohort) so that line can read "N samples (X core, Y rolling)";
  // points without one (e.g. paper-release baselines) fall back to `note`.
  function tooltipHtml(s, p, opts) {
    const yLabel = opts.yLabel || "Value";
    const metricValue = opts.format(p.raw.value);
    const reasoning = s.reasoning_level ? ` (${s.reasoning_level})` : "";
    // Give the score a gentle visual emphasis without overpowering the model.
    const metricLine =
      `<div style="margin-top:2px;font-size:13px;line-height:1.4">` +
      `<span style="color:#4b5563">${yLabel}:</span> ` +
      `<span style="font-weight:600">${metricValue}</span></div>`;
    let samplesLine = "";
    if (p.raw.panels && Object.keys(p.raw.panels).length) {
      const keys = Object.keys(p.raw.panels);
      if (keys.length === 1) {
        const key = keys[0];
        const n = (p.raw.panels[key] && p.raw.panels[key].reviews) || 0;
        // Rolling points are month-specific cohorts; "rolling" is implied by the
        // filter, so name just the month panel (e.g. "July 2026 panel").
        if (key === "core") {
          samplesLine = `${n} samples (core set)`;
        } else {
          const MONTHS = {
            Jan: "January", Feb: "February", Mar: "March", Apr: "April",
            May: "May", Jun: "June", Jul: "July", Aug: "August",
            Sep: "September", Oct: "October", Nov: "November", Dec: "December",
          };
          const raw = p.raw.label || "";
          const month = raw.replace(
            /^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b/,
            (m) => MONTHS[m] || m
          );
          samplesLine = `${n} samples (${month || "rolling"} panel)`;
        }
      } else {
        const core = (p.raw.panels.core && p.raw.panels.core.reviews) || 0;
        const rolling = keys
          .filter((k) => k !== "core")
          .reduce((sum, k) => sum + ((p.raw.panels[k] && p.raw.panels[k].reviews) || 0), 0);
        samplesLine = `${core + rolling} samples (${core} core · ${rolling} rolling)`;
      }
    } else if (p.raw.note) {
      samplesLine = p.raw.note;
    }
    return (
      `<b style="color:${s.color}">${s.display_name}${reasoning}</b><br>` +
      `<span style="color:#6b7280">${p.raw.label}</span>` +
      `<hr style="margin:5px 0;border:none;border-top:1px solid #e5e7eb;">` +
      metricLine +
      (samplesLine
        ? `<div style="margin-top:2px;color:#6b7280;font-size:11px">${samplesLine}</div>`
        : "")
    );
  }

  /* ---------------------------------------------------------------- *
   * Multi-series line chart over ordered categorical months.
   * series: [{ display_name, color, icon, points: [{ label, value }] }]
   * ---------------------------------------------------------------- */
  function lineChart(mount, series, options) {
    const opts = Object.assign(
      {
        height: 320, yLabel: "", yMax: null, yMin: null,
        format: (v) => v.toFixed(3), badgeRadius: 12,
        endLabels: false, tickCount: 5, labelOrder: null, leadingGapLabel: null, frontier: false,
      },
      options
    );
    mount.innerHTML = "";
    if (!series.length) return;

    const discovered = [];
    series.forEach((s) =>
      s.points.forEach((p) => {
        if (!discovered.includes(p.label)) discovered.push(p.label);
      })
    );
    // Callers with a known chronology (e.g. a fixed "Paper" column ahead of
    // the monthly runs) can pin the x-axis order explicitly; anything not
    // named there still shows up, appended at the end.
    const labels = opts.labelOrder
      ? opts.labelOrder.filter((l) => discovered.includes(l)).concat(
          discovered.filter((l) => !opts.labelOrder.includes(l))
        )
      : discovered;

    const values = series.flatMap((s) =>
      s.points.map((p) => p.value).filter((v) => v !== null && v !== undefined)
    );
    if (!values.length) return;

    const pad = {
      top: opts.badgeRadius + 8,
      right: 8,
      bottom: 44,
      left: 46,
    };
    const IDEAL_COL_STEP = 145;
    const W = Math.max(
      860,
      pad.left + 80 + Math.max(0, labels.length - 1) * IDEAL_COL_STEP + pad.right
    );
    const H = opts.height;
    const innerH = H - pad.top - pad.bottom;

    let lo = opts.yMin !== null ? opts.yMin : Math.min(...values);
    let hi = opts.yMax !== null ? opts.yMax : Math.max(...values);
    if (hi === lo) { hi = lo + 0.1; lo = Math.max(0, lo - 0.1); }
    // Only pad with breathing room around auto-computed bounds; explicit
    // yMin/yMax from the caller are exact axis endpoints.
    if (opts.yMin === null || opts.yMax === null) {
      const span = hi - lo;
      if (opts.yMin === null) lo = Math.max(0, lo - span * 0.12);
      if (opts.yMax === null) hi = hi + span * 0.12;
    }

    // A "leading gap" label (e.g. a fixed paper-release snapshot) isn't part
    // of the monthly timeline — pin it to its own island a bit clear of the
    // y-axis, with real horizontal air before the actual chronological
    // columns start, so it can't read as "the month before" the first run.
    const hasGap =
      opts.leadingGapLabel && labels[0] === opts.leadingGapLabel && labels.length > 1;
    // Live monthly columns land a fixed, compact distance apart (like
    // ticks on a ruler) instead of always stretching to fill however much
    // width happens to be available — so with only two months of data,
    // August lands right next to July instead of clear across the chart
    // at a "last column" edge that has no real meaning yet. Only once
    // enough months accumulate to outgrow the chart does spacing shrink
    // below that ideal step, so everything still fits.
    const colStep = (count, span) => (count > 1 ? Math.min(IDEAL_COL_STEP, span / (count - 1)) : 0);
    let x;
    let leadX, restStart, restCount, colX;
    if (hasGap) {
      leadX = pad.left + 90;
      restStart = leadX + 160;
      restCount = labels.length - 1;
      colX = (rs, i) => rs + (i - 1) * colStep(restCount, W - pad.right - rs);
      x = (i) => (i === 0 ? leadX : colX(restStart, i));
    } else {
      // Inset the first column a bit from the y-axis — sitting exactly on
      // it (as a naive 0..innerW split would, when there's no leading gap
      // column ahead of it) reads as a badge glued to the axis line.
      const marginLeft = 80;
      const plotStart = pad.left + marginLeft;
      x = (i) => plotStart + i * colStep(labels.length, W - pad.right - plotStart);
    }
    const y = (v) => pad.top + innerH - ((v - lo) / (hi - lo)) * innerH;

    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none" });
    if (W > 860) svg.style.width = `${W}px`;

    const R = opts.badgeRadius;
    const clipId = `badge-clip-${Math.random().toString(36).slice(2, 9)}`;
    const defs = el("defs");
    const clipPath = el("clipPath", { id: clipId });
    clipPath.appendChild(el("circle", { cx: 0, cy: 0, r: R }));
    defs.appendChild(clipPath);
    svg.appendChild(defs);

    niceTicks(lo, hi, opts.tickCount).forEach((t) => {
      if (t < lo || t > hi) return;
      svg.appendChild(
        el("line", { class: "grid-line", x1: pad.left, x2: W - pad.right, y1: y(t), y2: y(t) })
      );
      svg.appendChild(
        el("text", { x: pad.left - 10, y: y(t) + 3.5, "text-anchor": "end" }, opts.tick ? opts.tick(t) : t.toFixed(2))
      );
    });

    svg.appendChild(
      el("line", {
        class: "axis-line",
        x1: pad.left, x2: W - pad.right,
        y1: pad.top + innerH, y2: pad.top + innerH,
      })
    );

    // The gap label (e.g. "Preprint Results (early-mid 2026)") is long
    // enough to wrap onto two short lines rather than run wide over its
    // narrow island.
    function wrapLabel(text, maxChars) {
      const words = text.split(" ");
      const lines = [];
      let cur = "";
      words.forEach((w) => {
        if (cur && (cur + " " + w).length > maxChars) {
          lines.push(cur);
          cur = w;
        } else {
          cur = cur ? cur + " " + w : w;
        }
      });
      if (cur) lines.push(cur);
      return lines;
    }

    // A badge's y is always its true value — the axis reading has to stay
    // trustworthy, so nothing ever nudges a dot vertically. When several
    // models land close together in the same month, only their x spreads
    // out a little (a month column carries no continuous meaning beyond
    // "this month"), just enough to keep the badges from fully stacking.
    const seriesPoints = series.map((s) => ({
      s,
      points: labels
        .map((label, i) => {
          const point = s.points.find((p) => p.label === label);
          if (!point || point.value === null || point.value === undefined) return null;
          return { i, x: x(i), y: y(point.value), raw: point };
        })
        .filter(Boolean),
    }));

    // Several models landing on nearly the same score, in the same column,
    // fan out horizontally with a small, controlled overlap. A badge's y
    // is never touched — it has to read correctly against the gridlines —
    // only x moves. Since y is fixed per badge, two badges only need to
    // satisfy: dx^2 + dy^2 >= minDist^2 — i.e. exactly enough x separation
    // to make up for however close they already are in y. This relaxes
    // every close pair repeatedly (pushing along x only) until all clear,
    // so even a tightly packed knot of a dozen badges always fully
    // resolves instead of falling back to overlap once a fixed search
    // budget runs out.
    // A 1.55R center distance leaves a narrow edge overlap while keeping each
    // logo's central mark visible and the month cluster compact.
    const minDist = R * 1.55;
    const columnGroups = new Map();
    seriesPoints.forEach(({ points }) =>
      points.forEach((p) => {
        if (!columnGroups.has(p.i)) columnGroups.set(p.i, []);
        columnGroups.get(p.i).push(p);
      })
    );
    columnGroups.forEach((group) => {
      if (group.length < 2) return;
      // Seed a vanishingly small left/right fan (by y-rank) purely so ties
      // (badges starting at the exact same x) have a direction to resolve
      // in — at 0.001px this has no visible effect on its own.
      group
        .slice()
        .sort((a, b) => a.y - b.y)
        .forEach((p, k) => {
          p.x += (k % 2 === 0 ? -1 : 1) * (Math.floor(k / 2) + 1) * 0.001;
        });
      for (let iter = 0; iter < 120; iter++) {
        let moved = false;
        for (let i = 0; i < group.length; i++) {
          for (let j = i + 1; j < group.length; j++) {
            const a = group[i];
            const b = group[j];
            const dy = b.y - a.y;
            if (Math.abs(dy) >= minDist) continue;
            const targetDx = Math.sqrt(Math.max(0, minDist * minDist - dy * dy));
            const dx = b.x - a.x;
            const absDx = Math.abs(dx);
            if (absDx >= targetDx) continue;
            moved = true;
            const deficit = targetDx - absDx;
            const sign = dx !== 0 ? dx / absDx : 1;
            a.x -= (sign * deficit) / 2;
            b.x += (sign * deficit) / 2;
          }
        }
        if (!moved) break;
      }
    });

    // The leading gap column (e.g. "Preprint Results") can carry a lot more
    // badges than any single monthly column ever will, so its fan-out can
    // grow wide enough to crowd the first live month even though the two
    // columns started with a comfortable gap. Rather than hand-tune that
    // gap for however many models happen to be in the snapshot today, push
    // the whole live timeline over — just enough to clear it, and no more —
    // whenever the packed badges actually get close. Later columns shift
    // less than the first one (columns beyond the chart's filled width
    // don't shift at all yet), so the timeline simply compresses a touch
    // instead of growing unbounded as more months are added.
    if (hasGap && restCount >= 1) {
      const col0 = columnGroups.get(0) || [];
      const col1 = columnGroups.get(1) || [];
      if (col0.length && col1.length) {
        const col0MaxEdge = Math.max(...col0.map((p) => p.x)) + R;
        const col1MinEdge = Math.min(...col1.map((p) => p.x)) - R;
        const minGap = R * 2.7;
        const shortfall = minGap - (col1MinEdge - col0MaxEdge);
        if (shortfall > 0) {
          const oldRestStart = restStart;
          const newRestStart = oldRestStart + shortfall;
          seriesPoints.forEach(({ points }) =>
            points.forEach((p) => {
              if (p.i >= 1) p.x += colX(newRestStart, p.i) - colX(oldRestStart, p.i);
            })
          );
          restStart = newRestStart;
        }
      }
    }

    labels.forEach((label, i) => {
      const isGapLabel = hasGap && i === 0;
      if (isGapLabel) {
        const lines = wrapLabel(label, 18);
        const text = el("text", {
          x: x(i), y: H - 12 - (lines.length - 1) * 12, "text-anchor": "middle",
          "font-style": "italic", class: "axis-gap-label axis-x-label",
        });
        lines.forEach((line, k) => {
          text.appendChild(el("tspan", { x: x(i), dy: k === 0 ? 0 : 12 }, line));
        });
        svg.appendChild(text);
      } else {
        svg.appendChild(
          el("text", { x: x(i), y: H - 12, "text-anchor": "middle", class: "axis-x-label" }, label)
        );
      }
    });

    // The pareto frontier: whichever model scores highest in each column,
    // connected across columns with a dotted line. The line itself still
    // traces every column's best, but per our labeling policy below, only
    // the single best-of-all-time point on it gets a name attached.
    if (opts.frontier) {
      const byColumn = new Map();
      seriesPoints.forEach(({ s, points }) =>
        points.forEach((p) => {
          const cur = byColumn.get(p.i);
          if (!cur || p.raw.value > cur.value) {
            byColumn.set(p.i, { x: p.x, y: p.y, value: p.raw.value, i: p.i });
          }
        })
      );
      const frontierPoints = Array.from(byColumn.values()).sort((a, b) => a.i - b.i);
      if (frontierPoints.length > 1) {
        svg.appendChild(
          el("path", {
            class: "frontier-line",
            d: frontierPoints
              .map((p, k) => `${k ? "L" : "M"}${p.x.toFixed(1)},${p.y.toFixed(1)}`)
              .join(""),
          })
        );
      }
    }

    // Own layer so hovered badges can be re-appended to the top of the
    // paint order without jumping past end-labels that are drawn after.
    const badgesLayer = el("g", { class: "series-badges" });
    svg.appendChild(badgesLayer);

    seriesPoints.forEach(({ s, points }) => {
      if (!points.length) return;

      const imgSrc = ICON_IMAGES[(s.icon || "").toLowerCase()];

      points.forEach((p) => {
        const badge = el("g", {
          class: `series-badge${s.control ? " series-badge--control" : ""}`,
          transform: `translate(${p.x.toFixed(1)},${p.y.toFixed(1)})`,
        });
        const visual = el("g", { class: "series-badge-visual" });
        badge.appendChild(visual);
        visual.appendChild(el("circle", { class: "series-badge-halo", r: R + 2, fill: "#fff" }));
        if (imgSrc) {
          // White backing keeps each brand mark legible regardless of its own
          // palette; a thin ring in the series colour still shows the host.
          visual.appendChild(el("circle", { r: R, fill: "#fff" }));
          visual.appendChild(
            el("image", {
              href: imgSrc,
              x: -R, y: -R, width: R * 2, height: R * 2,
              "clip-path": `url(#${clipId})`,
              preserveAspectRatio: "xMidYMid slice",
            })
          );
          visual.appendChild(
            el("circle", { class: "series-badge-ring", r: R - 0.75, fill: "none", stroke: s.color, "stroke-width": 1.5 })
          );
        } else {
          visual.appendChild(el("circle", { class: "series-badge-dot", r: R, fill: s.color }));
          drawIcon(visual, s.icon, s.provider_label || s.display_name);
        }
        bindTip(badge, tooltipHtml(s, p, opts));
        // SVG has no z-index among siblings — last child paints on top.
        badge.addEventListener("mouseenter", () => badgesLayer.appendChild(badge));
        badgesLayer.appendChild(badge);
      });
    });

    // Labeling policy: naming every model's dot turns the chart into
    // alphabet soup once more than a handful are plotted. Instead, only
    // the best- and worst-scoring model in EACH column (bucket) get a
    // name — not a single global best/worst across the whole chart.
    // Each is, by definition, the extreme badge in its own column, so the
    // space directly above (best) or below (worst) it is always clear —
    // no collision pass needed.
    if (opts.endLabels) {
      const byColumn = new Map();
      seriesPoints.forEach(({ s, points }) =>
        points.forEach((p) => {
          const v = p.raw.value;
          if (v === null || v === undefined) return;
          if (!byColumn.has(p.i)) byColumn.set(p.i, { best: null, worst: null });
          const col = byColumn.get(p.i);
          const entry = { x: p.x, y: p.y, value: v, name: s.display_name, color: s.color };
          if (!col.best || v > col.best.value) col.best = entry;
          if (!col.worst || v < col.worst.value) col.worst = entry;
        })
      );
      byColumn.forEach((col) => {
        if (col.best) {
          svg.appendChild(
            el(
              "text",
              {
                class: "frontier-label", x: col.best.x, y: Math.max(pad.top + 9, col.best.y - R - 9),
                "text-anchor": "middle", fill: col.best.color,
              },
              col.best.name
            )
          );
        }
        if (col.worst && !(col.best && col.worst.x === col.best.x && col.worst.y === col.best.y)) {
          svg.appendChild(
            el(
              "text",
              {
                class: "frontier-label", x: col.worst.x, y: Math.min(H - pad.bottom - 4, col.worst.y + R + 16),
                "text-anchor": "middle", fill: col.worst.color,
              },
              col.worst.name
            )
          );
        }
      });
    }

    mount.appendChild(svg);
  }

  /* ---------------------------------------------------------------- *
   * Stacked column chart. rows: [{ label, segments: [{value,color,name}] }]
   * ---------------------------------------------------------------- */
  function stackedBars(mount, rows, options) {
    const opts = Object.assign({ height: 300, unit: "" }, options);
    mount.innerHTML = "";
    if (!rows.length) return;

    const totals = rows.map((r) => r.segments.reduce((a, s) => a + s.value, 0));
    const max = Math.max(...totals) || 1;

    const pad = { top: 18, right: 12, bottom: 34, left: 42 };
    const W = 620;
    const H = opts.height;
    const innerW = W - pad.left - pad.right;
    const innerH = H - pad.top - pad.bottom;

    const slot = innerW / rows.length;
    const barW = Math.min(46, slot * 0.6);
    const y = (v) => pad.top + innerH - (v / max) * innerH;

    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none" });

    niceTicks(0, max, 4).forEach((t) => {
      svg.appendChild(
        el("line", { class: "grid-line", x1: pad.left, x2: W - pad.right, y1: y(t), y2: y(t) })
      );
      svg.appendChild(
        el("text", { x: pad.left - 9, y: y(t) + 3.5, "text-anchor": "end" }, String(t))
      );
    });

    svg.appendChild(
      el("line", {
        class: "axis-line",
        x1: pad.left, x2: W - pad.right,
        y1: pad.top + innerH, y2: pad.top + innerH,
      })
    );

    rows.forEach((row, i) => {
      const cx = pad.left + slot * i + slot / 2;
      let cursor = 0;
      row.segments.forEach((seg) => {
        if (seg.value <= 0) return;
        const top = y(cursor + seg.value);
        const height = y(cursor) - top;
        const rect = el("rect", {
          class: "bar",
          x: cx - barW / 2, y: top,
          width: barW, height: Math.max(1, height),
          fill: seg.color, rx: 2,
        });
        bindTip(
          rect,
          `<b>${row.label}</b><br>${seg.name}: ${seg.value}${opts.unit}<br>total: ${totals[i]}${opts.unit}`
        );
        svg.appendChild(rect);
        cursor += seg.value;
      });
      svg.appendChild(
        el("text", { x: cx, y: H - 12, "text-anchor": "middle" }, row.label)
      );
    });

    mount.appendChild(svg);
  }

  /* ---------------------------------------------------------------- *
   * Export the live SVG chart as a PNG (white background, inlined
   * styles + logo images) so the download matches what the page shows.
   * ---------------------------------------------------------------- */
  const CHART_EXPORT_CSS = `
    text {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      font-size: 10px;
      fill: #7c8592;
    }
    .axis-line { stroke: #dfe3e8; stroke-width: 1; }
    .grid-line { stroke: #eaedf1; stroke-width: 1; }
    .axis-gap-label { fill: #7c8592; }
    .axis-x-label { font-size: 12.5px; font-weight: 500; fill: #4d5560; }
    .series-badge-halo { opacity: 0.9; }
    .series-badge-dot { stroke: #fff; stroke-width: 1.5; }
    .series-badge--control image { filter: grayscale(1); }
    .frontier-line {
      fill: none;
      stroke: #f59e0b;
      stroke-width: 3.5;
      stroke-dasharray: 2 6;
      stroke-linecap: round;
      opacity: 0.95;
    }
    .frontier-label {
      font-size: 10.5px;
      font-weight: 700;
      paint-order: stroke;
      stroke: #fff;
      stroke-width: 3px;
      stroke-linejoin: round;
    }
  `;

  function fetchAsDataUrl(url) {
    return fetch(url, { cache: "force-cache" })
      .then((response) => {
        if (!response.ok) throw new Error("HTTP " + response.status);
        return response.blob();
      })
      .then(
        (blob) =>
          new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(blob);
          })
      );
  }

  function downloadChartPng(mount, filename) {
    const svg = mount && mount.querySelector("svg");
    if (!svg) return Promise.reject(new Error("No chart to download"));

    const clone = svg.cloneNode(true);
    const vb = (svg.getAttribute("viewBox") || "0 0 860 420").split(/[\s,]+/).map(Number);
    const width = vb[2] || 860;
    const height = vb[3] || 420;
    clone.setAttribute("xmlns", NS);
    clone.setAttribute("width", String(width));
    clone.setAttribute("height", String(height));
    clone.removeAttribute("style");

    const style = el("style");
    style.textContent = CHART_EXPORT_CSS;
    clone.insertBefore(style, clone.firstChild);
    clone.insertBefore(
      el("rect", { x: 0, y: 0, width, height, fill: "#ffffff" }),
      clone.firstChild
    );

    const imageNodes = Array.from(clone.querySelectorAll("image"));
    return Promise.all(
      imageNodes.map((img) => {
        const href = img.getAttribute("href") || img.getAttribute("xlink:href");
        if (!href || href.startsWith("data:")) return Promise.resolve();
        return fetchAsDataUrl(href)
          .then((dataUrl) => {
            img.setAttribute("href", dataUrl);
            img.removeAttribute("xlink:href");
          })
          .catch(() => {
            /* Keep the relative href if a logo fails to inline. */
          });
      })
    ).then(() => {
      const xml = new XMLSerializer().serializeToString(clone);
      const blob = new Blob([xml], { type: "image/svg+xml;charset=utf-8" });
      const svgUrl = URL.createObjectURL(blob);
      const scale = 2;
      return new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => {
          try {
            const canvas = document.createElement("canvas");
            canvas.width = Math.round(width * scale);
            canvas.height = Math.round(height * scale);
            const ctx = canvas.getContext("2d");
            ctx.fillStyle = "#ffffff";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
            URL.revokeObjectURL(svgUrl);
            canvas.toBlob((pngBlob) => {
              if (!pngBlob) {
                reject(new Error("PNG encode failed"));
                return;
              }
              const url = URL.createObjectURL(pngBlob);
              const a = document.createElement("a");
              a.href = url;
              a.download = filename || "sciconbench-progress.png";
              document.body.appendChild(a);
              a.click();
              a.remove();
              URL.revokeObjectURL(url);
              resolve();
            }, "image/png");
          } catch (err) {
            URL.revokeObjectURL(svgUrl);
            reject(err);
          }
        };
        image.onerror = () => {
          URL.revokeObjectURL(svgUrl);
          reject(new Error("Could not rasterize chart"));
        };
        image.src = svgUrl;
      });
    });
  }

  global.Charts = { lineChart, stackedBars, downloadChartPng };
})(window);
