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
      t.innerHTML = html;
      t.style.opacity = "1";
      move(event);
    });
    node.addEventListener("mousemove", move);
    node.addEventListener("mouseleave", () => {
      tooltip().style.opacity = "0";
    });
    function move(event) {
      const t = tooltip();
      const box = t.getBoundingClientRect();
      let x = event.clientX + 14;
      if (x + box.width > window.innerWidth - 12) x = event.clientX - box.width - 14;
      t.style.left = x + "px";
      t.style.top = Math.max(12, event.clientY - box.height - 12) + "px";
    }
  }

  /* ---------------------------------------------------------------- *
   * Multi-series line chart over ordered categorical months.
   * series: [{ display_name, color, points: [{ label, value }] }]
   * ---------------------------------------------------------------- */
  function lineChart(mount, series, options) {
    const opts = Object.assign(
      { height: 320, yLabel: "", yMax: null, yMin: null, format: (v) => v.toFixed(3) },
      options
    );
    mount.innerHTML = "";
    if (!series.length) return;

    const labels = [];
    series.forEach((s) =>
      s.points.forEach((p) => {
        if (!labels.includes(p.label)) labels.push(p.label);
      })
    );

    const values = series.flatMap((s) =>
      s.points.map((p) => p.value).filter((v) => v !== null && v !== undefined)
    );
    if (!values.length) return;

    const pad = { top: 18, right: 20, bottom: 34, left: 46 };
    const W = 860;
    const H = opts.height;
    const innerW = W - pad.left - pad.right;
    const innerH = H - pad.top - pad.bottom;

    let lo = opts.yMin !== null ? opts.yMin : Math.min(...values);
    let hi = opts.yMax !== null ? opts.yMax : Math.max(...values);
    if (hi === lo) { hi = lo + 0.1; lo = Math.max(0, lo - 0.1); }
    const span = hi - lo;
    lo = Math.max(0, lo - span * 0.12);
    hi = hi + span * 0.12;

    const x = (i) =>
      pad.left + (labels.length === 1 ? innerW / 2 : (i / (labels.length - 1)) * innerW);
    const y = (v) => pad.top + innerH - ((v - lo) / (hi - lo)) * innerH;

    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none" });

    niceTicks(lo, hi, 5).forEach((t) => {
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

    labels.forEach((label, i) => {
      svg.appendChild(
        el("text", { x: x(i), y: H - 12, "text-anchor": "middle" }, label)
      );
    });

    series.forEach((s) => {
      const points = labels
        .map((label, i) => {
          const point = s.points.find((p) => p.label === label);
          if (!point || point.value === null || point.value === undefined) return null;
          return { i, x: x(i), y: y(point.value), raw: point };
        })
        .filter(Boolean);
      if (!points.length) return;

      if (points.length > 1) {
        svg.appendChild(
          el("path", {
            class: "series-line",
            stroke: s.color,
            d: points.map((p, k) => `${k ? "L" : "M"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(""),
          })
        );
      }

      points.forEach((p) => {
        const dot = el("circle", {
          class: "series-dot",
          cx: p.x, cy: p.y, r: points.length === 1 ? 5 : 4,
          fill: s.color,
        });
        bindTip(
          dot,
          `<b style="color:${s.color}">${s.display_name}</b><br>${p.raw.label} · ${
            opts.yLabel || "value"
          } ${opts.format(p.raw.value)}${p.raw.note ? "<br>" + p.raw.note : ""}`
        );
        svg.appendChild(dot);
      });
    });

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

  global.Charts = { lineChart, stackedBars };
})(window);
