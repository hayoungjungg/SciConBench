/* SciConBench dashboard — renders public/data/dashboard.json into the page. */

(function () {
  "use strict";

  // Proxy that no-ops when an id is missing. A stale cached HTML/JS mismatch
  // (e.g. after removing the nav status pill) used to throw on null and abort
  // the whole render, which left the effort table and other sections empty.
  const $ = (id) => {
    const el = document.getElementById(id);
    if (el) return el;
    return new Proxy(
      {},
      {
        get(_t, prop) {
          if (prop === "style") return {};
          if (typeof prop === "string") return () => {};
          return undefined;
        },
        set() {
          return true;
        },
      }
    );
  };

  const fmtInt = (n) =>
    n === null || n === undefined ? "—" : Math.round(n).toLocaleString("en-US");

  const fmtPct = (v) =>
    v === null || v === undefined ? null : (v * 100).toFixed(1);

  const fmtCompact = (n) => {
    if (n === null || n === undefined) return "—";
    if (n >= 1000) return (n / 1000).toFixed(n >= 10000 ? 0 : 1) + "k";
    return Math.round(n).toLocaleString("en-US");
  };

  const fmtDate = (iso) => {
    if (!iso) return "—";
    const d = new Date(iso.length === 10 ? iso + "T12:00:00Z" : iso);
    if (isNaN(d)) return iso;
    return d.toLocaleDateString("en-US", {
      year: "numeric", month: "short", day: "numeric", timeZone: "UTC",
    });
  };

  const escape = (s) =>
    String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));

  const linkHtml = (text, href) =>
    href.startsWith("#")
      ? `<a href="${href}">${text}</a>`
      : `<a href="${href}" target="_blank" rel="noopener">${text}</a>`;

  const inlineMd = (s) =>
    escape(s)
      .replace(/\+\+([^+]+)\+\+/g, "<u>$1</u>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/\*([^*]+)\*/g, "<em>$1</em>")
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      // Double brackets preserve visible scholarly citation brackets:
      // [[1]](url) renders the entire "[1]" as a superscript link, matching
      // the numbered citations in the page introduction. Adjacent citations
      // such as [[1]](url), [[2]](url) collapse into one "[1, 2]" superscript.
      .replace(
        /\[\[(\d+)\]\]\(([^)\s]+)\)(?:, \[\[(\d+)\]\]\(([^)\s]+)\))+/g,
        (match) => {
          const links = [...match.matchAll(/\[\[(\d+)\]\]\(([^)\s]+)\)/g)]
            .map(([, n, href]) => linkHtml(n, href))
            .join(", ");
          return `<sup class="intro-citations">[${links}]</sup>`;
        }
      )
      .replace(/\[\[([^\]]+)\]\]\(([^)\s]+)\)/g, (_, text, href) =>
        `<sup class="intro-citations">${linkHtml(`[${text}]`, href)}</sup>`
      )
      .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, text, href) => linkHtml(text, href));

  // Blank lines separate paragraphs; runs of lines starting with "- " become
  // a bullet list.
  function prose(text) {
    const out = [];
    let para = [];
    let bullets = [];
    const flush = () => {
      if (para.length) {
        out.push(`<p>${inlineMd(para.join(" "))}</p>`);
        para = [];
      }
      if (bullets.length) {
        out.push(
          `<ul class="prose-list">${bullets
            .map((b) => `<li>${inlineMd(b)}</li>`)
            .join("")}</ul>`
        );
        bullets = [];
      }
    };

    String(text)
      .split("\n")
      .forEach((raw) => {
        const line = raw.trim();
        if (!line) return flush();
        if (line.startsWith("- ")) {
          if (para.length) flush();
          bullets.push(line.slice(2));
        } else {
          if (bullets.length) flush();
          para.push(line);
        }
      });

    flush();
    return out.join("");
  }

  const REASONING_LABELS = new Set([
    "max", "high", "xhigh", "reasoning", "thinking", "extended thinking", "fixed reasoning",
  ]);

  function modelNameParts(displayName) {
    const match = String(displayName).match(/^(.*) \(([^()]+)\)$/);
    if (!match || !REASONING_LABELS.has(match[2].toLowerCase())) {
      return { name: String(displayName), reasoning: null };
    }
    return { name: match[1], reasoning: match[2] };
  }

  function modelNameHtml(displayName) {
    const { name, reasoning } = modelNameParts(displayName);
    if (!reasoning) return `<span class="model-name">${escape(name)}</span>`;
    return (
      `<span class="model-name model-name--reasoning" tabindex="0">${escape(name)}` +
      `<span class="model-reasoning"> (${escape(reasoning)})</span></span>`
    );
  }

  // Brand marks for the hero action buttons — each is the real logo art
  // (arXiv's own logomark, GitHub's Octocat badge, HuggingFace's emoji
  // mark) rather than a flattened single-color redraw.
  const ICONS = {
    arxiv: '<img src="assets/icon-arxiv.png" alt="" />',
    huggingface: '<img src="assets/icon-huggingface.png" alt="" />',
    github: '<img src="assets/icon-github.png" alt="" />',
    blog: '<img src="assets/icon-blog.png" alt="" />',
  };

  // Real provider marks, vendored with the site so model tables never need
  // third-party image requests. Unknown future providers get a neutral
  // monogram rather than an arbitrary series-colour dot.
  const MODEL_LOGOS = {
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

  function modelLogoHtml(row) {
    const key = String(row.icon || row.family || "").toLowerCase();
    const src = MODEL_LOGOS[key];
    if (src) {
      return (
        `<span class="model-logo"><img src="${src}" alt="" ` +
        `title="${escape(row.provider_label)}" loading="lazy"></span>`
      );
    }
    const initial = String(row.provider_label || row.display_name || "?").trim().slice(0, 1);
    return (
      `<span class="model-logo model-logo--fallback" ` +
      `title="${escape(row.provider_label || "Model provider")}">${escape(initial)}</span>`
    );
  }

  /* ------------------------------------------------------------------ *
   * hero
   * ------------------------------------------------------------------ */

  function renderHero(data) {
    const site = data.site || {};

    $("hero-tagline").textContent = site.tagline || "";

    const links = site.links || {};
    const actions = [
      links.paper && { href: links.paper, label: "Paper", icon: "arxiv" },
      links.dataset && {
        href: links.dataset, label: "Dataset", icon: "huggingface",
      },
      links.code && { href: links.code, label: "GitHub", icon: "github" },
      links.blog && { href: links.blog, label: "Blog", icon: "blog" },
    ].filter(Boolean);

    $("hero-actions").innerHTML = actions
      .map((a) => {
        const icon = a.icon ? `<span class="btn-icon">${ICONS[a.icon]}</span>` : "";
        const sub = a.sub ? `<span class="btn-sub">${escape(a.sub)}</span>` : "";
        return `<a class="btn${a.primary ? " btn--primary" : ""}" href="${escape(a.href)}"${
          a.href.startsWith("#") ? "" : ' target="_blank" rel="noopener"'
        }>${icon}${escape(a.label)}${sub}</a>`;
      })
      .join("");
  }

  /* ------------------------------------------------------------------ *
   * leaderboard
   * ------------------------------------------------------------------ */

  let boardState = { metric: "f1", panel: "all", includePaper: false, base: [], paper: [] };

  // The exported leaderboard rows carry a blended precision/recall/f1 plus a
  // per-panel breakdown (`row.panels[key]`). Swapping `boardState.panel`
  // re-projects each row onto that slice — "all" leaves the blended score.
  // Paper-release baselines (a one-off snapshot, not a monthly panel) only
  // make sense alongside the blended view, so they're appended there.
  function boardRows() {
    let rows = boardState.panel === "all" ? boardState.base : boardState.base.map((row) => {
      const slice = (row.panels || {})[boardState.panel];
      return Object.assign({}, row, {
        precision: slice ? slice.precision : null,
        recall: slice ? slice.recall : null,
        f1: slice ? slice.f1 : null,
        reviews: slice ? slice.reviews : row.reviews,
        run_month_label: slice ? slice.run_month_label : row.run_month_label,
      });
    });
    if (boardState.includePaper && boardState.panel === "all") {
      rows = rows.concat(
        boardState.paper.map((b) =>
          Object.assign({}, b, { run_month_label: "Preprint", reviews: null, is_paper: true })
        )
      );
    }
    return rows;
  }

  function renderBoard() {
    const metric = boardState.metric;
    const rows = boardRows().slice().sort((a, b) => {
      const av = a[metric], bv = b[metric];
      if (av === null && bv === null) return a.display_name.localeCompare(b.display_name);
      if (av === null) return 1;
      if (bv === null) return -1;
      return bv - av;
    });

    const graded = rows.filter((r) => r[metric] !== null);
    const best = graded.length ? graded[0][metric] : null;

    document.querySelectorAll("#board th[data-metric]").forEach((th) => {
      th.classList.toggle("is-sorted", th.dataset.metric === metric);
    });
    document.querySelectorAll("#metric-toggle button").forEach((b) => {
      b.classList.toggle("is-active", b.dataset.metric === metric);
    });

    if (!rows.length) {
      $("board-body").innerHTML =
        '<tr><td colspan="5" class="empty"><strong>No evaluations recorded yet</strong>' +
        "The first monthly run will populate this table.</td></tr>";
      return;
    }

    // Use the metric's absolute 0–1 scale: 0 is red, 0.5 is amber, and 1 is
    // blue. Colours therefore retain the same meaning across panels and runs.
    const performanceColor = (value) => {
      const position = Math.max(0, Math.min(1, value));
      const stops = [
        [194, 65, 59],
        [217, 154, 34],
        [37, 99, 235],
      ];
      const segment = position <= 0.5 ? 0 : 1;
      const amount = position <= 0.5 ? position * 2 : (position - 0.5) * 2;
      const rgb = stops[segment].map((channel, index) =>
        Math.round(channel + (stops[segment + 1][index] - channel) * amount)
      );
      return `rgb(${rgb.join(",")})`;
    };

    const cell = (value, isBest) => {
      const pct = fmtPct(value);
      if (pct === null) return '<td class="col-num"><span class="tag tag--pending">pending</span></td>';
      const color = performanceColor(value);
      return (
        `<td class="col-num"><span class="score" style="--score-color:${color}">` +
        `<span class="score-value${isBest ? " score-value--lead" : ""}">${pct}</span>` +
        `<span class="score-bar"><i style="width:${(value * 100).toFixed(1)}%"></i></span>` +
        `</span></td>`
      );
    };

    let previousDisplayedScore = null;
    let sharedRank = 0;
    const bestDisplayedScore = fmtPct(best);

    $("board-body").innerHTML = rows
      .map((row, index) => {
        const displayedScore = fmtPct(row[metric]);
        let rank = "—";
        if (displayedScore !== null) {
          // Rank what readers can see: scores equal at the displayed precision
          // share a competition rank (1, 1, 3), including a tied lead.
          if (displayedScore !== previousDisplayedScore) {
            sharedRank = index + 1;
            previousDisplayedScore = displayedScore;
          }
          rank = sharedRank;
        }
        const isBest =
          displayedScore !== null &&
          bestDisplayedScore !== null &&
          displayedScore === bestDisplayedScore;
        return (
          `<tr>` +
          `<td class="col-rank">${rank}</td>` +
          `<td class="col-model"><span class="model-cell">` +
          modelLogoHtml(row) +
          `<span><span class="model-label">${modelNameHtml(row.display_name)}</span>` +
          `<span class="model-provider">${escape(row.provider_label)}` +
          `${row.is_paper ? '<span class="tag tag--paper">Preprint</span>' : ""}</span></span></span></td>` +
          cell(row.precision, metric === "precision" && isBest) +
          cell(row.recall, metric === "recall" && isBest) +
          cell(row.f1, metric === "f1" && isBest) +
          `</tr>`
        );
      })
      .join("");
  }

  function populatePanelSelect(data) {
    const select = document.getElementById("panel-select");
    if (!select) return;
    const views = [{ key: "all", label: "All reviews" }, ...(data.panel_views || [])];
    select.innerHTML = views
      .map((v) => `<option value="${escape(v.key)}">${escape(v.label)}</option>`)
      .join("");
    select.value = boardState.panel;
  }

  function renderLeaderboardMeta(data) {
    const summary = data.summary;
    const pending = summary.total_responses - summary.graded_responses;

    const notes = [];
    if (pending > 0) {
      notes.push(
        `${fmtInt(pending)} of ${fmtInt(summary.total_responses)} model responses are still ` +
          `awaiting the factual precision and recall judges; those rows show as <em>pending</em>.`
      );
    }
    const preprintNotice = $("board-preprint-notice");
    if (preprintNotice) preprintNotice.hidden = !boardState.includePaper;
    $("board-footnote").innerHTML = notes.join(" ");
  }

  /* ------------------------------------------------------------------ *
   * trend
   * ------------------------------------------------------------------ */

  let trendState = { metric: "f1", families: new Set(), panel: "all" };
  const METRIC_LABEL = { f1: "Factual F1", precision: "Factual Precision", recall: "Factual Recall" };
  const CONTROL_MODELS = new Set([
    "DeepSeek-V4-Flash-0731",
    "qwen/qwen3.8-27b",
    "minimax/minimax-m3",
  ]);
  const CONTROL_MARK = "\u2020";
  const CONTROL_COLOR = "#6b7280";
  // The superscript "¹" ties this axis label to the matching footnote below
  // the chart — it's baked into the label itself so it travels with it
  // wherever the string is used (axis, tooltip date line, etc.).
  const PAPER_LABEL = "Preprint Results (early-mid 2026)¹";

  // "All models" plus one option per model family (lab) — e.g. selecting
  // "OpenAI" shows every OpenAI model together (o3 Deep Research, GPT-5.1,
  // GPT-5.6 Sol, ...), not one at a time. Family is decided by the model's
  // *name* server-side (export_data.py's resolve_family), not by which API
  // happened to host it, so this list needs no maintenance as new models
  // land — a future "gpt-6" is already "OpenAI" the moment it has a score.
  function trendFamilyOptions(data) {
    const byKey = new Map();
    (data.series || []).forEach((s) => byKey.set(s.family, s.provider_label));
    (data.paper_baselines || []).forEach((b) => byKey.set(b.family, b.provider_label));
    return Array.from(byKey.entries())
      .map(([key, label]) => ({ key, label }))
      .sort((a, b) => a.label.localeCompare(b.label));
  }

  function updateTrendFamilySummary(data) {
    const summary = document.getElementById("trend-family-summary");
    if (!summary) return;
    const families = trendFamilyOptions(data);
    const selected = families.filter((family) => trendState.families.has(family.key));
    if (selected.length === families.length) summary.textContent = "All models";
    else if (!selected.length) summary.textContent = "No models";
    else if (selected.length === 1) summary.textContent = selected[0].label;
    else summary.textContent = `${selected.length} families`;
  }

  function populateTrendControls(data) {
    const familyOptions = document.getElementById("trend-family-options");
    if (familyOptions) {
      const families = trendFamilyOptions(data);
      trendState.families = new Set(families.map((family) => family.key));
      familyOptions.innerHTML =
        '<label class="family-picker-option family-picker-option--all">' +
        '<input type="checkbox" value="all" checked> All models</label>' +
        families
          .map(
            (family) =>
              `<label class="family-picker-option"><input type="checkbox" ` +
              `value="${escape(family.key)}" checked> ${escape(family.label)}</label>`
          )
          .join("");
      updateTrendFamilySummary(data);
    }
    const panelToggle = document.getElementById("trend-panel-toggle");
    if (panelToggle) {
      const views = [{ key: "all", label: "All reviews" }, ...(data.panel_views || [])];
      // "All reviews" is the only label long enough to need shortening now
      // that rolling cohorts are just "Rolling panel" — no month name to trim.
      const short = (v) => (v.key === "all" ? "All" : v.label);
      panelToggle.innerHTML = views
        .map(
          (v) =>
            `<button data-panel="${escape(v.key)}" title="${escape(v.label)}" class="${
              v.key === trendState.panel ? "is-active" : ""
            }">${escape(short(v))}</button>`
        )
        .join("");
    }
  }

  function trendPointsForPanel(series, panel, metric) {
    return series.points.map((point) => {
      const slice = panel === "all" ? null : (point.panels || {})[panel];
      const value = panel === "all" ? point[metric] : slice ? slice[metric] : null;
      const visiblePanels = {};
      if (panel === "all") {
        const cumulativePanels = point.all_panels || point.panels || {};
        ["core", "rolling"].forEach((key) => {
          if (cumulativePanels[key]) visiblePanels[key] = cumulativePanels[key];
        });
      } else if (slice) {
        // Panel filter: only the active slice, so the hover count matches
        // the plotted score (e.g. 20 rolling, not 119+20).
        visiblePanels[panel] = slice;
      }
      const reviews =
        panel === "all"
          ? point.reviews
          : slice && slice.reviews != null
            ? slice.reviews
            : null;
      return {
        label: point.label,
        value,
        panels: visiblePanels,
        note: reviews != null ? `${reviews} samples` : undefined,
      };
    });
  }

  function renderTrend(data) {
    const mount = $("trend-chart");
    const metric = trendState.metric;
    const panel = trendState.panel;

    const liveSeries = (data.series || []).map((s) => {
      const isControl = CONTROL_MODELS.has(s.model);
      const { name, reasoning } = modelNameParts(s.display_name);
      return {
        display_name: `${name}${isControl ? CONTROL_MARK : ""}`,
        reasoning_level: reasoning,
        family: s.family,
        provider_label: isControl ? `${s.provider_label} · control model` : s.provider_label,
        color: isControl ? CONTROL_COLOR : s.color,
        icon: s.icon,
        control: isControl,
        points: trendPointsForPanel(s, panel, metric),
      };
    });

    // Paper-release baselines are a fixed one-off snapshot (no core/rolling
    // breakdown), so they only render under "All reviews" — and always as
    // their own left-most column, never pinned to a live run month.
    const paperSeries =
      panel === "all"
        ? (data.paper_baselines || []).map((b) => {
            const { name, reasoning } = modelNameParts(b.display_name);
            return {
              display_name: name,
              reasoning_level: reasoning,
              family: b.family,
              provider_label: b.provider_label,
              color: b.color,
              icon: b.icon,
              points: [
                {
                  label: PAPER_LABEL,
                  value: b[metric],
                },
              ],
            };
          })
        : [];

    // Chronological month labels, taken from the live series in the order
    // they already appear (build_series sorts each model's own points by
    // run month); "Paper" is pinned to the front regardless.
    const monthLabels = [];
    liveSeries.forEach((s) =>
      s.points.forEach((p) => {
        if (!monthLabels.includes(p.label)) monthLabels.push(p.label);
      })
    );

    const series = paperSeries
      .concat(liveSeries)
      .filter((s) => trendState.families.has(s.family))
      .filter((s) => s.points.some((p) => p.value !== null && p.value !== undefined));
    const showsPaper = series.some((s) =>
      s.points.some(
        (p) => p.label === PAPER_LABEL && p.value !== null && p.value !== undefined
      )
    );

    document.querySelectorAll("#trend-metric-toggle button").forEach((b) => {
      b.classList.toggle("is-active", b.dataset.metric === metric);
    });
    document.querySelectorAll("#trend-panel-toggle button").forEach((b) => {
      b.classList.toggle("is-active", b.dataset.panel === panel);
    });

    const notes = [];
    if (showsPaper) {
      notes.push(
        "\u00b9 Preprint results use a fixed N=268 subset published after the latest model knowledge " +
        'cutoff (Gemini 3 Pro, Jan 31, 2025); later models were evaluated on the same subset for comparability. <span class="axis-break-key" role="img" ' +
        'aria-label="Axis break"></span> marks the longer gap to July 2026; subsequent intervals are monthly.'
      );
    }
    notes.push(
      escape(`${CONTROL_MARK} Control open-weight models are shown in gray: DeepSeek-V4-Flash, Qwen3.8 27B, ` +
        "and MiniMax M3. These fixed model versions remain unchanged across monthly runs, " +
        "providing a stable baseline for gauging frontier-model progress and changes in " +
        "benchmark difficulty.")
    );
    $("trend-footnote").innerHTML = notes.join("<br>");

    if (!series.length) {
      mount.innerHTML =
        '<div class="empty"><strong>No data for this selection</strong>' +
        "Try a different model or panel filter.</div>";
      const downloadBtn = document.getElementById("trend-download");
      if (downloadBtn) downloadBtn.disabled = true;
      return;
    }

    Charts.lineChart(mount, series, {
      height: 420,
      yLabel: METRIC_LABEL[metric] || metric,
      format: (v) => (v * 100).toFixed(1),
      tick: (t) => (t * 100).toFixed(0),
      yMin: 0,
      yMax: 0.6,
      tickCount: 7,
      endLabels: true,
      frontier: true,
      labelOrder: [PAPER_LABEL].concat(monthLabels),
      leadingGapLabel: PAPER_LABEL,
    });
    const downloadBtn = document.getElementById("trend-download");
    if (downloadBtn) downloadBtn.disabled = false;
  }

  /* ------------------------------------------------------------------ *
   * dataset
   * ------------------------------------------------------------------ */

  function renderDataset(data) {
    const dataset = data.dataset;

    const rows = (dataset.growth || []).map((g) => {
      const existingRolling = Math.max(0, g.rolling - g.added);
      return {
        label: g.label,
        segments: [
          { name: "Core Set", value: g.core, color: "#5b8def" },
          { name: "Existing Rolling Panels", value: existingRolling, color: "#8bdcc4" },
          { name: "New This Month", value: g.added, color: "#159a70" },
        ],
      };
    });

    if (rows.length) {
      Charts.stackedBars($("growth-chart"), rows, { unit: " reviews" });
    } else {
      $("growth-chart").innerHTML =
        '<div class="empty"><strong>No cohorts yet</strong>Monthly cohorts appear after the first run.</div>';
    }

    const facts = [
      ["Total reviews", fmtInt(dataset.total_reviews)],
      ["Core panel (fixed)", fmtInt(dataset.core_reviews)],
      ["Rolling cohorts", fmtInt(dataset.rolling_reviews)],
      [
        "Publication window",
        `${fmtDate(dataset.publication_range.from)} – ${fmtDate(dataset.publication_range.to)}`,
      ],
    ];

    $("dataset-facts").innerHTML = facts
      .map(
        ([k, v]) =>
          `<div><span class="k">${escape(k)}</span><span class="v">${escape(v)}</span></div>`
      )
      .join("");
  }

  /* ------------------------------------------------------------------ *
   * fact diagnostics + effort
   * ------------------------------------------------------------------ */

  function renderFactDiagnostics(data) {
    const rows = (data.leaderboard || [])
      .filter((row) => row.fact_stats)
      .slice()
      .sort(
        (a, b) =>
          (b.fact_stats.responses.with_contradiction_rate || 0) -
          (a.fact_stats.responses.with_contradiction_rate || 0)
      );
    if (!rows.length) {
      $("response-diagnostics-body").innerHTML =
        '<tr><td colspan="2" class="empty">No evaluated responses recorded yet.</td></tr>';
      return;
    }

    const modelCell = (row) =>
      `<td class="col-model"><span class="model-cell">` +
      modelLogoHtml(row) +
      `${modelNameHtml(row.display_name)}</span></td>`;

    const issueCell = (count, rate, evaluated) => {
      const pct = fmtPct(rate);
      return `<td class="col-num">${pct === null ? "—" : `${pct}% (${fmtInt(count)}/${fmtInt(evaluated)})`}</td>`;
    };
    $("response-diagnostics-body").innerHTML = rows
      .map((row) => {
        const responses = row.fact_stats.responses;
        return (
          `<tr>${modelCell(row)}` +
          issueCell(
            responses.with_contradiction,
            responses.with_contradiction_rate,
            responses.evaluated
          ) +
          `</tr>`
        );
      })
      .join("");
  }

  let effortState = { sort: "tool_calls", direction: "desc" };

  function renderEffort(data) {
    const rows = data.leaderboard || [];
    if (!rows.length) {
      $("effort-body").innerHTML =
        '<tr><td colspan="5" class="empty">No evaluations recorded yet.</td></tr>';
      return;
    }

    const field = effortState.sort === "agent_turns" ? "avg_iterations" : "avg_tool_calls";
    const direction = effortState.direction === "asc" ? 1 : -1;
    const sorted = rows.slice().sort((a, b) => {
      const av = a[field], bv = b[field];
      if (av === null || av === undefined) return 1;
      if (bv === null || bv === undefined) return -1;
      return direction * (av - bv);
    });

    document.querySelectorAll("[data-effort-sort]").forEach((button) => {
      const active = button.dataset.effortSort === effortState.sort;
      const th = button.closest("th");
      th.classList.toggle("is-sorted", active);
      th.setAttribute(
        "aria-sort",
        active ? (effortState.direction === "asc" ? "ascending" : "descending") : "none"
      );
    });

    const toolName = (raw) =>
      ({
        serper_google_webpage_search: "google_search",
        se_google_webpage_search: "google_search",
        fetch_webpage_content: "web_browse",
        jina_fetch_webpage_content: "web_browse",
        semantic_scholar_snippet_search: "paper_search",
        semantic_scholarly_snippet_search: "paper_search",
        semantic_scholr_snippet_search: "paper_search",
        senantic_scholar_snippet_search: "paper_search",
      })[raw] || raw.replace(/_/g, " ");

    $("effort-body").innerHTML = sorted
      .map((row, index) => {
        const usage = new Map();
        (row.tool_usage || []).forEach(({ tool, count }) => {
          const label = toolName(tool);
          usage.set(label, (usage.get(label) || 0) + count);
        });
        const top = Array.from(usage.entries()).sort((a, b) => b[1] - a[1])[0];
        const totalCalls = Array.from(usage.values()).reduce((sum, count) => sum + count, 0);
        const distribution = ["google_search", "web_browse", "paper_search"]
          .map((label) => {
            const count = usage.get(label) || 0;
            const pct = totalCalls ? ((count / totalCalls) * 100).toFixed(1) : "0.0";
            return (
              `<span class="tool-share${top && label === top[0] ? " tool-share--top" : ""}"><code>${label}</code>` +
              `<span>${pct}%</span></span>`
            );
          })
          .join("");
        return (
          `<tr>` +
          `<td class="col-rank">${index + 1}</td>` +
          `<td class="col-model"><span class="model-cell">` +
          modelLogoHtml(row) +
          `${modelNameHtml(row.display_name)}</span></td>` +
          `<td class="col-num">${row.avg_tool_calls === null ? "—" : row.avg_tool_calls.toFixed(1)}</td>` +
          `<td class="col-num">${row.avg_iterations === null ? "—" : row.avg_iterations.toFixed(1)}</td>` +
          `<td class="col-tools"><span class="tool-distribution">${distribution}</span></td>` +
          `</tr>`
        );
      })
      .join("");
  }

  /* ------------------------------------------------------------------ *
   * static content from site.config.json
   * ------------------------------------------------------------------ */

  let questionExamples = [];
  let questionExampleStart = 0;

  function questionExamplePageSize() {
    return window.innerWidth <= 740 ? 1 : 2;
  }

  function renderQuestionExamples() {
    const section = document.getElementById("question-examples");
    const grid = document.getElementById("question-examples-grid");
    if (!section || !grid) return;

    if (!questionExamples.length) {
      section.hidden = true;
      return;
    }

    section.hidden = false;
    const pageSize = questionExamplePageSize();
    const maxStart = Math.max(0, questionExamples.length - pageSize);
    questionExampleStart = Math.min(questionExampleStart, maxStart);
    const visible = questionExamples.slice(
      questionExampleStart,
      questionExampleStart + pageSize
    );

    grid.innerHTML = visible
      .map(
        (item) =>
          `<article class="question-example-card">` +
          `<div class="question-example-field"><span aria-hidden="true">✚</span>${escape(
            item.field
          )}</div>` +
          `<div class="question-example-body">` +
          `<h4>Question</h4><p>${escape(item.question)}</p>` +
          `</div>` +
          `<footer><span>Cochrane Database of Systematic Reviews</span>` +
          `<span class="question-example-source">` +
          `<a href="https://doi.org/${escape(item.doi)}" target="_blank" rel="noopener">${escape(
            item.doi
          )}</a></span></footer>` +
          `</article>`
      )
      .join("");

    const first = questionExampleStart + 1;
    const last = questionExampleStart + visible.length;
    $("question-examples-counter").textContent =
      `Examples ${first}\u2013${last} of ${questionExamples.length}`;
    $("question-examples-prev").disabled = questionExampleStart === 0;
    $("question-examples-next").disabled = questionExampleStart === maxStart;
  }

  function renderContent(data) {
    const site = data.site || {};

    $("news-list").innerHTML = (site.news || [])
      .map(
        (n) =>
          `<li><img class="news-icon" src="assets/logo.png" alt="" aria-hidden="true" /><time>[${escape(
            fmtDate(n.date)
          )}]:</time> <span>${escape(n.text).replace(
            /\*([^*]+)\*/g,
            "<em>$1</em>"
          )}</span></li>`
      )
      .join("");

    $("faq").innerHTML = (site.faq || [])
      .map((f) => {
        const id = f.id ? ` id="faq-${escape(f.id)}"` : "";
        return (
          `<details${id}><summary>${escape(f.question)}</summary>` +
          `<div class="faq-body">${prose(f.answer)}</div></details>`
        );
      })
      .join("");

    if (site.headline && site.headline.text) {
      $("headline").innerHTML =
        escape(site.headline.text) +
        (site.headline.source
          ? `<cite>${
              (site.links || {}).paper
                ? `<a href="${escape(site.links.paper)}" target="_blank" rel="noopener">${escape(
                    site.headline.source
                  )}</a>`
                : escape(site.headline.source)
            }</cite>`
          : "");
      $("headline").hidden = false;
    }

    const intro = Array.isArray(site.introduction)
      ? site.introduction
      : site.introduction
      ? [site.introduction]
      : [];
    $("introduction-text").innerHTML = intro
      .map((p) => `<p>${inlineMd(p)}</p>`)
      .join("");

    questionExamples = site.question_examples || [];
    questionExampleStart = 0;
    renderQuestionExamples();

    $("citation").textContent = site.citation || "";

    $("contact").textContent =
      "\u2709\ufe0f For any inquiries, questions, or feedback, please contact us at " +
      "hayoung [at] cs [dot] princeton [dot] edu";

    const ack = Array.isArray(site.acknowledgements)
      ? site.acknowledgements
      : site.acknowledgements
      ? [site.acknowledgements]
      : [];
    $("ack-text").innerHTML = ack
      .map((p) => `<p class="team-toggle-legend">${escape(p)}</p>`)
      .join("");

    // FAQ deep links open their accordion before scrolling to it. This also
    // handles direct visits to a saved FAQ URL after dynamic content renders.
    const openFaqTarget = (hash, scroll) => {
      if (!hash || !hash.startsWith("#faq-")) return;
      const target = document.getElementById(hash.slice(1));
      if (!target || target.tagName !== "DETAILS") return;
      target.open = true;
      if (scroll) requestAnimationFrame(() => target.scrollIntoView({ block: "start" }));
    };
    document.querySelectorAll('a[href^="#faq-"]').forEach((link) => {
      link.addEventListener("click", () => openFaqTarget(link.getAttribute("href"), false));
    });
    openFaqTarget(window.location.hash, true);
  }

  /* ------------------------------------------------------------------ *
   * side nav scrollspy
   * ------------------------------------------------------------------ */

  // Runs independently of the dashboard fetch below: every section it
  // targets is static markup, not generated from data.
  function setupScrollSpy() {
    const links = Array.from(document.querySelectorAll(".side-nav a[data-section]"));
    const sections = links
      .map((link) => ({ link, el: document.getElementById(link.dataset.section) }))
      .filter((s) => s.el);
    if (!sections.length) return;

    const setActive = (id) => {
      links.forEach((link) => link.classList.toggle("is-active", link.dataset.section === id));
    };

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible.length) setActive(visible[0].target.id);
      },
      { rootMargin: "-15% 0px -70% 0px" }
    );
    sections.forEach((s) => observer.observe(s.el));
  }

  // The rail stays out of the way while the hero/overview is in view, and
  // slides in once the reader has scrolled past it.
  function setupSideNavReveal() {
    const nav = document.querySelector(".side-nav");
    const overview = document.getElementById("overview");
    if (!nav || !overview) return;

    const observer = new IntersectionObserver(
      ([entry]) => nav.classList.toggle("is-visible", !entry.isIntersecting),
      { threshold: 0 }
    );
    observer.observe(overview);
  }

  function setupThemeToggle() {
    const button = document.getElementById("theme-toggle");
    if (!button) return;
    const root = document.documentElement;
    const systemTheme = window.matchMedia("(prefers-color-scheme: dark)");

    const applyTheme = (theme) => {
      const dark = theme === "dark";
      root.dataset.theme = dark ? "dark" : "light";
      button.setAttribute("aria-pressed", String(dark));
      button.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
      button.title = dark ? "Switch to light mode" : "Switch to dark mode";
    };

    applyTheme(root.dataset.theme || (systemTheme.matches ? "dark" : "light"));
    button.addEventListener("click", () => {
      const next = root.dataset.theme === "dark" ? "light" : "dark";
      applyTheme(next);
      try {
        localStorage.setItem("sciconbench-theme", next);
      } catch (_) {
        /* The selected theme still applies when storage is unavailable. */
      }
    });

    const followSystemTheme = (event) => {
      try {
        if (localStorage.getItem("sciconbench-theme")) return;
      } catch (_) {
        /* Follow the system preference when storage is unavailable. */
      }
      applyTheme(event.matches ? "dark" : "light");
    };
    if (typeof systemTheme.addEventListener === "function") {
      systemTheme.addEventListener("change", followSystemTheme);
    } else if (typeof systemTheme.addListener === "function") {
      systemTheme.addListener(followSystemTheme);
    }
  }

  setupScrollSpy();
  setupSideNavReveal();
  setupThemeToggle();

  /* ------------------------------------------------------------------ *
   * boot
   * ------------------------------------------------------------------ */

  function bindControls(data) {
    document.querySelectorAll("#metric-toggle button").forEach((button) => {
      button.addEventListener("click", () => {
        boardState.metric = button.dataset.metric;
        renderBoard();
      });
    });
    document.querySelectorAll("#board th[data-metric]").forEach((th) => {
      th.style.cursor = "pointer";
      th.addEventListener("click", () => {
        boardState.metric = th.dataset.metric;
        renderBoard();
      });
    });

    const panelSelect = document.getElementById("panel-select");
    if (panelSelect) {
      panelSelect.addEventListener("change", () => {
        boardState.panel = panelSelect.value;
        renderBoard();
        renderLeaderboardMeta(data);
      });
    }

    const boardPaperToggle = document.getElementById("board-paper-toggle");
    if (boardPaperToggle) {
      boardPaperToggle.addEventListener("click", () => {
        boardState.includePaper = !boardState.includePaper;
        boardPaperToggle.classList.toggle("is-active", boardState.includePaper);
        boardPaperToggle.setAttribute("aria-pressed", String(boardState.includePaper));
        renderBoard();
        renderLeaderboardMeta(data);
      });
    }

    document.querySelectorAll("[data-effort-sort]").forEach((button) => {
      button.addEventListener("click", () => {
        const nextSort = button.dataset.effortSort;
        if (effortState.sort === nextSort) {
          effortState.direction = effortState.direction === "desc" ? "asc" : "desc";
        } else {
          effortState.sort = nextSort;
          effortState.direction = "desc";
        }
        renderEffort(data);
      });
    });

    document.querySelectorAll("#trend-metric-toggle button").forEach((button) => {
      button.addEventListener("click", () => {
        trendState.metric = button.dataset.metric;
        renderTrend(data);
      });
    });

    const trendFamilyMenu = document.getElementById("trend-family-options");
    if (trendFamilyMenu) {
      trendFamilyMenu.addEventListener("change", (event) => {
        const changed = event.target.closest('input[type="checkbox"]');
        if (!changed) return;
        const all = trendFamilyMenu.querySelector('input[value="all"]');
        const familyInputs = Array.from(
          trendFamilyMenu.querySelectorAll('input[type="checkbox"]:not([value="all"])')
        );
        if (changed.value === "all") {
          familyInputs.forEach((input) => {
            input.checked = changed.checked;
          });
        } else {
          all.checked = familyInputs.every((input) => input.checked);
        }
        trendState.families = new Set(
          familyInputs.filter((input) => input.checked).map((input) => input.value)
        );
        updateTrendFamilySummary(data);
        renderTrend(data);
      });
    }

    const trendPanelToggle = document.getElementById("trend-panel-toggle");
    if (trendPanelToggle) {
      trendPanelToggle.addEventListener("click", (event) => {
        const button = event.target.closest("button[data-panel]");
        if (!button) return;
        trendState.panel = button.dataset.panel;
        renderTrend(data);
      });
    }

    const trendDownload = document.getElementById("trend-download");
    if (trendDownload) {
      trendDownload.addEventListener("click", () => {
        const mount = $("trend-chart");
        const metric = trendState.metric || "f1";
        const panel = trendState.panel || "all";
        const filename = `sciconbench-progress-${metric}-${panel}.png`;
        trendDownload.disabled = true;
        Charts.downloadChartPng(mount, filename)
          .catch(() => {
            /* Silent fail — button re-enables below. */
          })
          .finally(() => {
            trendDownload.disabled = !mount.querySelector("svg");
          });
      });
    }

    const examplesPrev = document.getElementById("question-examples-prev");
    const examplesNext = document.getElementById("question-examples-next");
    if (examplesPrev && examplesNext) {
      examplesPrev.addEventListener("click", () => {
        questionExampleStart = Math.max(
          0,
          questionExampleStart - questionExamplePageSize()
        );
        renderQuestionExamples();
      });
      examplesNext.addEventListener("click", () => {
        const pageSize = questionExamplePageSize();
        questionExampleStart = Math.min(
          Math.max(0, questionExamples.length - pageSize),
          questionExampleStart + pageSize
        );
        renderQuestionExamples();
      });

      let previousPageSize = questionExamplePageSize();
      window.addEventListener("resize", () => {
        const nextPageSize = questionExamplePageSize();
        if (nextPageSize === previousPageSize) return;
        previousPageSize = nextPageSize;
        renderQuestionExamples();
      });
    }

    const copyBtn = document.getElementById("copy-citation");
    if (copyBtn) {
      copyBtn.addEventListener("click", () => {
        const text = document.getElementById("citation").textContent;
        navigator.clipboard.writeText(text).then(() => {
          const label = copyBtn.querySelector(".copy-label");
          const prev = label.textContent;
          copyBtn.classList.add("is-copied");
          label.textContent = "Copied!";
          setTimeout(() => {
            copyBtn.classList.remove("is-copied");
            label.textContent = prev;
          }, 1600);
        });
      });
    }
  }

  fetch("data/dashboard.json", { cache: "no-cache" })
    .then((response) => {
      if (!response.ok) throw new Error("HTTP " + response.status);
      return response.json();
    })
    .then((data) => {
      if (data.demo) $("demo-banner").hidden = false;
      boardState.base = data.leaderboard || [];
      boardState.paper = data.paper_baselines || [];
      populatePanelSelect(data);
      populateTrendControls(data);
      renderHero(data);
      renderLeaderboardMeta(data);
      renderBoard();
      renderTrend(data);
      renderDataset(data);
      renderFactDiagnostics(data);
      renderEffort(data);
      renderContent(data);
      bindControls(data);
    })
    .catch((error) => {
      $("hero-tagline").textContent =
        "Dashboard data could not be loaded (" + error.message + ").";
    });
})();
