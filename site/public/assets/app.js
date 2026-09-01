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

  // Brand marks for the hero action buttons — each is the real logo art
  // (arXiv's own logomark, GitHub's Octocat badge, HuggingFace's emoji
  // mark) rather than a flattened single-color redraw.
  const ICONS = {
    arxiv: '<img src="assets/icon-arxiv.png" alt="" />',
    huggingface: '<img src="assets/icon-huggingface.png" alt="" />',
    github: '<img src="assets/icon-github.png" alt="" />',
    blog: '<img src="assets/icon-blog.png" alt="" />',
  };

  /* The next run always fires at 00:00 America/New_York on the 1st. */
  function nextRunLabel() {
    const now = new Date();
    const next = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth() + 1, 1));
    const days = Math.max(0, Math.ceil((next - now) / 86400000));
    return {
      label: next.toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" }),
      days,
    };
  }

  /* ------------------------------------------------------------------ *
   * hero
   * ------------------------------------------------------------------ */

  function renderHero(data) {
    const site = data.site || {};
    const summary = data.summary;
    const dataset = data.dataset;

    $("hero-tagline").textContent = site.tagline || "";
    $("intro-lede").textContent = site.description || "";

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

    const next = nextRunLabel();
    const benchmark = data.benchmark || {};
    const stats = [
      {
        label: "Benchmark questions",
        value: benchmark.available ? fmtCompact(benchmark.reviews) : fmtInt(dataset.total_reviews),
        note: benchmark.available
          ? `${fmtCompact(benchmark.atomic_facts)} expert atomic facts`
          : "published on HuggingFace",
      },
      {
        label: "Live evaluation panel",
        value: fmtInt(dataset.total_reviews),
        note: `${fmtInt(dataset.core_reviews)} core · ${fmtInt(dataset.rolling_reviews)} rolling`,
      },
      {
        label: "Models tracked",
        value: fmtInt(summary.models_tracked),
        note: `${fmtInt(summary.total_responses)} evaluations run`,
      },
      {
        label: "Latest cohort",
        value: summary.latest_run_month_label || "—",
        note: `next run ${next.label} · in ${next.days}d`,
      },
    ];

    $("hero-stats").innerHTML = stats
      .map(
        (s) =>
          `<div><dt>${escape(s.label)}</dt><dd>${escape(s.value)}<span class="stat-note">${escape(
            s.note
          )}</span></dd></div>`
      )
      .join("");
  }

  /* ------------------------------------------------------------------ *
   * leaderboard
   * ------------------------------------------------------------------ */

  let boardState = { metric: "f1", rows: [] };

  function renderBoard() {
    const metric = boardState.metric;
    const rows = boardState.rows.slice().sort((a, b) => {
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
        '<tr><td colspan="7" class="empty"><strong>No evaluations recorded yet</strong>' +
        "The first monthly run will populate this table.</td></tr>";
      return;
    }

    const cell = (value, color, isBest) => {
      const pct = fmtPct(value);
      if (pct === null) return '<td class="col-num"><span class="tag tag--pending">pending</span></td>';
      return (
        `<td class="col-num"><span class="score">` +
        `<span class="score-value${isBest ? " score-value--lead" : ""}">${pct}</span>` +
        `<span class="score-bar"><i style="width:${(value * 100).toFixed(1)}%;background:${color}"></i></span>` +
        `</span></td>`
      );
    };

    $("board-body").innerHTML = rows
      .map((row, index) => {
        const rank = row[metric] === null ? "—" : index + 1;
        const isBest = row[metric] !== null && best !== null && row[metric] === best;
        return (
          `<tr>` +
          `<td class="col-rank">${rank}</td>` +
          `<td class="col-model"><span class="model-cell">` +
          `<span class="model-dot" style="background:${row.color};color:${row.color}"></span>` +
          `<span><span class="model-name">${escape(row.display_name)}</span>` +
          `<span class="model-provider">${escape(row.provider_label)}</span></span></span></td>` +
          cell(row.precision, row.color, metric === "precision" && isBest) +
          cell(row.recall, row.color, metric === "recall" && isBest) +
          cell(row.f1, row.color, metric === "f1" && isBest) +
          `<td class="col-num col-hide-sm">${fmtInt(row.reviews)}</td>` +
          `<td class="col-num col-hide-sm"><span class="tag">${escape(
            row.run_month_label
          )}</span></td>` +
          `</tr>`
        );
      })
      .join("");
  }

  function renderLeaderboardMeta(data) {
    const summary = data.summary;
    const pending = summary.total_responses - summary.graded_responses;

    $("leaderboard-sub").innerHTML =
      `Macro-averaged over ${fmtInt(data.dataset.total_reviews)} systematic reviews, ` +
      `clean-room configuration (<code>${escape(data.eval_config)}</code>). ` +
      `Each model is shown at its most recent evaluated month.`;

    const notes = [];
    if (pending > 0) {
      notes.push(
        `${fmtInt(pending)} of ${fmtInt(summary.total_responses)} model responses are still ` +
          `awaiting the factual precision and recall judges; those rows show as <em>pending</em>.`
      );
    }
    notes.push(
      "Higher is better for every column. Precision penalizes claims the review contradicts; " +
        "recall measures how much of the expert conclusion the model recovered."
    );
    $("board-footnote").innerHTML = notes.join(" ");
  }

  /* ------------------------------------------------------------------ *
   * trend
   * ------------------------------------------------------------------ */

  function renderTrend(data) {
    const mount = $("trend-chart");
    const series = (data.series || []).map((s) => ({
      display_name: s.display_name,
      color: s.color,
      points: s.points.map((p) => ({
        label: p.label,
        value: p.f1,
        note: `${p.reviews} reviews`,
      })),
    }));

    if (!series.length) {
      mount.innerHTML =
        '<div class="empty"><strong>Not enough history yet</strong>' +
        "F1 trends appear once the judging stages complete for at least one monthly run.</div>";
      $("trend-legend").innerHTML = "";
      return;
    }

    Charts.lineChart(mount, series, {
      yLabel: "F1",
      format: (v) => (v * 100).toFixed(1),
      tick: (t) => (t * 100).toFixed(0),
    });

    $("trend-legend").innerHTML = series
      .map(
        (s) =>
          `<span><i style="background:${s.color}"></i>${escape(s.display_name)}</span>`
      )
      .join("");
  }

  /* ------------------------------------------------------------------ *
   * pipeline
   * ------------------------------------------------------------------ */

  const STATE_LABEL = { ok: "done", running: "running", failed: "failed", pending: "queued" };

  function renderPipeline(data) {
    const pipeline = data.pipeline || {};
    const stages = pipeline.stages || [];

    const done = stages.filter((s) => s.status === "ok").length;
    const running = stages.find((s) => s.status === "running");

    if (pipeline.available) {
      $("pipeline-sub").innerHTML =
        `Run for cohort <strong>${escape(pipeline.target_month || "—")}</strong>, started ` +
        `${escape(fmtDate(pipeline.started_at))} — ${done} of ${stages.length} stages complete` +
        (running ? `, currently <strong>${escape(running.short)}</strong>.` : ".") +
        " The pipeline runs unattended on the first of every month.";
    } else {
      $("pipeline-sub").textContent =
        "The twelve stages of the monthly pipeline. Status appears here once a run has been logged.";
    }

    $("stages").innerHTML = stages
      .map(
        (s, i) =>
          `<div class="stage" data-status="${s.status}">` +
          `<div class="stage-top"><span class="stage-num">${String(i + 1).padStart(2, "0")}</span>` +
          `<span class="stage-state">${STATE_LABEL[s.status] || s.status}</span></div>` +
          `<div class="stage-name">${escape(s.short)}</div>` +
          `<div class="stage-desc">${escape(s.description)}</div>` +
          (s.detail ? `<div class="stage-detail">${escape(s.detail)}</div>` : "") +
          `</div>`
      )
      .join("");

  }

  /* ------------------------------------------------------------------ *
   * dataset
   * ------------------------------------------------------------------ */

  function renderDataset(data) {
    const dataset = data.dataset;

    const rows = (dataset.growth || []).map((g) => ({
      label: g.label,
      segments: [
        { name: "Core panel", value: g.core, color: "#5b8def" },
        { name: "Rolling cohorts", value: g.rolling, color: "#39d0a0" },
      ],
    }));

    if (rows.length) {
      Charts.stackedBars($("growth-chart"), rows, { unit: " reviews" });
    } else {
      $("growth-chart").innerHTML =
        '<div class="empty"><strong>No cohorts yet</strong>Monthly cohorts appear after the first run.</div>';
    }

    const facts = [
      ["Total reviews", fmtInt(dataset.total_reviews)],
      ["Core panel (frozen)", fmtInt(dataset.core_reviews)],
      ["Rolling cohorts", fmtInt(dataset.rolling_reviews)],
      ["Benchmark questions", fmtInt(dataset.questions)],
      ["Expert atomic facts", fmtInt(dataset.atomic_facts)],
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

    const types = dataset.review_types || [];
    const max = Math.max(1, ...types.map((t) => t.count));
    $("review-types").innerHTML = types
      .map(
        (t) =>
          `<div class="bar-row"><div class="bar-row-top"><span>${escape(
            t.label
          )}</span><span>${fmtInt(t.count)}</span></div>` +
          `<div class="bar-track"><i style="width:${(t.count / max) * 100}%"></i></div></div>`
      )
      .join("");
  }

  /* ------------------------------------------------------------------ *
   * effort
   * ------------------------------------------------------------------ */

  function renderEffort(data) {
    const rows = data.leaderboard || [];
    if (!rows.length) {
      $("effort-body").innerHTML =
        '<tr><td colspan="6" class="empty">No evaluations recorded yet.</td></tr>';
      return;
    }

    const sorted = rows
      .slice()
      .sort((a, b) => (b.avg_tool_calls || 0) - (a.avg_tool_calls || 0));

    $("effort-body").innerHTML = sorted
      .map((row) => {
        const top = (row.tool_usage || [])[0];
        return (
          `<tr>` +
          `<td class="col-model"><span class="model-cell">` +
          `<span class="model-dot" style="background:${row.color};color:${row.color}"></span>` +
          `<span class="model-name">${escape(row.display_name)}</span></span></td>` +
          `<td class="col-num">${row.avg_tool_calls === null ? "—" : row.avg_tool_calls.toFixed(1)}</td>` +
          `<td class="col-num">${row.avg_iterations === null ? "—" : row.avg_iterations.toFixed(1)}</td>` +
          `<td class="col-num">${fmtCompact(row.avg_input_tokens)}</td>` +
          `<td class="col-num">${fmtCompact(row.avg_output_tokens)}</td>` +
          `<td class="col-num col-hide-sm">${
            top ? escape(top.tool.replace(/_/g, " ")) : "—"
          }</td>` +
          `</tr>`
        );
      })
      .join("");
  }

  /* ------------------------------------------------------------------ *
   * static content from site.config.json
   * ------------------------------------------------------------------ */

  function renderContent(data) {
    const site = data.site || {};

    $("news-list").innerHTML = (site.news || [])
      .map(
        (n) =>
          `<li><img class="news-icon" src="assets/logo.png" alt="" aria-hidden="true" /><time>[${escape(
            fmtDate(n.date)
          )}]:</time> <span>${escape(n.text)}</span></li>`
      )
      .join("");

    $("faq").innerHTML = (site.faq || [])
      .map(
        (f, i) =>
          `<details${i === 0 ? " open" : ""}><summary>${escape(
            f.question
          )}</summary><p>${escape(f.answer)}</p></details>`
      )
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

    $("citation").textContent = site.citation || "";

    $("contact").textContent =
      "For any inquiries, questions, or feedback, please contact us at " +
      "hayoung [at] cs [dot] princeton [dot] edu!";

    const ack = Array.isArray(site.acknowledgements)
      ? site.acknowledgements
      : site.acknowledgements
      ? [site.acknowledgements]
      : [];
    $("ack-text").innerHTML = ack
      .map((p) => `<p class="team-toggle-legend">${escape(p)}</p>`)
      .join("");

    const generated = new Date(data.generated_at);
    $("footer-generated").textContent =
      `SciConBench · ${site.institution || ""} · data generated ` +
      (isNaN(generated)
        ? data.generated_at
        : generated.toLocaleString("en-US", {
            dateStyle: "medium", timeStyle: "short", timeZone: "UTC",
          }) + " UTC");

    const links = site.links || {};
    $("footer-links").innerHTML = Object.entries({
      Paper: links.paper, Dataset: links.dataset, Code: links.code,
    })
      .filter(([, href]) => href)
      .map(
        ([label, href]) =>
          `<a href="${escape(href)}" target="_blank" rel="noopener">${escape(label)}</a>`
      )
      .join("");
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

  setupScrollSpy();
  setupSideNavReveal();

  /* ------------------------------------------------------------------ *
   * boot
   * ------------------------------------------------------------------ */

  function bindControls() {
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
      boardState.rows = data.leaderboard || [];
      renderHero(data);
      renderLeaderboardMeta(data);
      renderBoard();
      renderTrend(data);
      renderPipeline(data);
      renderDataset(data);
      renderEffort(data);
      renderContent(data);
      bindControls();
    })
    .catch((error) => {
      $("hero-tagline").textContent =
        "Dashboard data could not be loaded (" + error.message + ").";
    });
})();
