// statiq web — quick calc command bar + distribution table.
// Grammar:
//   <dist> <op> <value> [params]      e.g.  norm cdf 1.96 | t quant .975 df=20
//   <dist> P(X<=v) [params]           e.g.  binom P(X<=3) n=10 p=.5
//   <dist> table [params]             e.g.  chi2 table df=5
// Ops: cdf pdf pmf sf quant (aliases ppf/inv). Missing params fall back to defaults.

(function () {
  const ALIASES = {
    norm: "Normal", normal: "Normal",
    z: "Standard Normal", stdnorm: "Standard Normal", snorm: "Standard Normal",
    t: "Student-t", student: "Student-t", studentt: "Student-t",
    chi2: "Chi-square", chisq: "Chi-square", chisquare: "Chi-square",
    f: "F", fish: "F", fisher: "F",
    binom: "Binomial", binomial: "Binomial",
    pois: "Poisson", poisson: "Poisson",
    exp: "Exponential", exponential: "Exponential",
    geom: "Geometric",
    nbinom: "Negative Binomial", negbin: "Negative Binomial",
    hyper: "Hypergeometric", hypergeom: "Hypergeometric",
    dunif: "Discrete Uniform", discunif: "Discrete Uniform",
    unif: "Continuous Uniform", uniform: "Continuous Uniform",
    beta: "Beta", gamma: "Gamma",
    lognorm: "Lognormal", lognormal: "Lognormal",
    weibull: "Weibull", weib: "Weibull",
    bern: "Bernoulli", bernoulli: "Bernoulli",
  };

  const OP_ALIASES = {
    cdf: "cdf", prob: "cdf", p: "cdf",
    pdf: "pdf", density: "pdf", d: "pdf",
    pmf: "pmf", mass: "pmf",
    sf: "sf", survival: "sf", ccdf: "sf",
    quant: "quant", quantile: "quant", ppf: "quant", inv: "quant", q: "quant",
  };

  const PARAM_ALIASES = {
    mu: "μ", mean: "μ", m: "μ",
    sigma: "σ", sd: "σ", s: "σ",
    lambda: "λ", lam: "λ", rate: "λ", l: "λ",
    alpha: "α", beta: "β", df1: "df₁", df2: "df₂", "df_1": "df₁", "df_2": "df₂",
    theta: "θ", scale: "θ", shape: "k",
  };

  function normParam(raw) {
    const key = String(raw).toLowerCase().replace(/[=:]$/, "");
    return PARAM_ALIASES[key] || raw;
  }

  function findParamSpec(dist, key) {
    const wanted = normParam(key);
    let spec = dist.params.find(p => p.name === wanted);
    if (!spec) {
      const low = wanted.toLowerCase();
      spec = dist.params.find(p => p.name.toLowerCase() === low);
    }
    return spec;
  }

  // Returns {distName, dist, mode, op, value, params, table, note[]}
  function parseCommand(raw) {
    let text = String(raw || "").trim().replace(/\s+/g, " ");
    if (!text) throw new Error("Empty command.");
    let table = false;
    // Event pattern: P(X <= 3), P(X=2) ...
    let event = null;
    const em = text.match(/p\s*\(\s*x\s*(<=|>=|<|>|=)\s*(-?[\d.]+)\s*\)/i);
    if (em) {
      event = { cmp: em[1], x: parseFloat(em[2]) };
      text = (text.slice(0, em.index) + " " + text.slice(em.index + em[0].length)).replace(/\s+/g, " ").trim();
    }
    const tokens = text.split(" ").filter(Boolean);
    if (!tokens.length) throw new Error("Missing distribution.");
    const alias = tokens.shift().toLowerCase();
    const distName = ALIASES[alias];
    if (!distName) throw new Error(`Unknown distribution "${alias}". Try: norm, t, chi2, f, binom, pois, exp, beta, gamma, unif, ...`);
    const dist = BY_NAME[distName];
    if (tokens.length && tokens[0].toLowerCase() === "table") {
      table = true;
      tokens.shift();
    }

    let op = null;
    if (tokens.length && OP_ALIASES[tokens[0].toLowerCase()]) op = OP_ALIASES[tokens.shift().toLowerCase()];

    let value = null;
    if (tokens.length && !isNaN(parseFloat(tokens[0])) && !tokens[0].includes("=")) {
      value = parseFloat(tokens.shift());
    }

    // Everything else must be key=value or key value pairs.
    const params = {};
    while (tokens.length) {
      let key = tokens.shift();
      let val = null;
      if (key.includes("=")) {
        const parts = key.split("=");
        key = parts[0];
        val = parseFloat(parts[1]);
        if (isNaN(val)) throw new Error(`Bad value for parameter "${key}".`);
      } else {
        if (!tokens.length || isNaN(parseFloat(tokens[0]))) throw new Error(`Missing value for parameter "${key}".`);
        val = parseFloat(tokens.shift());
      }
      const spec = findParamSpec(dist, key);
      if (!spec) throw new Error(`"${dist.name}" has no parameter "${key}". Available: ${dist.params.map(p => p.name).join(", ") || "none"}.`);
      params[spec.name] = spec.integer ? Math.round(val) : val;
    }

    if (event) {
      if (value !== null) throw new Error("Value given twice (P(X≤v) and number).");
      value = event.x;
      if (!op) op = event.cmp === "=" ? "pmf" : (event.cmp === "<" || event.cmp === "<=") ? "cdf" : "sf";
      if (event.cmp === "<") event.mode = "lt";
      else if (event.cmp === "<=") event.mode = "le";
      else if (event.cmp === ">") event.mode = "gt";
      else if (event.cmp === ">=") event.mode = "ge";
      else event.mode = "eq";
    }

    if (table) return { distName, dist, table: true, params, event: null, op: null, value: null };

    if (!op) throw new Error("Missing operation: cdf | pdf | pmf | sf | quant — e.g. norm cdf 1.96");
    if (op !== "quant" && !op.startsWith("pmf") && dist.discrete && op === "pdf") op = "pmf";
    if (op === "pmf" && !dist.discrete) op = "pdf";
    if (value === null) throw new Error("Missing value, e.g. norm cdf 1.96");
    return { distName, dist, table: false, op, value, params, event };
  }

  function mergedParams(dist, given) {
    const merged = {}, usedDefault = [];
    for (const p of dist.params) {
      if (given[p.name] !== undefined) merged[p.name] = given[p.name];
      else { merged[p.name] = p.default; usedDefault.push(p.name); }
    }
    return { merged, usedDefault };
  }

  const SYM = { le: "≤", lt: "<", ge: "≥", gt: ">", eq: "=" };

  function computeCommand(cmd) {
    const { merged } = mergedParams(cmd.dist, cmd.params);
    const d = cmd.dist, x = cmd.value;
    if (cmd.op === "quant") {
      if (!(x > 0 && x < 1)) throw new Error("Quantile needs q in (0,1), e.g. t quant 0.975 df=20");
      const qx = d.ppf(x, merged);
      return { text: `x = ${d.discrete ? Math.round(qx) : (+qx).toFixed(6)}`, detail: `P(X ≤ x) = ${x}` };
    }
    let p;
    if (cmd.op === "cdf") {
      const mode = cmd.event ? cmd.event.mode : "le";
      if (mode === "le") p = d.cdf(x, merged);
      else if (mode === "lt") p = d.discrete ? d.cdf(x - 1, merged) : d.cdf(x, merged);
      else if (mode === "ge") p = 1 - (d.discrete ? d.cdf(x - 1, merged) : d.cdf(x, merged));
      else if (mode === "gt") p = 1 - d.cdf(x, merged);
      else p = d.discrete ? d.pdf(x, merged) : 0;
      const sym = cmd.event ? SYM[cmd.event.mode] : "≤";
      return { text: `P(X ${sym} ${x}) = ${p.toFixed(6)}` };
    }
    if (cmd.op === "sf") {
      if (cmd.event && cmd.event.mode === "ge") {
        p = 1 - (d.discrete ? d.cdf(x - 1, merged) : d.cdf(x, merged));
        return { text: `P(X ≥ ${x}) = ${p.toFixed(6)}` };
      }
      p = 1 - d.cdf(x, merged);
      return { text: `P(X > ${x}) = ${p.toFixed(6)}` };
    }
    if (cmd.op === "pdf") { p = d.pdf(x, merged); return { text: `f(${x}) = ${(+p).toFixed(6)}` }; }
    if (cmd.op === "pmf") { p = d.pdf(x, merged); return { text: `P(X = ${x}) = ${p.toFixed(6)}` }; }
    throw new Error("Unsupported operation.");
  }

  function fmtParams(dist, given) {
    const { merged, usedDefault } = mergedParams(dist, given);
    if (!dist.params.length) return "";
    const parts = dist.params.map(p => `${p.name}=${merged[p.name]}`);
    const note = usedDefault.length ? ` · default for ${usedDefault.join(", ")}` : "";
    return ` (${parts.join(", ")})${note}`;
  }

  // ---------------------------------------------------------------------------
  // UI wiring
  // ---------------------------------------------------------------------------

  function el(id) { return document.getElementById(id); }

  function setParamValue(name, value) {
    const numEl = el(`param-num-${cssId(name)}`);
    if (!numEl) return;
    numEl.value = value;
    numEl.dispatchEvent(new Event("input", { bubbles: true }));
  }

  function applyToExplorer(cmd) {
    selectDistribution(cmd.distName);
    for (const [k, v] of Object.entries(cmd.params)) setParamValue(k, v);
    if (cmd.op === "quant") {
      switchOp("quantile");
      setNumber("quant-q", cmd.value);
      runQuantile();
    } else {
      switchOp("probability");
      const modeSel = el("prob-mode");
      if (cmd.event) modeSel.value = cmd.event.mode === "lt" ? "lt" : cmd.event.mode === "gt" ? "gt" : cmd.event.mode === "ge" ? "ge" : cmd.event.mode === "eq" ? "eq" : "le";
      else modeSel.value = cmd.op === "sf" ? (currentDist.discrete ? "gt" : "ge") : cmd.op === "pmf" ? "eq" : "le";
      modeSel.dispatchEvent(new Event("change"));
      setNumber("prob-x", cmd.value);
      runProbability();
    }
  }

  function buildTable(cmd) {
    const { merged } = mergedParams(cmd.dist, cmd.params);
    const d = cmd.dist;
    const [lo, hi] = d.defaultSupport(merged);
    const rows = [];
    if (d.discrete) {
      for (let k = Math.floor(lo); k <= Math.ceil(hi); k++) rows.push([k, d.pdf(k, merged), d.cdf(k, merged)]);
    } else {
      const N = 49;
      for (let i = 0; i < N; i++) {
        const v = lo + (hi - lo) * i / (N - 1);
        rows.push([v, d.pdf(v, merged), d.cdf(v, merged)]);
      }
    }
    let html = `<table><thead><tr><th>x</th><th>${d.discrete ? "P(X = x)" : "f(x)"}</th><th>F(x) = P(X ≤ x)</th></tr></thead><tbody>`;
    for (const [xv, fv, cv] of rows) {
      html += `<tr><td>${(+xv).toFixed(d.discrete ? 0 : 2)}</td><td>${(+fv).toFixed(6)}</td><td>${(+cv).toFixed(6)}</td></tr>`;
    }
    html += "</tbody></table>";
    el("cmd-table").innerHTML = html;
    el("cmd-table-title").textContent = `${d.name} — table${fmtParams(d, cmd.params)}`;
    el("cmd-table-wrap").hidden = false;
    applyToExplorer(cmd.table ? { ...cmd, op: "cdf", value: d.default_x ?? (lo + hi) / 2, event: null } : cmd);
  }

  function runCommand() {
    const input = el("cmd-input");
    const resultEl = el("cmd-result");
    el("cmd-table-wrap").hidden = true;
    try {
      const cmd = parseCommand(input.value);
      if (cmd.table) {
        buildTable(cmd);
        resultEl.innerHTML = `<span class="cmd-ok">Table ready.</span>`;
        return;
      }
      const res = computeCommand(cmd);
      const suffix = fmtParams(cmd.dist, cmd.params);
      resultEl.innerHTML = `<span class="cmd-ok">${escapeHtml(res.text)}</span><span class="cmd-detail">${escapeHtml(res.detail || "")}${escapeHtml(suffix)}</span>`;
      applyToExplorer(cmd);
    } catch (e) {
      resultEl.innerHTML = `<span class="cmd-err">${escapeHtml(e.message || String(e))}</span>`;
    }
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  }

  document.addEventListener("DOMContentLoaded", () => {
    el("cmd-run").addEventListener("click", runCommand);
    el("cmd-input").addEventListener("keydown", (e) => { if (e.key === "Enter") runCommand(); });
    el("cmd-table-copy").addEventListener("click", () => {
      const text = el("cmd-table").innerText;
      navigator.clipboard.writeText(text).then(() => {
        el("cmd-table-copy").textContent = "Copied";
        setTimeout(() => { el("cmd-table-copy").textContent = "Copy"; }, 1200);
      });
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "/" && document.activeElement !== el("cmd-input") && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) {
        e.preventDefault();
        el("cmd-input").focus();
      }
    });
  });
})();
