// statiq web — pure JS via jStat. ~17 most-used distributions.

const SLIDER_STEPS = 1000;

// ---------------------------------------------------------------------------
// Distribution definitions
//
// Each distribution exposes: pdf(x, p), cdf(x, p), ppf(q, p), rvs(p, rng).
// For discrete distributions, `pdf` returns the pmf.
// `defaultSupport(p)` returns a reasonable [lo, hi] for plotting.
// ---------------------------------------------------------------------------

const J = jStat;

// Marsaglia & Tsang RNG seeded for reproducible samples
function makeRng(seed) {
  if (!seed || seed === 0) return Math.random;
  let s = seed >>> 0;
  return function() {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function sampleDiscrete(p, rng, ppf) {
  return Math.round(ppf(rng(), p));
}

// Helper: ppf for discrete via cdf scan
function discretePpf(cdf, p, q, kMin = 0, kMaxHint = 1000) {
  let k = kMin;
  while (k < kMaxHint) {
    if (cdf(k, p) >= q) return k;
    k++;
  }
  return kMaxHint;
}

// Numerical root-finder (brentq-style bisection with linear interpolation)
function findRoot(f, lo, hi, tol = 1e-9, maxIter = 200) {
  let fa = f(lo), fb = f(hi);
  if (isNaN(fa) || isNaN(fb)) throw new Error("Search bounds yield invalid values.");
  if (fa * fb > 0) throw new Error("Sign does not change across search interval — widen bounds.");
  for (let i = 0; i < maxIter; i++) {
    const mid = (lo + hi) / 2;
    const fm = f(mid);
    if (isNaN(fm) || Math.abs(hi - lo) < tol) return mid;
    if (fa * fm < 0) {
      hi = mid; fb = fm;
    } else {
      lo = mid; fa = fm;
    }
  }
  return (lo + hi) / 2;
}

const DISTS = [
  // ---- DISCRETE ----
  {
    name: "Bernoulli", discrete: true,
    params: [{name: "p", description: "Probability of success", default: 0.5, integer: false, min: 0, max: 1}],
    default_x: 0,
    info_text: "<b>What it is:</b> A single yes/no trial.<br><br><b>When to use:</b> One coin flip, one product pass/fail.",
    pdf: (k, p) => (k === 0 ? 1 - p.p : k === 1 ? p.p : 0),
    cdf: (k, p) => (k < 0 ? 0 : k < 1 ? 1 - p.p : 1),
    ppf: (q, p) => (q <= 1 - p.p ? 0 : 1),
    rvs: (p, rng) => (rng() < p.p ? 1 : 0),
    defaultSupport: () => [0, 1],
  },
  {
    name: "Binomial", discrete: true,
    params: [
      {name: "n", description: "Number of trials", default: 10, integer: true, min: 1},
      {name: "p", description: "Probability of success", default: 0.5, integer: false, min: 0, max: 1},
    ],
    default_x: 5,
    info_text: "<b>What it is:</b> Number of successes in n independent yes/no trials.<br><br><b>When to use:</b> How many of 20 students pass a test, defective items in a batch.",
    pdf: (k, p) => J.binomial.pdf(Math.round(k), Math.round(p.n), p.p),
    cdf: (k, p) => J.binomial.cdf(Math.floor(k), Math.round(p.n), p.p),
    ppf: (q, p) => discretePpf((k, pp) => J.binomial.cdf(k, Math.round(pp.n), pp.p), p, q, 0, Math.round(p.n) + 1),
    rvs: (p, rng) => {
      let k = 0;
      for (let i = 0; i < Math.round(p.n); i++) if (rng() < p.p) k++;
      return k;
    },
    defaultSupport: (p) => [0, Math.round(p.n)],
  },
  {
    name: "Geometric", discrete: true,
    params: [{name: "p", description: "Probability of success", default: 0.3, integer: false, min: 0, max: 1, exclusive_min: true}],
    default_x: 3,
    info_text: "<b>What it is:</b> Number of failures before the first success (scipy convention: k = 1, 2, …).<br><br><b>When to use:</b> Trials until first success.",
    pdf: (k, p) => (k >= 1 ? Math.pow(1 - p.p, k - 1) * p.p : 0),
    cdf: (k, p) => (k < 1 ? 0 : 1 - Math.pow(1 - p.p, Math.floor(k))),
    ppf: (q, p) => Math.max(1, Math.ceil(Math.log(1 - q) / Math.log(1 - p.p))),
    rvs: (p, rng) => Math.max(1, Math.ceil(Math.log(rng()) / Math.log(1 - p.p))),
    defaultSupport: (p) => [1, Math.max(15, Math.ceil(5 / p.p))],
  },
  {
    name: "Negative Binomial", discrete: true,
    params: [
      {name: "r", description: "Number of successes target", default: 5, integer: true, min: 1},
      {name: "p", description: "Probability of success", default: 0.5, integer: false, min: 0, max: 1, exclusive_min: true},
    ],
    default_x: 5,
    info_text: "<b>What it is:</b> Number of failures before r-th success.<br><br><b>When to use:</b> Quality testing until a target number of successes.",
    pdf: (k, p) => J.negbin.pdf(Math.round(k), Math.round(p.r), p.p),
    cdf: (k, p) => J.negbin.cdf(Math.floor(k), Math.round(p.r), p.p),
    ppf: (q, p) => discretePpf((k, pp) => J.negbin.cdf(k, Math.round(pp.r), pp.p), p, q, 0, 5000),
    rvs: (p, rng) => {
      let succ = 0, fail = 0;
      while (succ < Math.round(p.r)) { if (rng() < p.p) succ++; else fail++; }
      return fail;
    },
    defaultSupport: (p) => [0, Math.ceil(p.r * (1 - p.p) / p.p * 3 + 10)],
  },
  {
    name: "Poisson", discrete: true,
    params: [{name: "λ", description: "Rate (mean)", default: 4.0, integer: false, min: 0, exclusive_min: true}],
    default_x: 4,
    info_text: "<b>What it is:</b> Count of rare events in a fixed interval.<br><br><b>When to use:</b> Customer arrivals per hour, typos per page, radioactive decay counts.",
    pdf: (k, p) => J.poisson.pdf(Math.round(k), p.λ),
    cdf: (k, p) => J.poisson.cdf(Math.floor(k), p.λ),
    ppf: (q, p) => discretePpf((k, pp) => J.poisson.cdf(k, pp.λ), p, q, 0, Math.ceil(p.λ * 5 + 20)),
    rvs: (p, rng) => {
      // Knuth's algorithm
      const L = Math.exp(-p.λ);
      let k = 0, prod = 1;
      do { k++; prod *= rng(); } while (prod > L);
      return k - 1;
    },
    defaultSupport: (p) => [0, Math.max(15, Math.ceil(p.λ + 4 * Math.sqrt(p.λ)))],
  },
  {
    name: "Hypergeometric", discrete: true,
    params: [
      {name: "N", description: "Population size", default: 50, integer: true, min: 1},
      {name: "K", description: "Successes in population", default: 20, integer: true, min: 0},
      {name: "n", description: "Sample size", default: 10, integer: true, min: 1},
    ],
    default_x: 4,
    info_text: "<b>What it is:</b> Successes drawn in n picks without replacement from population N with K successes.<br><br><b>When to use:</b> Lottery, audit samples, capture-recapture.",
    pdf: (k, p) => hypergeomPmf(Math.round(k), Math.round(p.N), Math.round(p.K), Math.round(p.n)),
    cdf: (k, p) => {
      let s = 0;
      for (let i = 0; i <= Math.floor(k); i++) s += hypergeomPmf(i, Math.round(p.N), Math.round(p.K), Math.round(p.n));
      return s;
    },
    ppf: (q, p) => discretePpf((k, pp) => {
      let s = 0;
      for (let i = 0; i <= k; i++) s += hypergeomPmf(i, Math.round(pp.N), Math.round(pp.K), Math.round(pp.n));
      return s;
    }, p, q, 0, Math.min(Math.round(p.K), Math.round(p.n))),
    rvs: (p, rng) => {
      // Draw without replacement simulation
      const N = Math.round(p.N), K = Math.round(p.K), n = Math.round(p.n);
      let remaining = N, successes = K, draws = 0;
      for (let i = 0; i < n; i++) {
        if (rng() < successes / remaining) { draws++; successes--; }
        remaining--;
      }
      return draws;
    },
    defaultSupport: (p) => [Math.max(0, Math.round(p.n) - (Math.round(p.N) - Math.round(p.K))), Math.min(Math.round(p.K), Math.round(p.n))],
  },
  {
    name: "Discrete Uniform", discrete: true,
    params: [
      {name: "a", description: "Lower bound", default: 1, integer: true},
      {name: "b", description: "Upper bound (inclusive)", default: 6, integer: true},
    ],
    default_x: 3,
    info_text: "<b>What it is:</b> Each integer in [a, b] equally likely.<br><br><b>When to use:</b> Rolling a fair die, choosing a random integer.",
    pdf: (k, p) => {
      const a = Math.round(p.a), b = Math.round(p.b);
      return (k >= a && k <= b && k === Math.round(k)) ? 1 / (b - a + 1) : 0;
    },
    cdf: (k, p) => {
      const a = Math.round(p.a), b = Math.round(p.b);
      if (k < a) return 0;
      if (k >= b) return 1;
      return (Math.floor(k) - a + 1) / (b - a + 1);
    },
    ppf: (q, p) => {
      const a = Math.round(p.a), b = Math.round(p.b);
      return Math.min(b, a + Math.ceil(q * (b - a + 1)) - 1);
    },
    rvs: (p, rng) => {
      const a = Math.round(p.a), b = Math.round(p.b);
      return a + Math.floor(rng() * (b - a + 1));
    },
    defaultSupport: (p) => [Math.round(p.a), Math.round(p.b)],
  },

  // ---- CONTINUOUS ----
  {
    name: "Normal", discrete: false,
    params: [
      {name: "μ", description: "Mean", default: 0.0, integer: false},
      {name: "σ", description: "Standard deviation", default: 1.0, integer: false, min: 0, exclusive_min: true},
    ],
    default_x: 0.0,
    info_text: "<b>What it is:</b> The bell curve. Symmetric around the mean.<br><br><b>When to use:</b> Heights, test scores, measurement errors, sums of many small independent effects.",
    pdf: (x, p) => J.normal.pdf(x, p.μ, p.σ),
    cdf: (x, p) => J.normal.cdf(x, p.μ, p.σ),
    ppf: (q, p) => J.normal.inv(q, p.μ, p.σ),
    rvs: (p, rng) => p.μ + p.σ * boxMuller(rng),
    defaultSupport: (p) => [p.μ - 4 * p.σ, p.μ + 4 * p.σ],
  },
  {
    name: "Standard Normal", discrete: false,
    params: [],
    default_x: 0.0,
    info_text: "<b>What it is:</b> Normal with μ=0 and σ=1.<br><br><b>When to use:</b> Z-scores, standardized test statistics, the basis for tables in stats textbooks.",
    pdf: (x) => J.normal.pdf(x, 0, 1),
    cdf: (x) => J.normal.cdf(x, 0, 1),
    ppf: (q) => J.normal.inv(q, 0, 1),
    rvs: (_, rng) => boxMuller(rng),
    defaultSupport: () => [-4, 4],
  },
  {
    name: "Lognormal", discrete: false,
    params: [
      {name: "μ", description: "Mean of ln X", default: 0.0, integer: false},
      {name: "σ", description: "SD of ln X", default: 1.0, integer: false, min: 0, exclusive_min: true},
    ],
    default_x: 1.0,
    info_text: "<b>What it is:</b> Variable whose logarithm is normal. Right-skewed.<br><br><b>When to use:</b> Incomes, stock prices, biological growth, anything multiplicative.",
    pdf: (x, p) => x > 0 ? J.lognormal.pdf(x, p.μ, p.σ) : 0,
    cdf: (x, p) => x > 0 ? J.lognormal.cdf(x, p.μ, p.σ) : 0,
    ppf: (q, p) => J.lognormal.inv(q, p.μ, p.σ),
    rvs: (p, rng) => Math.exp(p.μ + p.σ * boxMuller(rng)),
    defaultSupport: (p) => [0, Math.exp(p.μ + 3 * p.σ)],
  },
  {
    name: "Exponential", discrete: false,
    params: [{name: "λ", description: "Rate", default: 1.0, integer: false, min: 0, exclusive_min: true}],
    default_x: 1.0,
    info_text: "<b>What it is:</b> Waiting time between Poisson events. Memoryless.<br><br><b>When to use:</b> Time to next phone call, time until next radioactive decay.",
    pdf: (x, p) => x >= 0 ? J.exponential.pdf(x, p.λ) : 0,
    cdf: (x, p) => x >= 0 ? J.exponential.cdf(x, p.λ) : 0,
    ppf: (q, p) => -Math.log(1 - q) / p.λ,
    rvs: (p, rng) => -Math.log(rng()) / p.λ,
    defaultSupport: (p) => [0, 6 / p.λ],
  },
  {
    name: "Gamma", discrete: false,
    params: [
      {name: "k", description: "Shape", default: 2.0, integer: false, min: 0, exclusive_min: true},
      {name: "θ", description: "Scale", default: 1.0, integer: false, min: 0, exclusive_min: true},
    ],
    default_x: 2.0,
    info_text: "<b>What it is:</b> Generalization of exponential. Sum of k exponentials when shape is integer.<br><br><b>When to use:</b> Total waiting time for k events, rainfall amounts, insurance claims.",
    pdf: (x, p) => x > 0 ? J.gamma.pdf(x, p.k, p.θ) : 0,
    cdf: (x, p) => x > 0 ? J.gamma.cdf(x, p.k, p.θ) : 0,
    ppf: (q, p) => J.gamma.inv(q, p.k, p.θ),
    rvs: (p, rng) => J.gamma.sample(p.k, p.θ),
    defaultSupport: (p) => [0, (p.k + 4 * Math.sqrt(p.k)) * p.θ],
  },
  {
    name: "Beta", discrete: false,
    params: [
      {name: "α", description: "Shape 1", default: 2.0, integer: false, min: 0, exclusive_min: true},
      {name: "β", description: "Shape 2", default: 2.0, integer: false, min: 0, exclusive_min: true},
    ],
    default_x: 0.5,
    info_text: "<b>What it is:</b> Flexible distribution on [0, 1].<br><br><b>When to use:</b> Modeling proportions, Bayesian prior for probabilities, project completion fractions.",
    pdf: (x, p) => (x > 0 && x < 1) ? J.beta.pdf(x, p.α, p.β) : 0,
    cdf: (x, p) => x <= 0 ? 0 : x >= 1 ? 1 : J.beta.cdf(x, p.α, p.β),
    ppf: (q, p) => J.beta.inv(q, p.α, p.β),
    rvs: (p, rng) => J.beta.sample(p.α, p.β),
    defaultSupport: () => [0, 1],
  },
  {
    name: "Chi-square", discrete: false,
    params: [{name: "df", description: "Degrees of freedom", default: 5.0, integer: false, min: 0, exclusive_min: true}],
    default_x: 5.0,
    info_text: "<b>What it is:</b> Sum of squared independent standard normals.<br><br><b>When to use:</b> Goodness-of-fit tests, variance confidence intervals, contingency tables.",
    pdf: (x, p) => x > 0 ? J.chisquare.pdf(x, p.df) : 0,
    cdf: (x, p) => x > 0 ? J.chisquare.cdf(x, p.df) : 0,
    ppf: (q, p) => J.chisquare.inv(q, p.df),
    rvs: (p, rng) => J.chisquare.sample(p.df),
    defaultSupport: (p) => [0, p.df + 5 * Math.sqrt(2 * p.df)],
  },
  {
    name: "Student-t", discrete: false,
    params: [{name: "df", description: "Degrees of freedom", default: 10.0, integer: false, min: 0, exclusive_min: true}],
    default_x: 0.0,
    info_text: "<b>What it is:</b> Symmetric around 0, fatter tails than normal. Approaches normal as df grows.<br><br><b>When to use:</b> t-tests, confidence intervals for the mean when σ is unknown.",
    pdf: (x, p) => J.studentt.pdf(x, p.df),
    cdf: (x, p) => J.studentt.cdf(x, p.df),
    ppf: (q, p) => J.studentt.inv(q, p.df),
    rvs: (p, rng) => J.studentt.sample(p.df),
    defaultSupport: (p) => {
      const s = p.df > 2 ? Math.sqrt(p.df / (p.df - 2)) : 5;
      return [-4 * s, 4 * s];
    },
  },
  {
    name: "F", discrete: false,
    params: [
      {name: "df₁", description: "Numerator df", default: 5.0, integer: false, min: 0, exclusive_min: true},
      {name: "df₂", description: "Denominator df", default: 10.0, integer: false, min: 0, exclusive_min: true},
    ],
    default_x: 1.0,
    info_text: "<b>What it is:</b> Ratio of two scaled chi-squares.<br><br><b>When to use:</b> ANOVA, comparing two sample variances, regression overall significance tests.",
    pdf: (x, p) => x > 0 ? J.centralF.pdf(x, p["df₁"], p["df₂"]) : 0,
    cdf: (x, p) => x > 0 ? J.centralF.cdf(x, p["df₁"], p["df₂"]) : 0,
    ppf: (q, p) => J.centralF.inv(q, p["df₁"], p["df₂"]),
    rvs: (p, rng) => J.centralF.sample(p["df₁"], p["df₂"]),
    defaultSupport: (p) => [0, 5],
  },
  {
    name: "Continuous Uniform", discrete: false,
    params: [
      {name: "a", description: "Lower bound", default: 0.0, integer: false},
      {name: "b", description: "Upper bound", default: 1.0, integer: false},
    ],
    default_x: 0.5,
    info_text: "<b>What it is:</b> All values in [a, b] equally likely.<br><br><b>When to use:</b> A random number, baseline for simulation, uninformative continuous priors.",
    pdf: (x, p) => (x >= p.a && x <= p.b) ? 1 / (p.b - p.a) : 0,
    cdf: (x, p) => x < p.a ? 0 : x > p.b ? 1 : (x - p.a) / (p.b - p.a),
    ppf: (q, p) => p.a + q * (p.b - p.a),
    rvs: (p, rng) => p.a + rng() * (p.b - p.a),
    defaultSupport: (p) => [p.a, p.b],
  },
  {
    name: "Weibull", discrete: false,
    params: [
      {name: "k", description: "Shape", default: 1.5, integer: false, min: 0, exclusive_min: true},
      {name: "λ", description: "Scale", default: 1.0, integer: false, min: 0, exclusive_min: true},
    ],
    default_x: 1.0,
    info_text: "<b>What it is:</b> Flexible lifetime distribution. Shape < 1: decreasing failure rate. > 1: increasing.<br><br><b>When to use:</b> Reliability engineering, time-to-failure, extreme value modeling.",
    pdf: (x, p) => x >= 0 ? J.weibull.pdf(x, p.λ, p.k) : 0,
    cdf: (x, p) => x >= 0 ? J.weibull.cdf(x, p.λ, p.k) : 0,
    ppf: (q, p) => p.λ * Math.pow(-Math.log(1 - q), 1 / p.k),
    rvs: (p, rng) => p.λ * Math.pow(-Math.log(rng()), 1 / p.k),
    defaultSupport: (p) => [0, p.λ * Math.pow(-Math.log(0.001), 1 / p.k)],
  },
];

// Hypergeometric pmf (computed via logs to avoid overflow)
function logChoose(n, k) {
  if (k < 0 || k > n) return -Infinity;
  return J.lgamma(n + 1) - J.lgamma(k + 1) - J.lgamma(n - k + 1);
}
function hypergeomPmf(k, N, K, n) {
  if (k < Math.max(0, n - (N - K)) || k > Math.min(K, n)) return 0;
  return Math.exp(logChoose(K, k) + logChoose(N - K, n - k) - logChoose(N, n));
}

// Box-Muller transform for normal sampling
function boxMuller(rng) {
  const u1 = Math.max(1e-12, rng());
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

const BY_NAME = Object.fromEntries(DISTS.map(d => [d.name, d]));

// ---------------------------------------------------------------------------
// UI state
// ---------------------------------------------------------------------------

let currentDist = DISTS.find(d => d.name === "Standard Normal");
let currentParams = {};
let currentOp = "probability";
let lastPlotData = null;

const loadingEl = document.getElementById("loading");
const uiEl = document.getElementById("ui");

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

function init() {
  if (typeof jStat === "undefined") {
    document.getElementById("loading-sub").textContent = "Failed: jStat not loaded";
    return;
  }
  setupUI();
  loadingEl.hidden = true;
  uiEl.hidden = false;
}

// ---------------------------------------------------------------------------
// UI setup
// ---------------------------------------------------------------------------

function setupUI() {
  const select = document.getElementById("dist-select");
  const discreteGroup = document.createElement("optgroup");
  discreteGroup.label = `Discrete (${DISTS.filter(d => d.discrete).length})`;
  const continuousGroup = document.createElement("optgroup");
  continuousGroup.label = `Continuous (${DISTS.filter(d => !d.discrete).length})`;
  for (const d of DISTS) {
    const opt = document.createElement("option");
    opt.value = d.name;
    opt.textContent = d.name;
    (d.discrete ? discreteGroup : continuousGroup).appendChild(opt);
  }
  select.appendChild(discreteGroup);
  select.appendChild(continuousGroup);
  select.addEventListener("change", () => selectDistribution(select.value));

  document.querySelectorAll(".tab").forEach(btn => {
    btn.addEventListener("click", () => switchOp(btn.dataset.op));
  });

  document.getElementById("prob-mode").addEventListener("change", () => {
    const mode = document.getElementById("prob-mode").value;
    document.getElementById("prob-b-row").hidden = mode !== "between";
    runProbability();
  });
  setupLinkedNumberSlider("prob-x", () => runProbability());
  setupLinkedNumberSlider("prob-b", () => runProbability());
  setupLinkedNumberSlider("quant-q", () => runQuantile());

  document.getElementById("sample-btn").addEventListener("click", runSample);
  document.getElementById("solve-btn").addEventListener("click", runSolve);

  selectDistribution("Standard Normal");
}

function switchOp(op) {
  currentOp = op;
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.dataset.op === op));
  document.querySelectorAll(".op-panel").forEach(p => p.hidden = p.dataset.op !== op);
  redrawPlot();
  if (op === "probability") runProbability();
  else if (op === "quantile") runQuantile();
}

// ---------------------------------------------------------------------------
// Distribution change
// ---------------------------------------------------------------------------

function selectDistribution(name) {
  currentDist = BY_NAME[name];
  document.getElementById("dist-select").value = name;
  currentParams = {};
  for (const p of currentDist.params) currentParams[p.name] = p.default;

  document.getElementById("dist-info").innerHTML = currentDist.info_text || "";
  buildParamsUI();
  buildSolveParamSelect();

  setNumber("prob-x", currentDist.default_x);
  setNumber("prob-b", currentDist.default_x + 1.0);

  updateProbabilitySliderRange();
  redrawPlot();
  if (currentOp === "probability") runProbability();
  if (currentOp === "quantile") runQuantile();
}

function buildParamsUI() {
  const container = document.getElementById("params");
  container.innerHTML = "";
  if (currentDist.params.length === 0) {
    container.innerHTML = `<p class="hint">No parameters.</p>`;
    return;
  }
  for (const p of currentDist.params) {
    const row = document.createElement("div");
    row.className = "slider-row";
    row.innerHTML = `
      <span class="slider-label" title="${escapeAttr(p.description)}">${p.name}</span>
      <input type="number" id="param-num-${cssId(p.name)}" step="${p.integer ? 1 : "any"}">
      <input type="range" id="param-slider-${cssId(p.name)}" min="0" max="${SLIDER_STEPS}" value="500">
    `;
    container.appendChild(row);

    const numEl = document.getElementById(`param-num-${cssId(p.name)}`);
    const sliderEl = document.getElementById(`param-slider-${cssId(p.name)}`);
    numEl.value = p.default;

    const [lo, hi] = sliderRangeForParam(p);
    linkNumSlider(numEl, sliderEl, lo, hi, p.integer, (v) => {
      currentParams[p.name] = v;
      onParamChange();
    });
  }
}

function buildSolveParamSelect() {
  const sel = document.getElementById("solve-param");
  sel.innerHTML = "";
  for (const p of currentDist.params) {
    const opt = document.createElement("option");
    opt.value = p.name;
    opt.textContent = p.name;
    sel.appendChild(opt);
  }
}

function sliderRangeForParam(p) {
  const def = p.default;
  const has_min = p.min !== undefined && p.min !== null;
  const has_max = p.max !== undefined && p.max !== null;
  if (has_min && has_max) return [p.min, p.max];
  if (has_min) {
    const span = Math.max(Math.abs(def - p.min) * 3, 5);
    return [p.min, p.min + span];
  }
  if (has_max) {
    const span = Math.max(Math.abs(p.max - def) * 3, 5);
    return [p.max - span, p.max];
  }
  const span = Math.max(Math.abs(def) * 3, 5);
  return [def - span, def + span];
}

function onParamChange() {
  updateProbabilitySliderRange();
  redrawPlot();
  if (currentOp === "probability") runProbability();
  else if (currentOp === "quantile") runQuantile();
}

function updateProbabilitySliderRange() {
  const [lo, hi] = currentDist.defaultSupport(currentParams);
  rescaleSlider("prob-x", lo, hi);
  rescaleSlider("prob-b", lo, hi);
}

// ---------------------------------------------------------------------------
// Linked number + slider
// ---------------------------------------------------------------------------

function linkNumSlider(numEl, sliderEl, lo, hi, integer, onChange) {
  numEl._linkedRange = { lo, hi, integer };
  let syncing = false;
  function fromSlider() {
    if (syncing) return;
    const pos = +sliderEl.value;
    let v = numEl._linkedRange.lo + (pos / SLIDER_STEPS) * (numEl._linkedRange.hi - numEl._linkedRange.lo);
    if (numEl._linkedRange.integer) v = Math.round(v);
    syncing = true;
    numEl.value = numEl._linkedRange.integer ? v : +v.toFixed(6);
    syncing = false;
    onChange(+numEl.value);
  }
  function fromNum() {
    if (syncing) return;
    const v = +numEl.value;
    let pos = (v - numEl._linkedRange.lo) / (numEl._linkedRange.hi - numEl._linkedRange.lo) * SLIDER_STEPS;
    pos = Math.max(0, Math.min(SLIDER_STEPS, pos));
    syncing = true;
    sliderEl.value = Math.round(pos);
    syncing = false;
    onChange(v);
  }
  sliderEl.addEventListener("input", fromSlider);
  numEl.addEventListener("input", fromNum);
  // Sync slider to initial number value without firing callbacks
  syncing = true;
  const v = +numEl.value;
  let pos = (v - lo) / (hi - lo) * SLIDER_STEPS;
  pos = Math.max(0, Math.min(SLIDER_STEPS, pos));
  sliderEl.value = Math.round(pos);
  syncing = false;
}

function setupLinkedNumberSlider(id, onChange) {
  const numEl = document.getElementById(`${id}-num`);
  const sliderEl = document.getElementById(`${id}-slider`);
  linkNumSlider(numEl, sliderEl, -5, 5, false, onChange);
}

function rescaleSlider(id, lo, hi) {
  const numEl = document.getElementById(`${id}-num`);
  if (!numEl) return;
  numEl._linkedRange.lo = lo;
  numEl._linkedRange.hi = hi;
  const sliderEl = document.getElementById(`${id}-slider`);
  const v = +numEl.value;
  let pos = (v - lo) / (hi - lo) * SLIDER_STEPS;
  pos = Math.max(0, Math.min(SLIDER_STEPS, pos));
  sliderEl.value = Math.round(pos);
}

function setNumber(id, value) {
  const numEl = document.getElementById(`${id}-num`);
  if (!numEl) return;
  numEl.value = +(+value).toFixed(6);
  const sliderEl = document.getElementById(`${id}-slider`);
  if (numEl._linkedRange && sliderEl) {
    let pos = (value - numEl._linkedRange.lo) / (numEl._linkedRange.hi - numEl._linkedRange.lo) * SLIDER_STEPS;
    pos = Math.max(0, Math.min(SLIDER_STEPS, pos));
    sliderEl.value = Math.round(pos);
  }
}

function getNumber(id) { return +document.getElementById(`${id}-num`).value; }

// ---------------------------------------------------------------------------
// Plot
// ---------------------------------------------------------------------------

function computePlotData() {
  const [lo, hi] = currentDist.defaultSupport(currentParams);
  let x, pdf, cdf;
  if (currentDist.discrete) {
    const kLo = Math.floor(lo);
    const kHi = Math.ceil(hi);
    x = [];
    pdf = [];
    cdf = [];
    for (let k = kLo; k <= kHi; k++) {
      x.push(k);
      pdf.push(currentDist.pdf(k, currentParams));
      cdf.push(currentDist.cdf(k, currentParams));
    }
  } else {
    const N = 400;
    x = []; pdf = []; cdf = [];
    for (let i = 0; i < N; i++) {
      const v = lo + (hi - lo) * i / (N - 1);
      x.push(v);
      pdf.push(currentDist.pdf(v, currentParams));
      cdf.push(currentDist.cdf(v, currentParams));
    }
  }
  return { x, pdf, cdf, lo, hi, discrete: currentDist.discrete };
}

function redrawPlot() {
  if (!currentDist) return;
  try {
    lastPlotData = computePlotData();
    drawPlotly(lastPlotData);
  } catch (e) {
    console.error("plot error:", e);
  }
}

function drawPlotly(data) {
  const { x, pdf, cdf, discrete } = data;
  const layoutCommon = {
    paper_bgcolor: "#1c1c1f",
    plot_bgcolor: "#1c1c1f",
    font: { color: "#e8e8ea", family: "system-ui, -apple-system, sans-serif", size: 12 },
    margin: { l: 50, r: 20, t: 30, b: 40 },
    xaxis: { gridcolor: "#2f2f33", zeroline: false, linecolor: "#555" },
    yaxis: { gridcolor: "#2f2f33", zeroline: false, linecolor: "#555" },
    showlegend: false,
    hovermode: "x unified",
  };

  let pdfTraces = [];
  let cdfTraces = [];
  if (discrete) {
    pdfTraces.push({ x, y: pdf, type: "bar", marker: { color: "#319bff", line: { color: "#0a84ff", width: 1 } } });
    cdfTraces.push({ x, y: cdf, type: "scatter", mode: "lines", line: { color: "#319bff", width: 2, shape: "hv" } });
  } else {
    pdfTraces.push({ x, y: pdf, type: "scatter", mode: "lines", line: { color: "#319bff", width: 2 }, fill: "tozeroy", fillcolor: "rgba(49,155,255,0.15)" });
    cdfTraces.push({ x, y: cdf, type: "scatter", mode: "lines", line: { color: "#319bff", width: 2 } });
  }

  if (currentOp === "probability") {
    const mode = document.getElementById("prob-mode").value;
    const xv = getNumber("prob-x");
    const bv = getNumber("prob-b");
    addProbabilityHighlights(pdfTraces, cdfTraces, mode, xv, bv, x, pdf, cdf, discrete);
  } else if (currentOp === "quantile") {
    const q = getNumber("quant-q");
    const xv = currentDist.ppf(q, currentParams);
    addProbabilityHighlights(pdfTraces, cdfTraces, "le", xv, 0, x, pdf, cdf, discrete);
  }

  const yMaxPdf = Math.max(...pdf.filter(v => isFinite(v))) * 1.1 || 1;
  Plotly.react("plot-pdf", pdfTraces, {
    ...layoutCommon,
    autosize: true,
    title: { text: `${currentDist.name} — ${discrete ? "PMF" : "PDF"}`, font: { size: 13 } },
    yaxis: { ...layoutCommon.yaxis, title: discrete ? "P(X = x)" : "f(x)", range: [0, yMaxPdf], fixedrange: true },
    xaxis: { ...layoutCommon.xaxis, title: "x" },
  }, { responsive: true, displayModeBar: false });

  Plotly.react("plot-cdf", cdfTraces, {
    ...layoutCommon,
    autosize: true,
    title: { text: "CDF", font: { size: 13 } },
    yaxis: { ...layoutCommon.yaxis, title: "P(X ≤ x)", range: [-0.02, 1.02], fixedrange: true },
    xaxis: { ...layoutCommon.xaxis, title: "x" },
  }, { responsive: true, displayModeBar: false });
}

function addProbabilityHighlights(pdfTraces, cdfTraces, mode, xv, bv, x, pdf, cdf, discrete) {
  const c = "#ff453a", fill = "rgba(255,69,58,0.4)";
  function sliceFill(xLo, xHi) {
    const idxs = [];
    for (let i = 0; i < x.length; i++) if (x[i] >= xLo && x[i] <= xHi) idxs.push(i);
    if (!idxs.length) return;
    if (discrete) {
      pdfTraces.push({
        x: idxs.map(i => x[i]),
        y: idxs.map(i => pdf[i]),
        type: "bar",
        marker: { color: c, line: { color: "#ff6961", width: 1 } },
      });
    } else {
      pdfTraces.push({
        x: idxs.map(i => x[i]),
        y: idxs.map(i => pdf[i]),
        type: "scatter",
        mode: "lines",
        line: { width: 0 },
        fill: "tozeroy",
        fillcolor: fill,
      });
    }
  }
  function vertical(xVal) {
    const yMax = Math.max(...pdf) * 1.05;
    pdfTraces.push({ x: [xVal, xVal], y: [0, yMax], type: "scatter", mode: "lines", line: { color: "#888", width: 1, dash: "dash" } });
    cdfTraces.push({ x: [xVal, xVal], y: [0, 1], type: "scatter", mode: "lines", line: { color: "#888", width: 1, dash: "dash" } });
    cdfTraces.push({ x: [xVal], y: [interpCDF(xVal, x, cdf)], type: "scatter", mode: "markers", marker: { color: c, size: 8 } });
  }
  if (mode === "le" || mode === "lt") {
    sliceFill(-Infinity, mode === "lt" && discrete ? xv - 1 : xv);
    vertical(xv);
  } else if (mode === "ge" || mode === "gt") {
    sliceFill(mode === "gt" && !discrete ? xv : xv, Infinity);
    vertical(xv);
  } else if (mode === "between") {
    sliceFill(xv, bv);
    vertical(xv);
    vertical(bv);
  } else if (mode === "eq") {
    vertical(xv);
  }
}

function interpCDF(xv, xs, cdf) {
  for (let i = 0; i < xs.length - 1; i++) {
    if (xv >= xs[i] && xv <= xs[i + 1]) {
      const t = (xv - xs[i]) / (xs[i + 1] - xs[i] || 1);
      return cdf[i] + t * (cdf[i + 1] - cdf[i]);
    }
  }
  if (xv < xs[0]) return cdf[0];
  return cdf[cdf.length - 1];
}

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

function runProbability() {
  if (!currentDist) return;
  try {
    const mode = document.getElementById("prob-mode").value;
    const x = getNumber("prob-x");
    const b = getNumber("prob-b");
    let p;
    if (mode === "le") p = currentDist.cdf(x, currentParams);
    else if (mode === "lt") p = currentDist.discrete ? currentDist.cdf(x - 1, currentParams) : currentDist.cdf(x, currentParams);
    else if (mode === "ge") p = 1 - (currentDist.discrete ? currentDist.cdf(x - 1, currentParams) : currentDist.cdf(x, currentParams));
    else if (mode === "gt") p = 1 - currentDist.cdf(x, currentParams);
    else if (mode === "eq") p = currentDist.discrete ? currentDist.pdf(x, currentParams) : 0;
    else if (mode === "between") {
      p = currentDist.discrete
        ? currentDist.cdf(b, currentParams) - currentDist.cdf(x - 1, currentParams)
        : currentDist.cdf(b, currentParams) - currentDist.cdf(x, currentParams);
    }
    document.getElementById("prob-result").textContent = p.toFixed(6);
    redrawPlot();
  } catch (e) {
    document.getElementById("prob-result").textContent = "error";
    console.error(e);
  }
}

function runQuantile() {
  if (!currentDist) return;
  try {
    const q = getNumber("quant-q");
    const x = currentDist.ppf(q, currentParams);
    document.getElementById("quant-result").textContent = currentDist.discrete ? String(Math.round(x)) : (+x).toFixed(6);
    redrawPlot();
  } catch (e) {
    document.getElementById("quant-result").textContent = "error";
    console.error(e);
  }
}

function runSample() {
  if (!currentDist) return;
  try {
    const n = Math.max(1, Math.min(10000, +document.getElementById("sample-n").value));
    const seed = +document.getElementById("sample-seed").value;
    const rng = makeRng(seed);
    const samples = [];
    for (let i = 0; i < n; i++) samples.push(currentDist.rvs(currentParams, rng));
    const out = currentDist.discrete ? samples.join(", ") : samples.map(v => v.toFixed(4)).join(", ");
    document.getElementById("sample-output").textContent = out;
    const mean = samples.reduce((a, b) => a + b, 0) / n;
    const variance = n > 1 ? samples.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1) : 0;
    const sd = Math.sqrt(variance);
    const min = Math.min(...samples);
    const max = Math.max(...samples);
    document.getElementById("sample-stats").innerHTML = `
      <div><span>Mean</span><b>${mean.toFixed(4)}</b></div>
      <div><span>SD</span><b>${sd.toFixed(4)}</b></div>
      <div><span>Min</span><b>${min.toFixed(4)}</b></div>
      <div><span>Max</span><b>${max.toFixed(4)}</b></div>
    `;
  } catch (e) {
    document.getElementById("sample-output").textContent = `error: ${e.message}`;
    console.error(e);
  }
}

function runSolve() {
  if (!currentDist) return;
  const resultEl = document.getElementById("solve-result");
  const panel = document.querySelector('.op-panel[data-op="solve"]');
  panel.querySelectorAll(".error").forEach(n => n.remove());
  try {
    if (currentDist.params.length === 0) {
      resultEl.textContent = "(no parameters)";
      return;
    }
    const unknown = document.getElementById("solve-param").value;
    const xv = +document.getElementById("solve-x").value;
    const targetP = +document.getElementById("solve-p").value;
    const lo = +document.getElementById("solve-lo").value;
    const hi = +document.getElementById("solve-hi").value;
    const paramSpec = currentDist.params.find(p => p.name === unknown);

    const f = (theta) => {
      const vals = { ...currentParams, [unknown]: theta };
      try {
        return currentDist.cdf(xv, vals) - targetP;
      } catch {
        return NaN;
      }
    };

    let root;
    let intMode = false;
    if (paramSpec.integer) {
      const loI = Math.ceil(lo), hiI = Math.floor(hi);
      let best = null, bestErr = Infinity;
      for (let k = loI; k <= hiI; k++) {
        const err = f(k);
        if (!isNaN(err) && Math.abs(err) < bestErr) { best = k; bestErr = Math.abs(err); }
      }
      if (best === null) throw new Error("No solution found in integer search range.");
      root = best;
      intMode = true;
      resultEl.textContent = `${unknown} = ${root}  (Δ=${bestErr.toExponential(2)})`;
    } else {
      root = findRoot(f, lo, hi);
      resultEl.textContent = `${unknown} = ${root.toFixed(6)}`;
    }
  } catch (e) {
    resultEl.textContent = "—";
    const note = document.createElement("div");
    note.className = "error";
    note.textContent = e.message || String(e);
    panel.appendChild(note);
    console.error(e);
  }
}

// ---------------------------------------------------------------------------
// Utils
// ---------------------------------------------------------------------------

function escapeAttr(s) { return String(s).replace(/"/g, "&quot;"); }
function cssId(s) { return s.replace(/[^a-zA-Z0-9_-]/g, "_"); }

// Go!
init();
