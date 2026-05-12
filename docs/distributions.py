"""Distribution definitions for PyQRS Clone — 10 discrete + 22 continuous."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import stats


@dataclass
class ParamSpec:
    name: str
    description: str
    default: float
    integer: bool = False
    min_value: float = -math.inf
    max_value: float = math.inf
    exclusive_min: bool = False
    exclusive_max: bool = False

    def validate(self, value: float, enforce_integer: bool = True) -> float:
        if self.integer and enforce_integer:
            if abs(value - round(value)) > 1e-9:
                raise ValueError(f"{self.name} muss eine ganze Zahl sein")
            value = int(round(value))
        lo_ok = value > self.min_value if self.exclusive_min else value >= self.min_value
        hi_ok = value < self.max_value if self.exclusive_max else value <= self.max_value
        if not (lo_ok and hi_ok):
            raise ValueError(f"{self.name} liegt außerhalb des erlaubten Bereichs")
        return value


@dataclass
class Distribution:
    name: str
    discrete: bool
    params: list[ParamSpec]
    make_rv: Callable[..., object]
    default_x: float = 0.0
    category: str = ""
    info_text: str = ""

    def build(self, values: dict[str, float], enforce_integer: bool = True):
        kwargs = {p.name: p.validate(values[p.name], enforce_integer) for p in self.params}
        return self.make_rv(**kwargs)

    def support_range(self, rv, eps: float = 1e-4) -> tuple[float, float]:
        try:
            lo, hi = rv.ppf(eps), rv.ppf(1 - eps)
        except Exception:
            lo, hi = rv.support()
        if not np.isfinite(lo):
            lo = rv.ppf(0.001)
        if not np.isfinite(hi):
            hi = rv.ppf(0.999)
        return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank & rank-sum exact null distributions (custom DP)
# ---------------------------------------------------------------------------

def _signed_rank_pmf(n: int) -> np.ndarray:
    """PMF of W+ for Wilcoxon signed-rank, n ranks 1..n, support 0..n(n+1)/2."""
    smax = n * (n + 1) // 2
    counts = np.zeros(smax + 1, dtype=np.float64)
    counts[0] = 1.0
    for r in range(1, n + 1):
        new = counts.copy()
        new[r:] += counts[:smax + 1 - r]
        counts = new
    return counts / (2.0 ** n)


def _ranksum_pmf(n1: int, n2: int) -> np.ndarray:
    """PMF of U for Mann-Whitney, n1 vs n2, support 0..n1*n2."""
    umax = n1 * n2
    counts = np.zeros(umax + 1, dtype=np.float64)
    counts[0] = 1.0
    from math import comb
    total = comb(n1 + n2, n1)
    # Use DP: number of sequences in {0,1}^(n1+n2) with n1 ones giving sum-of-positions
    # Classical recursion: f(n1,n2,u) = f(n1-1,n2,u-n2) + f(n1,n2-1,u)
    f = np.zeros((n1 + 1, n2 + 1, umax + 1))
    f[0, :, 0] = 1.0
    f[:, 0, 0] = 1.0
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            f[i, j, :] = f[i, j - 1, :]
            shift = i
            if shift <= umax:
                f[i, j, shift:] += f[i - 1, j, :umax + 1 - shift]
    return f[n1, n2] / total


class _WilcoxonSR:
    def __init__(self, n: int):
        self.n = n
        self.pmf_array = _signed_rank_pmf(n)
        self.support_max = n * (n + 1) // 2
        self.cdf_array = np.cumsum(self.pmf_array)

    def pmf(self, k):
        k = np.asarray(k)
        out = np.zeros_like(k, dtype=float)
        mask = (k >= 0) & (k <= self.support_max) & (k == np.round(k))
        idx = k[mask].astype(int)
        out[mask] = self.pmf_array[idx]
        return out if out.shape else float(out)

    def cdf(self, k):
        k = np.asarray(k, dtype=float)
        out = np.zeros_like(k, dtype=float)
        k_clip = np.clip(np.floor(k), 0, self.support_max).astype(int)
        out = self.cdf_array[k_clip]
        out = np.where(k < 0, 0.0, out)
        out = np.where(k >= self.support_max, 1.0, out)
        return out if out.shape else float(out)

    def ppf(self, q):
        q = np.asarray(q, dtype=float)
        out = np.searchsorted(self.cdf_array, q, side="left").astype(float)
        return out if out.shape else float(out)

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        u = rng.random(size)
        return self.ppf(u).astype(int)

    def support(self):
        return (0, self.support_max)

    def mean(self):
        return self.n * (self.n + 1) / 4

    def var(self):
        return self.n * (self.n + 1) * (2 * self.n + 1) / 24

    def std(self):
        return math.sqrt(self.var())


class _MannWhitney:
    def __init__(self, n1: int, n2: int):
        self.n1, self.n2 = n1, n2
        self.pmf_array = _ranksum_pmf(n1, n2)
        self.support_max = n1 * n2
        self.cdf_array = np.cumsum(self.pmf_array)

    def pmf(self, k):
        k = np.asarray(k)
        out = np.zeros_like(k, dtype=float)
        mask = (k >= 0) & (k <= self.support_max) & (k == np.round(k))
        idx = k[mask].astype(int)
        out[mask] = self.pmf_array[idx]
        return out if out.shape else float(out)

    def cdf(self, k):
        k = np.asarray(k, dtype=float)
        k_clip = np.clip(np.floor(k), 0, self.support_max).astype(int)
        out = self.cdf_array[k_clip]
        out = np.where(k < 0, 0.0, out)
        out = np.where(k >= self.support_max, 1.0, out)
        return out if out.shape else float(out)

    def ppf(self, q):
        q = np.asarray(q, dtype=float)
        out = np.searchsorted(self.cdf_array, q, side="left").astype(float)
        return out if out.shape else float(out)

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        u = rng.random(size)
        return self.ppf(u).astype(int)

    def support(self):
        return (0, self.support_max)

    def mean(self):
        return self.n1 * self.n2 / 2

    def var(self):
        return self.n1 * self.n2 * (self.n1 + self.n2 + 1) / 12

    def std(self):
        return math.sqrt(self.var())


# ---------------------------------------------------------------------------
# Distribution registry
# ---------------------------------------------------------------------------

DISCRETE: list[Distribution] = [
    Distribution(
        "Bernoulli", True,
        [ParamSpec("p", "Probability of success", 0.5, min_value=0, max_value=1)],
        lambda p: stats.bernoulli(p),
        category="discrete",
        info_text="<b>What it is:</b> Describes a single event with exactly two outcomes (success/failure).<br><br><b>When to use:</b> Flipping a coin, or checking if a single manufactured part is defective.",
    ),
    Distribution(
        "Binomial", True,
        [ParamSpec("n", "Number of trials", 10, integer=True, min_value=1),
         ParamSpec("p", "Probability of success", 0.5, min_value=0, max_value=1)],
        lambda n, p: stats.binom(int(n), p),
        category="discrete",
        info_text="<b>What it is:</b> The number of successes in a fixed number of independent Yes/No trials.<br><br><b>When to use:</b> Finding out how many of 10 random people are left-handed, or how many products in a batch of 50 are defective.",
    ),
    Distribution(
        "Geometric", True,
        [ParamSpec("p", "Probability of success", 0.3, min_value=0, max_value=1, exclusive_min=True)],
        lambda p: stats.geom(p),
        category="discrete",
        info_text="<b>What it is:</b> The number of trials needed to get the first success.<br><br><b>When to use:</b> How many times you need to roll a die until you get a six, or how many customers enter a shop before the first purchase.",
    ),
    Distribution(
        "Hypergeometric", True,
        [ParamSpec("N", "Population size", 50, integer=True, min_value=1),
         ParamSpec("K", "Successes in population", 20, integer=True, min_value=0),
         ParamSpec("n", "Sample size", 10, integer=True, min_value=1)],
        lambda N, K, n: stats.hypergeom(int(N), int(K), int(n)),
        category="discrete",
        info_text="<b>What it is:</b> The number of successes when drawing without replacement from a finite population.<br><br><b>When to use:</b> Drawing cards from a deck, or selecting items from a lot for quality inspection without putting them back.",
    ),
    Distribution(
        "Negative Binomial", True,
        [ParamSpec("r", "Number of successes", 5, integer=True, min_value=1),
         ParamSpec("p", "Probability of success", 0.5, min_value=0, max_value=1, exclusive_min=True)],
        lambda r, p: stats.nbinom(int(r), p),
        category="discrete",
        info_text="<b>What it is:</b> The number of failures before achieving a specified number of successes.<br><br><b>When to use:</b> How many failed sales calls before closing 5 deals, or modeling overdispersed count data.",
    ),
    Distribution(
        "Poisson", True,
        [ParamSpec("λ", "Rate (Expected events)", 4.0, min_value=0, exclusive_min=True)],
        lambda **kw: stats.poisson(kw["λ"]),
        category="discrete",
        info_text="<b>What it is:</b> Models the number of events occurring randomly in a fixed interval of time or space.<br><br><b>When to use:</b> Counting the number of emails you receive per hour, or lightning strikes in a year.",
    ),
    Distribution(
        "Discrete Uniform", True,
        [ParamSpec("a", "Lower bound", 1, integer=True),
         ParamSpec("b", "Upper bound (incl.)", 6, integer=True)],
        lambda a, b: stats.randint(int(a), int(b) + 1),
        category="discrete",
        info_text="<b>What it is:</b> Every integer in the range [a, b] has exactly the same probability.<br><br><b>When to use:</b> Rolling a fair die (1–6), or randomly picking a locker number.",
    ),
    Distribution(
        "Logarithmic (Log-Series)", True,
        [ParamSpec("p", "Shape parameter", 0.5, min_value=0, max_value=1, exclusive_min=True, exclusive_max=True)],
        lambda p: stats.logser(p),
        category="discrete",
        info_text="<b>What it is:</b> A heavy-tailed discrete distribution where lower values are much more likely than higher ones.<br><br><b>When to use:</b> Modeling species abundance in ecology, or word frequency in text corpora.",
    ),
    Distribution(
        "Wilcoxon Signed-Rank", True,
        [ParamSpec("n", "Sample size", 10, integer=True, min_value=1, max_value=50)],
        lambda n: _WilcoxonSR(int(n)),
        category="discrete",
        info_text="<b>What it is:</b> The exact null distribution of the Wilcoxon signed-rank statistic W⁺.<br><br><b>When to use:</b> Non-parametric alternative to the paired t-test — testing whether the median difference of paired observations is zero.",
    ),
    Distribution(
        "Mann-Whitney U", True,
        [ParamSpec("n₁", "Group 1 size", 5, integer=True, min_value=1, max_value=20),
         ParamSpec("n₂", "Group 2 size", 5, integer=True, min_value=1, max_value=20)],
        lambda **kw: _MannWhitney(int(kw["n₁"]), int(kw["n₂"])),
        category="discrete",
        info_text="<b>What it is:</b> The exact null distribution of the Mann-Whitney U statistic.<br><br><b>When to use:</b> Non-parametric alternative to the two-sample t-test — comparing two independent groups without assuming normality.",
    ),
]


CONTINUOUS: list[Distribution] = [
    Distribution(
        "Normal", False,
        [ParamSpec("μ", "Mean", 0.0),
         ParamSpec("σ", "Standard deviation", 1.0, min_value=0, exclusive_min=True)],
        lambda **kw: stats.norm(loc=kw["μ"], scale=kw["σ"]),
        category="continuous",
        info_text="<b>What it is:</b> The famous Bell Curve! Describes symmetric data clustered around a mean.<br><br><b>When to use:</b> Modeling natural phenomena like human height, measurement errors, or IQ scores.",
    ),
    Distribution(
        "Standard Normal", False,
        [],
        lambda: stats.norm(loc=0, scale=1),
        category="continuous",
        info_text="<b>What it is:</b> A special case of the normal distribution with μ = 0 and σ = 1.<br><br><b>When to use:</b> Z-scores, standardized test statistics, and as the reference distribution for hypothesis testing.",
    ),
    Distribution(
        "Lognormal", False,
        [ParamSpec("μ", "Mean of ln X", 0.0),
         ParamSpec("σ", "SD of ln X", 1.0, min_value=0, exclusive_min=True)],
        lambda **kw: stats.lognorm(s=kw["σ"], scale=math.exp(kw["μ"])),
        category="continuous",
        info_text="<b>What it is:</b> Used for right-skewed data that cannot be negative. The logarithm of the data is normally distributed.<br><br><b>When to use:</b> Income distribution, stock prices, or file sizes.",
    ),
    Distribution(
        "Exponential", False,
        [ParamSpec("λ", "Rate", 1.0, min_value=0, exclusive_min=True)],
        lambda **kw: stats.expon(scale=1.0 / kw["λ"]),
        category="continuous",
        info_text="<b>What it is:</b> Models the duration or distance between two consecutive random events.<br><br><b>When to use:</b> The lifespan of electronic components, or waiting time in a queue.",
    ),
    Distribution(
        "Gamma", False,
        [ParamSpec("k", "Shape", 2.0, min_value=0, exclusive_min=True),
         ParamSpec("θ", "Scale", 1.0, min_value=0, exclusive_min=True)],
        lambda **kw: stats.gamma(a=kw["k"], scale=kw["θ"]),
        category="continuous",
        info_text="<b>What it is:</b> Generalizes the exponential distribution. The sum of k independent exponential random variables.<br><br><b>When to use:</b> Modeling waiting times for multiple events, rainfall amounts, or insurance claim sizes.",
    ),
    Distribution(
        "Beta", False,
        [ParamSpec("α", "Shape 1", 2.0, min_value=0, exclusive_min=True),
         ParamSpec("β", "Shape 2", 2.0, min_value=0, exclusive_min=True)],
        lambda **kw: stats.beta(a=kw["α"], b=kw["β"]),
        default_x=0.5,
        category="continuous",
        info_text="<b>What it is:</b> A flexible distribution on the interval [0, 1] that can take many different shapes.<br><br><b>When to use:</b> Modeling proportions, probabilities, or rates — e.g. conversion rates, batting averages, or Bayesian prior distributions.",
    ),
    Distribution(
        "Chi-Square", False,
        [ParamSpec("df", "Degrees of freedom", 5.0, min_value=0, exclusive_min=True)],
        lambda df: stats.chi2(df),
        category="continuous",
        info_text="<b>What it is:</b> The sum of squared standard normal variables.<br><br><b>When to use:</b> Extremely important in inferential statistics, like testing for independence or goodness of fit.",
    ),
    Distribution(
        "Noncentral Chi-Square", False,
        [ParamSpec("df", "Degrees of freedom", 5.0, min_value=0, exclusive_min=True),
         ParamSpec("λ", "Noncentrality", 2.0, min_value=0)],
        lambda **kw: stats.ncx2(df=kw["df"], nc=kw["λ"]),
        category="continuous",
        info_text="<b>What it is:</b> Generalizes the chi-square distribution when the underlying normals have non-zero means.<br><br><b>When to use:</b> Power analysis for chi-square tests, or when testing hypotheses where the null is not exactly true.",
    ),
    Distribution(
        "Student-t", False,
        [ParamSpec("df", "Degrees of freedom", 10.0, min_value=0, exclusive_min=True)],
        lambda df: stats.t(df),
        category="continuous",
        info_text="<b>What it is:</b> Similar to the normal distribution but with 'heavier tails'.<br><br><b>When to use:</b> When sample sizes are small (n < 30) and the true population standard deviation is unknown (t-Test).",
    ),
    Distribution(
        "Noncentral t", False,
        [ParamSpec("df", "Degrees of freedom", 10.0, min_value=0, exclusive_min=True),
         ParamSpec("δ", "Noncentrality", 1.0)],
        lambda **kw: stats.nct(df=kw["df"], nc=kw["δ"]),
        category="continuous",
        info_text="<b>What it is:</b> Generalizes the Student-t distribution when the true mean differs from the hypothesized value.<br><br><b>When to use:</b> Power analysis and sample size planning for t-tests.",
    ),
    Distribution(
        "F-Distribution", False,
        [ParamSpec("df₁", "Numerator df", 5.0, min_value=0, exclusive_min=True),
         ParamSpec("df₂", "Denominator df", 10.0, min_value=0, exclusive_min=True)],
        lambda **kw: stats.f(dfn=kw["df₁"], dfd=kw["df₂"]),
        default_x=1.0,
        category="continuous",
        info_text="<b>What it is:</b> The ratio of two independent chi-square variables divided by their degrees of freedom.<br><br><b>When to use:</b> ANOVA tests for comparing group means, or testing whether two population variances are equal.",
    ),
    Distribution(
        "Noncentral F", False,
        [ParamSpec("df₁", "Numerator df", 5.0, min_value=0, exclusive_min=True),
         ParamSpec("df₂", "Denominator df", 10.0, min_value=0, exclusive_min=True),
         ParamSpec("λ", "Noncentrality", 2.0, min_value=0)],
        lambda **kw: stats.ncf(dfn=kw["df₁"], dfd=kw["df₂"], nc=kw["λ"]),
        default_x=1.0,
        category="continuous",
        info_text="<b>What it is:</b> Generalizes the F-distribution when group means truly differ.<br><br><b>When to use:</b> Power analysis for ANOVA and regression F-tests.",
    ),
    Distribution(
        "Continuous Uniform", False,
        [ParamSpec("a", "Lower bound", 0.0),
         ParamSpec("b", "Upper bound", 1.0)],
        lambda a, b: stats.uniform(loc=a, scale=b - a),
        default_x=0.5,
        category="continuous",
        info_text="<b>What it is:</b> Every value in the interval [a, b] is equally likely.<br><br><b>When to use:</b> Random number generation, modeling complete ignorance, or rounding errors.",
    ),
    Distribution(
        "Cauchy", False,
        [ParamSpec("x₀", "Location", 0.0),
         ParamSpec("γ", "Scale", 1.0, min_value=0, exclusive_min=True)],
        lambda **kw: stats.cauchy(loc=kw["x₀"], scale=kw["γ"]),
        category="continuous",
        info_text="<b>What it is:</b> A bell-shaped distribution with extremely heavy tails — it has no defined mean or variance!<br><br><b>When to use:</b> Modeling phenomena with frequent extreme outliers, or as a cautionary example in statistics.",
    ),
    Distribution(
        "Weibull", False,
        [ParamSpec("k", "Shape", 1.5, min_value=0, exclusive_min=True),
         ParamSpec("λ", "Scale", 1.0, min_value=0, exclusive_min=True)],
        lambda **kw: stats.weibull_min(c=kw["k"], scale=kw["λ"]),
        default_x=1.0,
        category="continuous",
        info_text="<b>What it is:</b> A flexible lifetime distribution. Shape k controls the failure rate: increasing (k > 1), constant (k = 1), or decreasing (k < 1).<br><br><b>When to use:</b> Reliability engineering, survival analysis, or modeling wind speeds.",
    ),
    Distribution(
        "Logistic", False,
        [ParamSpec("μ", "Location", 0.0),
         ParamSpec("s", "Scale", 1.0, min_value=0, exclusive_min=True)],
        lambda **kw: stats.logistic(loc=kw["μ"], scale=kw["s"]),
        category="continuous",
        info_text="<b>What it is:</b> Similar shape to the normal distribution but with heavier tails. Its CDF is the logistic function.<br><br><b>When to use:</b> Logistic regression, modeling growth curves, or as an alternative to the normal when tails matter more.",
    ),
    Distribution(
        "Laplace", False,
        [ParamSpec("μ", "Location", 0.0),
         ParamSpec("b", "Scale", 1.0, min_value=0, exclusive_min=True)],
        lambda **kw: stats.laplace(loc=kw["μ"], scale=kw["b"]),
        category="continuous",
        info_text="<b>What it is:</b> Also called the 'double exponential'. Looks like a normal distribution with a sharper peak and heavier tails.<br><br><b>When to use:</b> Signal processing, modeling financial returns, or as a Bayesian prior (L1 regularization / Lasso).",
    ),
    Distribution(
        "Pareto", False,
        [ParamSpec("α", "Shape", 3.0, min_value=0, exclusive_min=True),
         ParamSpec("xₘ", "Minimum", 1.0, min_value=0, exclusive_min=True)],
        lambda **kw: stats.pareto(b=kw["α"], scale=kw["xₘ"]),
        default_x=2.0,
        category="continuous",
        info_text="<b>What it is:</b> A power-law distribution — the basis of the '80/20 rule'.<br><br><b>When to use:</b> Wealth distribution, city sizes, file sizes on the internet, or any 'the rich get richer' phenomenon.",
    ),
    Distribution(
        "Triangular", False,
        [ParamSpec("a", "Min", 0.0),
         ParamSpec("c", "Mode", 0.5),
         ParamSpec("b", "Max", 1.0)],
        lambda a, c, b: stats.triang(c=(c - a) / (b - a), loc=a, scale=b - a),
        default_x=0.5,
        category="continuous",
        info_text="<b>What it is:</b> Defined by a minimum, maximum, and most likely value (mode). The PDF forms a triangle.<br><br><b>When to use:</b> Quick estimates when only rough bounds and a best guess are known — common in project management (PERT) and risk analysis.",
    ),
    Distribution(
        "Gumbel", False,
        [ParamSpec("μ", "Location", 0.0),
         ParamSpec("β", "Scale", 1.0, min_value=0, exclusive_min=True)],
        lambda **kw: stats.gumbel_r(loc=kw["μ"], scale=kw["β"]),
        category="continuous",
        info_text="<b>What it is:</b> An extreme value distribution — models the maximum (or minimum) of many samples.<br><br><b>When to use:</b> Predicting floods, extreme temperatures, or maximum earthquake magnitudes.",
    ),
    Distribution(
        "Rayleigh", False,
        [ParamSpec("σ", "Scale", 1.0, min_value=0, exclusive_min=True)],
        lambda **kw: stats.rayleigh(scale=kw["σ"]),
        default_x=1.0,
        category="continuous",
        info_text="<b>What it is:</b> The distribution of the distance from the origin of a 2D point with independent normal coordinates.<br><br><b>When to use:</b> Wind speed modeling, wireless signal fading, or the magnitude of 2D random vectors.",
    ),
    Distribution(
        "Inverse Gaussian", False,
        [ParamSpec("μ", "Mean", 1.0, min_value=0, exclusive_min=True),
         ParamSpec("λ", "Shape", 1.0, min_value=0, exclusive_min=True)],
        lambda **kw: stats.invgauss(mu=kw["μ"] / kw["λ"], scale=kw["λ"]),
        default_x=1.0,
        category="continuous",
        info_text="<b>What it is:</b> Models the first passage time of a Brownian motion with drift to a fixed boundary.<br><br><b>When to use:</b> Modeling reaction times, repair times, or any right-skewed positive data with a natural lower bound.",
    ),
]


ALL_DISTRIBUTIONS: list[Distribution] = DISCRETE + CONTINUOUS
BY_NAME: dict[str, Distribution] = {d.name: d for d in ALL_DISTRIBUTIONS}


assert len(DISCRETE) == 10, f"expected 10 discrete, got {len(DISCRETE)}"
assert len(CONTINUOUS) == 22, f"expected 22 continuous, got {len(CONTINUOUS)}"
