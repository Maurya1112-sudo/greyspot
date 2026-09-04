"""GAT + GRU research model: the graph-temporal comparator the dossier's core
research question is actually about (does connected-road-network modelling
improve on strong non-graph baselines?).

Architecture, matching the dossier's Section 9 description and its closest
academic precedent (Gao, Jiang, Haworth et al. 2024, arXiv:2309.05072,
STZITD-GNN): for each year, a Graph Attention layer aggregates each
segment's own features with its line-graph neighbours' features (see
`features.graph_temporal`); a GRU then consumes the resulting per-year
embeddings in sequence to capture temporal evolution; a small decoder head
turns the final hidden state into a non-negative predicted collision count.

**Regularisation** (dropout, weight decay) was added on 2026-08-31 after
reading the precedent paper's actual training setup (3 attention heads, 42
hidden units, dropout 0.2, weight decay 0.01, 20 epochs with early-stopping
patience 10 — extracted from the arXiv HTML full text): the first
(unregularised, 200-epoch) version of this model was consistently beaten by
its own "remove the graph" ablation on precision metrics, and an
attention-parameterised branch overfitting harder than a plain linear one
on ~7,500 training nodes is a plausible, testable explanation. See
`docs/decision_log.md` for the before/after comparison.

This is still intentionally the plain, defensible version relative to the
precedent: it is NOT the full four-parameter Zero-Inflated Tweedie decoder
Gao et al. use (their loss models the probability mass at exactly zero
separately from the continuous positive tail) - Poisson NLL is a documented
simplification, not a hidden shortcut. Interestingly, the precedent paper
itself does not include a graph-vs-no-graph or temporal-vs-no-temporal
ablation (confirmed via the same reading) - this project's ablation is a
genuine addition beyond the published literature, not a replication of it.

`use_graph` / `use_temporal` implement the dossier's minimum ablation plan
(Section 9: "remove graph structure; remove the temporal encoder ... compare
against XGBoost") as two independent switches on the *same* class, so an
ablation changes exactly one component and nothing else:
  - use_graph=False replaces the GATConv with a plain per-node Linear layer
    of the same output width - i.e. each segment sees only its own features,
    no neighbours. This isolates what the graph structure contributes.
  - use_temporal=False skips the GRU and decodes directly from the *last*
    timestep's embedding - i.e. no memory of earlier years. This isolates
    what the temporal encoder contributes.

**Zero-inflated Poisson decoder** (`zero_inflated`, added 2026-08-31): the
module has said since its first version that plain Poisson NLL is "NOT the
full four-parameter Zero-Inflated Tweedie decoder Gao et al. use" -
research (arXiv, PyTorch-forum and applied-stats searches, 2026-08-31)
confirms zero-inflated count models are exactly the standard tool for this
failure mode (a mixture of a Bernoulli "structural zero" gate and a
Poisson/negative-binomial component for the positive counts), and that a
full Tweedie decoder is a real engineering step up in complexity from a
2-parameter Zero-Inflated Poisson (ZIP) one. `zero_inflated=True` adds a
second linear head (`zero_gate`) alongside the existing rate decoder;
`zero_inflated_poisson_nll` implements the mixture log-likelihood
directly (no off-the-shelf PyTorch loss exists for this - confirmed by
the same research pass). This is a documented, isolated step toward the
precedent's decoder, not a full replication of its 4-parameter Tweedie
formulation.

**Residual/skip connection** (`use_residual`, added 2026-08-31): the
`gat_no_graph` ablation beating the full model across six independent runs
(see docs/decision_log.md) is consistent with the graph-attention step
over-smoothing away a segment's own distinguishing signal - averaging it
with ~7-8 line-graph neighbours' features dilutes exactly the information
that makes a genuinely high-risk segment stand out. Residual/skip
connections are the standard fix for this in the GNN literature (e.g.
GCNII, JKNet): let the model see the segment's own raw features *alongside*
the graph-aggregated ones, rather than being forced to replace one with
the other. When `use_residual=True` and `use_graph=True`, the raw input
features are projected to the same width as the GAT output and added
before the nonlinearity - a clean, minimal, single-variable test of the
over-smoothing hypothesis, not a general architecture rewrite.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

logger = logging.getLogger(__name__)


def zero_inflated_poisson_nll(
    lam: torch.Tensor, pi: torch.Tensor, y: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Negative log-likelihood of a Zero-Inflated Poisson mixture.

    `pi` is the probability of a "structural zero" (a segment that simply
    cannot have a collision this period, distinct from a segment that could
    but didn't); `lam` is the Poisson rate for the remaining process. No
    off-the-shelf PyTorch loss implements this (confirmed 2026-08-31 -
    PyTorch only ships plain `PoissonNLLLoss`), so it's implemented
    directly rather than approximated.

    y == 0:  -log( pi + (1-pi) * exp(-lam) )            [either process gives zero]
    y  > 0:  -log(1-pi) + lam - y*log(lam) + lgamma(y+1)  [must come from the Poisson part]
    """
    is_zero = (y == 0).float()
    lam = lam.clamp(min=eps)
    pi = pi.clamp(min=eps, max=1 - eps)

    # y == 0 term via a numerically stable log-sum-exp of the two log-probabilities
    log_pi = torch.log(pi)
    log_not_pi_poisson_zero = torch.log(1 - pi) - lam
    zero_term = -torch.logsumexp(torch.stack([log_pi, log_not_pi_poisson_zero]), dim=0)

    # y > 0 term: standard Poisson log-likelihood, gated by (1 - pi)
    positive_term = -torch.log(1 - pi) - y * torch.log(lam) + lam + torch.lgamma(y + 1)

    nll = is_zero * zero_term + (1 - is_zero) * positive_term
    return nll.mean()


def zero_inflated_negative_binomial_nll(
    mu: torch.Tensor, pi: torch.Tensor, r: torch.Tensor, y: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Negative log-likelihood of a Zero-Inflated Negative Binomial (ZINB)
    mixture - NB2 parameterisation (mean `mu`, dispersion `r`; variance =
    mu + mu^2/r, recovering Poisson as r -> infinity).

    Added 2026-09-01, reading Gao et al. (2024)'s full Methodology section
    (arXiv:2309.05072v4, Eq. 3-12, Appendix A/B) after the user asked to
    follow the paper's approach directly. Their own decoder is a
    Zero-Inflated Tweedie (ZITD) - a compound Poisson-Gamma distribution
    (crash *count* ~ Poisson, per-crash *severity* ~ Gamma, summed) with a
    learned index parameter rho in (1,2), because their target `y` is
    explicitly a severity-weighted composite ("the crash value applied to
    both crash counts and the associated severity", their Section 3.1) -
    a genuinely continuous, mixed discrete/continuous quantity that needs
    a compound distribution to represent.

    This project's own target, `collision_count`, is a plain integer count
    (a disclosed, structural difference from the paper's target - see
    `docs/publication_readiness.md`) - Tweedie's Gamma-severity component
    has no referent to fit here, since there is no severity being
    aggregated into the target at all. Adopting ZINB instead of ZITD is a
    deliberate, disclosed choice: it targets the *same underlying
    weakness* the paper's move from plain Poisson to Tweedie addresses
    (a fixed mean=variance assumption is wrong for over-dispersed,
    zero-inflated crash counts - `zero_inflated_poisson_nll` above shares
    this same flaw, since Poisson's variance is pinned to its mean) while
    remaining the architecturally correct distribution family for a count
    target rather than importing a distribution shaped for a different
    target just to match the paper's own choice superficially. Both `pi`
    (structural-zero gate) and `r` (learned per-row dispersion, not a
    fixed hyperparameter) are genuine model outputs, matching the paper's
    own principle of learning every distribution parameter from the
    spatiotemporal embedding rather than fixing any of them by hand.

    NB2 log-pmf: log P(y|mu,r) = lgamma(y+r) - lgamma(r) - lgamma(y+1)
                                  + r*log(r/(r+mu)) + y*log(mu/(r+mu))
    """
    is_zero = (y == 0).float()
    mu = mu.clamp(min=eps)
    r = r.clamp(min=eps)
    pi = pi.clamp(min=eps, max=1 - eps)

    log_r_ratio = r * (torch.log(r) - torch.log(r + mu))  # r*log(r/(r+mu)), stable form

    # y == 0: either the structural-zero gate fired, or the NB component
    # itself produced a zero - logsumexp of the two log-probabilities,
    # exactly mirroring zero_inflated_poisson_nll's own zero-term above.
    log_pi = torch.log(pi)
    log_not_pi_nb_zero = torch.log(1 - pi) + log_r_ratio
    zero_term = -torch.logsumexp(torch.stack([log_pi, log_not_pi_nb_zero]), dim=0)

    # y > 0: standard NB2 log-likelihood, gated by (1 - pi) - must come
    # from the count process, since the structural-zero gate can only
    # ever produce a zero.
    log_nb_positive = (
        torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1)
        + log_r_ratio + y * (torch.log(mu) - torch.log(r + mu))
    )
    positive_term = -torch.log(1 - pi) - log_nb_positive

    nll = is_zero * zero_term + (1 - is_zero) * positive_term
    return nll.mean()


def zero_inflated_tweedie_nll(
    mu: torch.Tensor, pi: torch.Tensor, phi: torch.Tensor, y: torch.Tensor, rho: float = 1.5, eps: float = 1e-8
) -> torch.Tensor:
    """Negative log-likelihood of the Zero-Inflated Tweedie (ZITD) mixture -
    Gao et al. (2024)'s OWN reported decoder, Eq. 5-6 of the paper (text
    verified directly from the PDF, 2026-09-01, not reconstructed from
    memory - see docs/decision_log.md), added after `zero_inflated_negative_binomial_nll`
    above was deliberately chosen as a *disclosed substitute* for this
    project's plain-count target. This function is the real thing, for
    use once a severity-weighted target (`tcr_score`,
    `daily_features.TCR_SEVERITY_WEIGHT`) makes a compound Poisson-Gamma
    decoder architecturally appropriate again.

    The paper's own Eq. 3-6, quoted exactly: a Tweedie-distributed random
    variable has density `f_TD(y|theta,phi) = a(y,phi) * exp[(y*theta -
    kappa(theta))/phi]` (an exponential dispersion model), with mean
    `mu = kappa'(theta)` and variance `phi * mu^rho` - `rho` in (1,2)
    gives the "Compound Poisson-Gamma" special case the paper uses for
    crash risk. Their ZITD mixture (Eq. 5-6):
        y = 0       w.p. pi + (1-pi)*f_TD(0|mu,phi,rho)
        y = Y > 0   w.p. (1-pi)*f_TD(Y|mu,phi,rho)
    and their own Eq. 6 gives the EXACT y=0 mass in closed form:
        f_TD(0|mu,phi,rho) = exp(-mu^(2-rho) / (phi*(2-rho)))
    (this is the well-known compound-Poisson "no claims occurred"
    probability, exp(-lambda) with lambda = mu^(2-rho)/(phi*(2-rho)) -
    consistent with, not just asserted by, the paper's own text).

    **For y > 0, ported directly from the paper's own linked reference
    code** (2026-09-02: the paper states "our code is available on
    Github" at github.com/STTDAnonymous/STTD, verified by fetching the
    paper's own HTML/PDF text, not assumed - see docs/decision_log.md;
    the repo's `Accident_risk` folder is an empty stub, confirmed a dead
    end for this project's actual task, but its `utils.py::tweedie_nll_loss`
    is a genuine, MIT-licensed, differentiable Tweedie NLL that the SAME
    authors wrote for a sibling paper - a legitimate reference for the
    identical loss family). `f_TD(y>0|...)` itself has no closed form -
    Dunn & Smyth (2005) express it as an infinite series over Poisson-
    count terms with rho-dependent Gamma functions - so their code
    approximates `log(a(y,phi,rho))` (the intractable normalising
    constant) with the single dominant term of that series (a
    saddlepoint-style approximation, evaluated at the series' own peak
    index `j_max`), rather than dropping it entirely:
        alpha = (2-rho)/(1-rho)                      [always < 0 for 1<rho<2]
        j_max = y^(2-rho) / ((2-rho)*phi)
        log_a_approx = -log(y) + j_max*(alpha-1) - log(j_max) - 0.5*log(-alpha)
        -log f_TD(y|mu,phi,rho) ~= -y*mu^(1-rho)/(phi*(1-rho))
                                     + mu^(2-rho)/(phi*(2-rho)) + log_a_approx
    Verified independently before adopting it: this project's own earlier
    from-scratch derivation (the standard GLM "Tweedie unit deviance,"
    dropping the mu-independent normalising term entirely - McCullagh &
    Nelder) produces the exact SAME mu-dependent terms
    (`-y*mu^(1-rho)/(phi*(1-rho)) + mu^(2-rho)/(phi*(2-rho))`) as this
    reference code - independent confirmation the derivation is correct,
    with the reference code's `log_a_approx` term simply being a more
    faithful (paper-authors'-own-choice) stand-in for the piece the
    from-scratch deviance approach discarded. Adopting their version
    gives `phi` a real (if still approximate) gradient signal from y>0
    rows too, which the discarded-term version could not.

    `rho` (added as a plain float hyperparameter here, NOT a 4th learned
    per-row output unlike the paper's own architecture) - a genuinely
    differentiable, LEARNED-rho version of the series above is a
    substantially larger, riskier undertaking than this pass attempts;
    disclosed as a real, scoped difference from the paper's exact
    4-parameter design, not silently matched. Default 1.5 sits at the
    midpoint of the paper's own valid (1,2) compound Poisson-Gamma range.
    """
    if not 1.0 < rho < 2.0:
        raise ValueError(f"zero_inflated_tweedie_nll: rho must be in (1, 2) for the compound Poisson-Gamma case, got {rho}.")
    is_zero = (y == 0).float()
    mu = mu.clamp(min=eps)
    phi = phi.clamp(min=eps)
    pi = pi.clamp(min=eps, max=1 - eps)
    p1, p2 = 1.0 - rho, 2.0 - rho  # p1 < 0, p2 > 0 for 1 < rho < 2

    log_p_zero_td = -mu.pow(p2) / (phi * p2)  # exact, from the paper's own Eq. 6

    # y == 0: either the zero-inflation gate fired, or the Tweedie
    # component itself landed on its (real, positive-probability) zero
    # atom - logsumexp of the two log-probabilities, mirroring
    # zero_inflated_poisson_nll/zero_inflated_negative_binomial_nll's own
    # zero-term above exactly.
    log_pi = torch.log(pi)
    log_not_pi_td_zero = torch.log(1 - pi) + log_p_zero_td
    zero_term = -torch.logsumexp(torch.stack([log_pi, log_not_pi_td_zero]), dim=0)

    # y > 0: reference-code Tweedie log-likelihood approximation (see
    # docstring - ported from the paper's own linked repo), gated by
    # (1 - pi) - must come from the Tweedie component, since the
    # zero-inflation gate can only ever produce zero. `y` is clamped
    # here (not above, alongside mu/phi/pi) specifically because `is_zero
    # * zero_term + (1 - is_zero) * positive_term` below evaluates BOTH
    # terms for every element before masking - `positive_term` must stay
    # finite even at y=0 (where log(y) would otherwise be -inf) so that
    # `0 * positive_term` is exactly 0, not `0 * -inf = NaN`.
    y_safe = y.clamp(min=eps)
    alpha = p2 / p1  # always < 0 for 1 < rho < 2 (p2 > 0, p1 < 0) - a plain
    # float, since rho is a fixed hyperparameter here, not a per-element
    # tensor - math.log (not torch.log) is correct and simpler for it.
    log_neg_alpha = math.log(-alpha + eps)
    j_max = y_safe.pow(p2) / (p2 * phi)
    log_a_approx = -torch.log(y_safe) + j_max * (alpha - 1) - torch.log(j_max.clamp(min=eps)) - 0.5 * log_neg_alpha
    log_f_td_positive = y_safe * mu.pow(p1) / (phi * p1) - mu.pow(p2) / (phi * p2) - log_a_approx
    positive_term = -torch.log(1 - pi) - log_f_td_positive

    nll = is_zero * zero_term + (1 - is_zero) * positive_term
    return nll.mean()


def top_k_hinge_loss(scores: torch.Tensor, y: torch.Tensor, top_fraction: float = 0.20,
                     margin: float = 0.1) -> torch.Tensor:
    """A TOP-K hinge loss that targets AccHR@20's actual decision boundary,
    added 2026-09-03 from the learning-to-rank literature on metric-aware
    losses (LambdaRank's core idea: weight by the change in the metric a
    swap causes, rather than treating all pairs alike; and the top-k hinge
    family for deep imbalanced classification).

    **Why the existing `pairwise_rank_hinge_loss` is poorly matched to
    this metric.** It penalises every (positive, negative) pair equally.
    On this data a single day has roughly 2 positives and ~11,594
    negatives, so almost all of the ~23,000 pairs it forms involve a
    negative sitting deep in the bottom 80% of scores - pairs that can
    never change whether the positive lands inside the top 20%. The
    gradient is therefore dominated by comparisons the evaluation metric
    is completely indifferent to. Tested 2026-09-01 and it hurt
    (48.29% at weight 0.1, 40.76% at weight 1.0, against a ~50% baseline).

    **What AccHR@20 actually requires**: for each day, every segment with
    a real crash must score at or above the k-th highest predicted score,
    where k = round(N * top_fraction). Nothing else matters - not the
    margin over the median segment, not the ordering within the bottom
    80%. This loss encodes exactly that: for each positive, penalise
    `max(0, margin - (score_positive - kth_highest_score))`.

    Gradient flows to the positive's own score and, through
    `torch.kthvalue`, to the single segment currently sitting on the
    boundary - the same mechanism by which max-pooling propagates
    gradient, and the reason this stays differentiable despite encoding a
    hard top-k rule.

    `scores` and `y` are `[N]` for one day, matching
    `eval/ucl_metrics.accuracy_hit_rate`'s per-day definition exactly so
    the training objective and the evaluation metric operate on the
    identical unit.

    Returns 0 when a day has no positives - a ranking objective is
    undefined without them, and at 99.98% sparsity such days are common.
    """
    positives = y > 0
    n_pos = int(positives.sum().item())
    if n_pos == 0:
        return scores.sum() * 0.0  # keeps the graph connected, contributes nothing

    n = scores.shape[0]
    k = max(1, int(round(n * top_fraction)))
    # k-th HIGHEST score = the boundary AccHR@20 thresholds on
    boundary = torch.kthvalue(scores, n - k + 1).values

    shortfall = margin - (scores[positives] - boundary)
    return torch.clamp(shortfall, min=0.0).mean()


def pairwise_rank_hinge_loss(scores: torch.Tensor, y: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """A learning-to-rank auxiliary loss, added 2026-09-01 after this
    project's own AccHR@20 sweep (see docs/decision_log.md) found real,
    substantial architecture effects (heads=1 vs 3, correlation as low as
    0.67 between models) that only partly moved the ranking metric - a
    plausible reason why: this project's models are all trained with a
    pure distributional NLL (`zero_inflated_poisson_nll` /
    `zero_inflated_negative_binomial_nll`), which optimises "fit the
    Poisson/NB distribution well," not "rank the roads that actually crash
    above the ones that don't" - a well-known gap in the learning-to-rank
    literature (a good density fit under extreme class imbalance does not
    guarantee good top-k ranking, since the NLL is dominated by correctly
    predicting the overwhelming majority of true zeros, not by getting the
    relative order of the rare positives right).

    This is a standard pairwise hinge ranking loss (in the spirit of BPR -
    Rendle et al. 2009 - and RankSVM's hinge formulation): for every
    (positive, negative) pair on the same day - a segment that actually
    had a crash `y>0` and one that didn't `y==0` - the model's *point
    estimate* score for the positive should exceed the negative's by at
    least `margin`; violations are penalised linearly. This directly
    targets what AccHR@20 measures (does a true crash's segment score
    above the non-crash segments' scores, not by how much), unlike NLL,
    which targets the full likelihood.

    `scores` and `y` are both `[N]` (a single day/step) - the caller loops
    over each column of a `[N, horizon]` target if computing this across a
    multi-step target, matching this project's per-day AccHR@20 definition
    (`eval/ucl_metrics.accuracy_hit_rate`) exactly, so the auxiliary loss
    optimises the identical unit the eval metric scores.

    Returns 0 (not NaN) when a day has no positives or no negatives, since
    a ranking loss is undefined without both classes present that day -
    common at this data's sparsity (see docs/publication_readiness.md's
    zero-inflation figures), so this must not crash the training loop.
    """
    is_pos = y > 0
    if is_pos.sum() == 0 or (~is_pos).sum() == 0:
        return torch.zeros((), device=scores.device, dtype=scores.dtype)
    pos_scores = scores[is_pos]  # [P]
    neg_scores = scores[~is_pos]  # [Ng]
    diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)  # [P, Ng]
    return torch.relu(margin - diff).mean()


class GATTemporal(nn.Module):
    def __init__(
        self,
        in_channels: int,
        gat_hidden: int = 16,
        heads: int = 4,
        gru_hidden: int = 32,
        use_graph: bool = True,
        use_temporal: bool = True,
        dropout: float = 0.0,
        use_residual: bool = False,
        zero_inflated: bool = False,
        negative_binomial: bool = False,
        tweedie: bool = False,
        tweedie_rho: float = 1.5,
        gat_layers: int = 1,
        horizon: int = 1,
        rate_link: str = "softplus",
        encoder_order: str = "spatial_first",
    ):
        """`horizon`: number of future steps predicted in one forward pass
        (added 2026-09-01 for the UCL-comparable daily/14-day-multi-step
        evaluation - docs/publication_readiness.md). Defaults to 1, the
        original single-step behaviour every existing caller (the annual
        pipeline) already relies on - `forward()` still returns a plain
        `[N]` tensor (or `(lam, pi)` pair of `[N]` tensors) when
        `horizon == 1`, exactly as before. `horizon > 1` is a genuine
        "direct multi-horizon" decoder (the paper's own p=14 forecast
        window, see `features/daily_temporal.py`) - one linear head
        predicting all `horizon` future steps from the same final hidden
        state, not `horizon` separately-trained models or a recursive
        feedback loop.

        `negative_binomial` (added 2026-09-01, reading Gao et al. (2024)'s
        Methodology in full): a genuine second decoder distribution
        alongside the existing Zero-Inflated Poisson, using
        `zero_inflated_negative_binomial_nll` - see that function's
        docstring for why ZINB, not the paper's own Zero-Inflated Tweedie,
        is the honest distributional upgrade for this project's plain-count
        target. Adds a third decoder head (`dispersion_gate`, softplus) for
        the learned per-row dispersion `r`, alongside the existing rate/
        zero-gate heads. Implies zero-inflation regardless of the
        `zero_inflated` flag (every real crash-count series here is
        zero-inflated; there is no "plain NB, no zero-gate" mode - matching
        the paper's own ZITD, which never separates the zero-inflation gate
        from the count distribution either).

        `gat_layers` (added 2026-09-01, matching the paper's stated "GNNs
        in STZITD-GNNs and baselines are all two-layered", Section 4.2):
        this project's own earlier over-smoothing investigation
        (`docs/decision_log.md`, 2026-08-31) found 1 head + a residual
        connection beat deeper/wider configurations on the *annual*
        line-graph topology, at annual grain (~7,500 nodes, only 2-3
        walk-forward transitions) - `gat_layers` defaults to 1 (that
        finding's configuration, unchanged for every existing caller) and
        is exposed as a real, testable option now that the daily pipeline
        has a much larger walk-forward instance count (128, not 2-3),
        rather than assuming the earlier finding transfers to a very
        different data regime unexamined.

        `tweedie` (added 2026-09-01, after directly re-reading and
        transcribing the paper's own Eq. 3-6 from its PDF text, not from
        memory - see `zero_inflated_tweedie_nll`'s own docstring for the
        full derivation and its disclosed deviance-based approximation
        for y > 0): the paper's OWN reported third decoder option
        alongside the existing Zero-Inflated Poisson/ZINB alternatives
        above, for use with a severity-weighted target (`tcr_score`) where
        a compound Poisson-Gamma decoder is architecturally appropriate
        (unlike this project's plain-count target, where ZINB was chosen
        instead - see that function's docstring). Adds a third decoder
        head (`phi_gate`, softplus) for the learned per-row dispersion
        `phi`; `tweedie_rho` (the paper's index parameter) is a FIXED
        hyperparameter here, not a 4th learned output like the paper's
        own architecture - a genuinely differentiable learned-rho version
        is a substantially larger undertaking than this pass attempts,
        disclosed as a real, scoped difference. Mutually exclusive with
        `negative_binomial` (both are alternative third-head choices for
        the same decoder slot) - passing both raises `ValueError`.

        `rate_link` (added 2026-09-02, re-reading the paper's own linked
        reference repo a second time at the user's request to "connect
        every dot"): controls the activation applied to the rate/mean
        (`lam`/`mu`) decoder head. `"softplus"` (the default, unchanged
        from every prior caller) was this project's own original choice.
        `"exp"` matches what the reference code
        (github.com/STTDAnonymous/STTD's `model.py`/`utils.py`) actually
        does for its own mu parameter: the model's raw linear output gets
        NO activation at all, and `mu = torch.exp(mu)` is applied only
        inside the loss function (a log-link, the standard convention for
        Poisson/Tweedie GLMs) - genuinely different from softplus's
        near-linear growth for large inputs. Offered here as a real,
        disclosed alternative to test, not assumed better a priori;
        clamped at +-15 before `exp()` for numerical safety (a standard
        log-link precaution, not paper-specific)."""
        super().__init__()
        if negative_binomial and tweedie:
            raise ValueError("GATTemporal: negative_binomial and tweedie are alternative third decoder heads - set at most one.")
        if rate_link not in ("softplus", "exp"):
            raise ValueError(f"GATTemporal: rate_link must be 'softplus' or 'exp', got {rate_link!r}.")
        self.use_graph = use_graph
        self.use_temporal = use_temporal
        self.use_residual = use_residual and use_graph  # meaningless without the graph branch
        self.zero_inflated = zero_inflated or negative_binomial or tweedie
        self.negative_binomial = negative_binomial
        self.tweedie = tweedie
        self.tweedie_rho = tweedie_rho
        self.gat_layers = max(1, gat_layers)
        self.horizon = horizon
        self.rate_link = rate_link
        if encoder_order not in ("spatial_first", "temporal_first"):
            raise ValueError(
                f"encoder_order must be 'spatial_first' or 'temporal_first', got {encoder_order!r}"
            )
        # "spatial_first" (default, this project's original): GAT at every
        # timestep, then GRU over those embeddings. "temporal_first": the
        # PAPER's own order (thesis Appendix A.3/A.4) - GRU over the raw
        # feature sequence, then a single GAT pass over the temporal
        # embedding. Kept switchable rather than replaced, so the original
        # behaviour every existing caller/test relies on is untouched.
        self.encoder_order = encoder_order
        spatial_out = gat_hidden * heads

        def make_spatial_layer(in_dim: int) -> nn.Module:
            return GATConv(in_dim, gat_hidden, heads=heads, concat=True, dropout=dropout) if use_graph \
                else nn.Linear(in_dim, spatial_out)

        # In the paper's own "temporal_first" order the GAT consumes the
        # GRU's output (width `gru_hidden`), not the raw features - so the
        # first spatial layer's input width differs between the two orders.
        first_spatial_in = gru_hidden if self.encoder_order == "temporal_first" else in_channels
        self.spatial_layers = nn.ModuleList(
            [make_spatial_layer(first_spatial_in if i == 0 else spatial_out) for i in range(self.gat_layers)]
        )

        if self.use_residual:
            self.residual_proj = nn.Linear(in_channels, spatial_out)
            if self.encoder_order == "temporal_first":
                # residual from the temporal embedding, not the raw features
                self.residual_proj_temporal = nn.Linear(gru_hidden, spatial_out)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.gru_first = None
        if self.encoder_order == "temporal_first":
            # GRU runs over the RAW feature sequence here, so its input
            # width is in_channels (not spatial_out as in the default
            # order), and the decoder then reads the GAT's output.
            self.gru_first = nn.GRU(input_size=in_channels, hidden_size=gru_hidden, batch_first=True)
            self.gru = None
            decoder_in = spatial_out
        elif use_temporal:
            self.gru = nn.GRU(input_size=spatial_out, hidden_size=gru_hidden, batch_first=True)
            decoder_in = gru_hidden
        else:
            self.gru = None
            decoder_in = spatial_out

        self.decoder = nn.Linear(decoder_in, horizon)
        if self.zero_inflated:
            self.zero_gate = nn.Linear(decoder_in, horizon)
        if negative_binomial:
            self.dispersion_gate = nn.Linear(decoder_in, horizon)
        if tweedie:
            self.phi_gate = nn.Linear(decoder_in, horizon)

    def _spatial_forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Runs every stacked spatial (GAT or Linear) layer in sequence,
        with a ReLU between layers - the paper's own two-layer GNN stacks
        the *same* kind of layer, not a GAT layer followed by something
        else."""
        h = x
        for i, layer in enumerate(self.spatial_layers):
            h = layer(h, edge_index) if self.use_graph else layer(h)
            if i < len(self.spatial_layers) - 1:
                h = torch.relu(h)
        return h

    def forward(self, x_seq: torch.Tensor, edge_index: torch.Tensor):
        """x_seq: [T, N, F] node features per timestep.

        Returns [N] predicted counts for the step *after* the sequence
        (the target year) when `zero_inflated=False` and `horizon == 1`
        (unchanged contract for all existing callers); returns a
        `(lam, pi)` tuple of [N] tensors when `zero_inflated=True` (and
        neither `negative_binomial` nor `tweedie`) and `horizon == 1`;
        returns a `(mu, pi, r)` triple when `negative_binomial=True`, or
        a `(mu, pi, phi)` triple when `tweedie=True`. When `horizon > 1`,
        the trailing dimension is kept instead of squeezed -
        `[N, horizon]` per tensor - one row of `horizon` future values per
        segment.
        """
        if self.encoder_order == "temporal_first":
            # The PAPER's own order (thesis Appendix A.3/A.4): the GRU
            # consumes the raw feature sequence first - "ZT = GRU(X_1:t,
            # Y_1:t)... Once the input features have been processed via
            # the GRU, we obtain historical temporal embeddings ZT for
            # all roads. These are subsequently directed to graph neural
            # encoders" - and the GAT then does ONE spatial pass over
            # that temporal embedding. This project's original order is
            # the reverse (GAT at every timestep, then GRU over the
            # results); the difference was found 2026-09-02 by reading
            # their appendix rather than only their results tables.
            # Also markedly cheaper: one GAT pass instead of T.
            if not self.use_temporal:
                raise ValueError(
                    "encoder_order='temporal_first' requires use_temporal=True - "
                    "there is no temporal encoder to run first otherwise."
                )
            sequence = x_seq.permute(1, 0, 2)  # [T, N, F] -> [N, T, F]
            _, h_n = self.gru_first(sequence)  # [1, N, gru_hidden]
            temporal_embedding = h_n.squeeze(0)  # [N, gru_hidden]
            final = self._spatial_forward(temporal_embedding, edge_index)
            if self.use_residual:
                final = final + self.residual_proj_temporal(temporal_embedding)
            final = torch.relu(final)
            final = self.dropout(final)
        else:
            per_year_embeddings = []
            for t in range(x_seq.shape[0]):
                h = self._spatial_forward(x_seq[t], edge_index)
                if self.use_residual:
                    h = h + self.residual_proj(x_seq[t])  # keep the segment's own signal alongside the graph-aggregated one
                h = torch.relu(h)
                h = self.dropout(h)
                per_year_embeddings.append(h)

            if self.use_temporal:
                sequence = torch.stack(per_year_embeddings, dim=1)  # [N, T, H]
                _, h_n = self.gru(sequence)  # h_n: [1, N, gru_hidden]
                final = h_n.squeeze(0)
            else:
                final = per_year_embeddings[-1]  # only the most recent year - no memory

        raw_rate = self.decoder(final)
        if self.rate_link == "exp":
            # log-link (rate = exp(raw)) - the convention found 2026-09-02
            # in the paper's own linked reference code
            # (github.com/STTDAnonymous/STTD's model.py/utils.py: the
            # model's mu head has NO activation, `mu = torch.exp(mu)` is
            # applied only inside the loss function) - a real,
            # deliberately different design from this project's own
            # original softplus choice, offered here as a genuine,
            # disclosed alternative to test, not assumed better a
            # priori. Clamped before exp() (a standard numerical-safety
            # practice for log-links, not specific to this project) since
            # an unclamped large pre-activation value can overflow to inf
            # - +-15 keeps rate within [~3e-7, ~3e6], far wider than any
            # plausible crash-count/TCR value this project's targets take.
            rate = torch.exp(raw_rate.clamp(min=-15.0, max=15.0))
        else:
            rate = torch.nn.functional.softplus(raw_rate)  # [N, horizon], non-negative rate/mean
        if self.horizon == 1:
            rate = rate.squeeze(-1)  # [N] - exact prior shape, every existing caller relies on this
        if not self.zero_inflated:
            return rate
        pi = torch.sigmoid(self.zero_gate(final))  # [N, horizon], structural-zero probability
        if self.horizon == 1:
            pi = pi.squeeze(-1)
        if self.negative_binomial:
            r = torch.nn.functional.softplus(self.dispersion_gate(final)) + 1e-3  # [N, horizon], NB dispersion
            if self.horizon == 1:
                r = r.squeeze(-1)
            return rate, pi, r
        if self.tweedie:
            phi = torch.nn.functional.softplus(self.phi_gate(final)) + 1e-3  # [N, horizon], Tweedie dispersion
            if self.horizon == 1:
                phi = phi.squeeze(-1)
            return rate, pi, phi
        return rate, pi


_device_logged = False


def _resolve_device(device: str | None) -> torch.device:
    """Auto-detect CUDA if available and not overridden - the RTX 4060 in
    this project's dev machine cuts a 200-epoch training run from ~15-20s
    to a fraction of that, which matters once experiments run in batches
    (architecture sweeps, multi-borough runs, cross-validation)."""
    global _device_logged
    resolved = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if not _device_logged:
        # Logged once per process (not once per call - this runs many times
        # per pipeline invocation across splits/ablations/boroughs) so a
        # pipeline run's log is verifiable evidence of which hardware
        # actually trained the model, without spamming every call site.
        if resolved.type == "cuda":
            logger.info("GAT training device: cuda (%s)", torch.cuda.get_device_name(resolved))
        else:
            logger.info("GAT training device: cpu (no CUDA device detected)")
        _device_logged = True
    return resolved


def train_gat_temporal(
    x_seq: np.ndarray,
    edge_index: np.ndarray,
    y_target: np.ndarray,
    train_mask: np.ndarray | None = None,
    epochs: int = 200,
    lr: float = 0.01,
    seed: int = 42,
    use_graph: bool = True,
    use_temporal: bool = True,
    dropout: float = 0.0,
    weight_decay: float = 0.0,
    use_residual: bool = False,
    gat_hidden: int = 16,
    heads: int = 4,
    zero_inflated: bool = False,
    device: str | None = None,
) -> GATTemporal:
    """Train on a single (transductive) graph: all segments' structure is
    visible, but the loss is only computed on `train_mask` segments -
    exactly how GNN spatial-holdout evaluation is standardly done (the
    held-out segments' features/edges still inform message passing, but
    their labels are never used to update weights).

    `dropout`/`weight_decay` default to 0 (matching the earlier
    unregularised version) so existing callers/tests are unaffected; the
    live pipeline passes the precedent-informed values explicitly.
    `use_residual`, `gat_hidden` and `heads` let a caller run the
    over-smoothing investigation (see the module docstring) without
    touching this function's body. `zero_inflated` switches the loss to
    the ZIP mixture (see `zero_inflated_poisson_nll`). `device` defaults to
    auto-detected CUDA if available (`_resolve_device`); the returned model
    lives on that device, so callers must move new input tensors there too
    (`predict_gat_temporal` already does this from the model's own device).
    """
    torch.manual_seed(seed)
    resolved_device = _resolve_device(device)
    n_segments = x_seq.shape[1]
    if train_mask is None:
        train_mask = np.ones(n_segments, dtype=bool)

    x_t = torch.tensor(x_seq, dtype=torch.float32, device=resolved_device)
    edge_index_t = torch.tensor(edge_index, dtype=torch.long, device=resolved_device)
    y_t = torch.tensor(y_target, dtype=torch.float32, device=resolved_device)
    mask_t = torch.tensor(train_mask, dtype=torch.bool, device=resolved_device)

    model = GATTemporal(
        in_channels=x_seq.shape[2], gat_hidden=gat_hidden, heads=heads,
        use_graph=use_graph, use_temporal=use_temporal, dropout=dropout, use_residual=use_residual,
        zero_inflated=zero_inflated,
    ).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    poisson_loss_fn = nn.PoissonNLLLoss(log_input=False, full=True, eps=1e-6)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        if zero_inflated:
            lam, pi = model(x_t, edge_index_t)
            loss = zero_inflated_poisson_nll(lam[mask_t], pi[mask_t], y_t[mask_t])
        else:
            pred = model(x_t, edge_index_t)
            loss = poisson_loss_fn(pred[mask_t], y_t[mask_t])
        loss.backward()
        optimizer.step()

    return model


def train_gat_temporal_walkforward(
    train_instances: list[tuple[np.ndarray, np.ndarray]],
    edge_index: np.ndarray,
    train_mask: np.ndarray | None = None,
    epochs: int = 200,
    lr: float = 0.01,
    seed: int = 42,
    use_graph: bool = True,
    use_temporal: bool = True,
    dropout: float = 0.0,
    weight_decay: float = 0.0,
    use_residual: bool = False,
    gat_hidden: int = 16,
    gru_hidden: int = 32,
    heads: int = 4,
    zero_inflated: bool = False,
    negative_binomial: bool = False,
    tweedie: bool = False,
    tweedie_rho: float = 1.5,
    gat_layers: int = 1,
    horizon: int = 1,
    rank_loss_weight: float = 0.0,
    topk_loss_weight: float = 0.0,
    rank_margin: float = 1.0,
    early_stopping_patience: int | None = None,
    early_stopping_val_instances: int = 1,
    rate_link: str = "softplus",
    encoder_order: str = "spatial_first",
    device: str | None = None,
) -> GATTemporal:
    """The methodologically correct version of `train_gat_temporal` for a
    genuine temporal held-out test: trains jointly across MULTIPLE,
    distinct (x_seq, y_target) transitions (see
    `features.graph_temporal.build_walkforward_instances`) - e.g. learn
    the 2021-2022->2023 transition, then apply that learned transition
    function, unmodified, to the held-out 2022-2023->2024 transition.

    Found and fixed 2026-08-31: `train_gat_temporal` alone, called with
    the *evaluation* target as `y_target`, trains directly against the
    exact value it is later scored against - an in-sample fit, not a
    temporal-generalisation test, however many epochs are used (confirmed
    by a 2000-epoch run collapsing to a suspicious PR-AUC of 0.85 - see
    docs/decision_log.md for the full account). This function is what the
    pipeline's "temporal" GAT evaluation should call instead; the last
    instance in `train_instances` is conventionally the held-out one and
    should be excluded from `train_instances` passed here (only earlier
    transitions belong in training) - the caller predicts on the held-out
    transition separately via `predict_gat_temporal`.

    `horizon` (added 2026-09-01, see `GATTemporal`'s own docstring):
    defaults to 1, the original single-step behaviour. For `horizon > 1`,
    every `y_target` in `train_instances` must be shaped `[N, horizon]`
    (not `[N]`) - the loss functions below are element-wise means, so no
    other change is needed here for multi-step training to work.

    `rank_loss_weight` (added 2026-09-01, see `pairwise_rank_hinge_loss`'s
    own docstring for the full motivation): 0.0 (the default) trains on
    pure distributional NLL only, unchanged from every prior caller.
    A nonzero weight adds `rank_loss_weight * pairwise_rank_hinge_loss`
    (margin `rank_margin`) to the loss every epoch, computed per
    day/horizon-step on the *point-estimate* score `(1-pi)*rate` (the
    same quantity `predict_gat_temporal` returns and `accuracy_hit_rate`
    ranks by) - directly optimising the ranking the eval metric measures,
    alongside (not instead of) the distributional fit.

    `early_stopping_patience` (added 2026-09-01, motivated directly by
    this project's own evidence: `epochs=20` badly underfit and
    `epochs=500` measurably overfit on the *identical* data and
    architecture - docs/decision_log.md's "research paper approach" entry
    - proving a real sweet spot exists that a hand-picked fixed epoch
    count can only guess at). When set, the LAST instance in
    `train_instances` is held out as a validation set (never trained on,
    distinct from the caller's own held-out test instance, which is
    already excluded from `train_instances` before this function is ever
    called - see this function's own docstring above); `epochs` becomes a
    maximum rather than a fixed count, training stops once validation
    loss hasn't improved for `early_stopping_patience` consecutive
    epochs, and the returned model has the BEST-validation-loss weights
    restored, not whichever epoch happened to run last. Requires at least
    2 instances in `train_instances` (one becomes validation) - with
    exactly 1, early stopping is silently disabled (nothing left to
    validate against) and training proceeds for the full `epochs` budget,
    matching every prior caller's behaviour exactly when this parameter
    is left at its default `None`.

    `tweedie`/`tweedie_rho` (added 2026-09-01, see `zero_inflated_tweedie_nll`'s
    own docstring for the full derivation, transcribed directly from the
    paper's PDF text): the paper's own reported third decoder option,
    for use with a severity-weighted target (`tcr_score`) where a
    compound Poisson-Gamma decoder is architecturally appropriate.
    Mutually exclusive with `negative_binomial` - `GATTemporal` raises
    `ValueError` if both are set. `tweedie_rho` (default 1.5) is a fixed
    hyperparameter, not a 4th learned output like the paper's own
    architecture - see that docstring for why.

    `early_stopping_val_instances` (added 2026-09-01, generalises the
    above after the single-held-out-instance version measurably HURT
    results - 26.37% mean AccHR@20 vs the fixed-200-epoch 49.57%,
    docs/decision_log.md's "think, think, think" entry): with the
    default `1`, behaviour is byte-for-byte identical to the original
    single-instance version above. Set higher (e.g. 3) to instead hold
    out the LAST `early_stopping_val_instances` instances and validate
    against their MEAN loss every epoch - the diagnosed failure mode was
    that one held-out instance is too noisy a generalisation signal at
    this pipeline's small per-window instance counts; averaging several
    reduces that noise the same way k-fold cross-validation does,
    without the expense of actually re-training k separate models.
    Ignored (falls back to `1`) if fewer than
    `early_stopping_val_instances + 1` instances are available - the
    minimum unaffected by this generalisation is still "at least 2
    instances total" from before.
    """
    if not train_instances:
        raise ValueError("train_gat_temporal_walkforward needs at least one training transition.")
    if early_stopping_val_instances < 1:
        raise ValueError("early_stopping_val_instances must be >= 1.")

    torch.manual_seed(seed)
    resolved_device = _resolve_device(device)
    n_segments = train_instances[0][0].shape[1]
    if train_mask is None:
        train_mask = np.ones(n_segments, dtype=bool)
    mask_t = torch.tensor(train_mask, dtype=torch.bool, device=resolved_device)
    edge_index_t = torch.tensor(edge_index, dtype=torch.long, device=resolved_device)

    n_val = early_stopping_val_instances if len(train_instances) > early_stopping_val_instances else 1
    use_early_stopping = early_stopping_patience is not None and len(train_instances) >= n_val + 1
    fit_instances = train_instances[:-n_val] if use_early_stopping else train_instances
    val_instances = train_instances[-n_val:] if use_early_stopping else []

    instances_t = [
        (
            torch.tensor(x_seq, dtype=torch.float32, device=resolved_device),
            torch.tensor(y_target, dtype=torch.float32, device=resolved_device),
        )
        for x_seq, y_target in fit_instances
    ]
    val_instances_t = [
        (
            torch.tensor(x_seq, dtype=torch.float32, device=resolved_device),
            torch.tensor(y_target, dtype=torch.float32, device=resolved_device),
        )
        for x_seq, y_target in val_instances
    ]

    model = GATTemporal(
        in_channels=train_instances[0][0].shape[2], gat_hidden=gat_hidden, gru_hidden=gru_hidden, heads=heads,
        use_graph=use_graph, use_temporal=use_temporal, dropout=dropout, use_residual=use_residual,
        zero_inflated=zero_inflated, negative_binomial=negative_binomial, tweedie=tweedie, tweedie_rho=tweedie_rho,
        rate_link=rate_link,
        encoder_order=encoder_order,
        gat_layers=gat_layers, horizon=horizon,
    ).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    poisson_loss_fn = nn.PoissonNLLLoss(log_input=False, full=True, eps=1e-6)

    # Per-instance backward + accumulate, not "sum every instance's loss
    # into one graph, then one big backward()" (2026-09-01, found the hard
    # way: a 128-instance data-richness run - see
    # docs/publication_readiness.md - OOM'd an 8GB GPU here, since the old
    # approach keeps every instance's full forward-pass computation graph
    # alive simultaneously until the single final backward() call).
    # Calling `.backward()` once per instance frees that instance's graph
    # immediately afterward (PyTorch's default `retain_graph=False`);
    # gradients accumulate on `model.parameters()` across multiple
    # `.backward()` calls between `zero_grad()`s exactly the same way
    # they'd sum inside one graph - `d(a+b)/dx == da/dx + db/dx` - so this
    # produces an identical update to the old code at every epoch, only
    # with peak memory bounded by one instance instead of all of them.
    def _topk_loss_over_days(rate: torch.Tensor, pi: torch.Tensor | None, y: torch.Tensor) -> torch.Tensor:
        """Per-day top-k hinge (see `top_k_hinge_loss`) over the same
        point estimate the evaluation ranks on: E[y] = (1-pi)*rate."""
        point = rate if pi is None else (1 - pi) * rate
        if point.dim() == 1:
            return top_k_hinge_loss(point, y if y.dim() == 1 else y[:, 0])
        total = point.new_zeros(())
        n_days = point.shape[1]
        for d in range(n_days):
            total = total + top_k_hinge_loss(point[:, d], y[:, d])
        return total / max(n_days, 1)

    def _rank_loss_over_days(rate: torch.Tensor, pi: torch.Tensor | None, y: torch.Tensor) -> torch.Tensor:
        """Point-estimate score = (1-pi)*rate (matches predict_gat_temporal
        exactly), ranking loss computed per day/horizon-step then averaged
        across days - `accuracy_hit_rate` ranks *within* a single day,
        never across days, so the auxiliary loss must respect that same
        boundary rather than pooling all (segment, day) pairs into one
        global ranking, which would be a different, wrong objective."""
        score = rate if pi is None else (1 - pi) * rate
        score, y = score[mask_t], y[mask_t]
        if score.dim() == 1:
            return pairwise_rank_hinge_loss(score, y, margin=rank_margin)
        per_day = [pairwise_rank_hinge_loss(score[:, h], y[:, h], margin=rank_margin) for h in range(score.shape[1])]
        return torch.stack(per_day).mean()

    def _compute_loss(x_t: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        if negative_binomial:
            mu, pi, r = model(x_t, edge_index_t)
            loss = zero_inflated_negative_binomial_nll(mu[mask_t], pi[mask_t], r[mask_t], y_t[mask_t])
            if rank_loss_weight > 0:
                loss = loss + rank_loss_weight * _rank_loss_over_days(mu, pi, y_t)
            if topk_loss_weight > 0:
                loss = loss + topk_loss_weight * _topk_loss_over_days(mu, pi, y_t)
        elif tweedie:
            mu, pi, phi = model(x_t, edge_index_t)
            loss = zero_inflated_tweedie_nll(mu[mask_t], pi[mask_t], phi[mask_t], y_t[mask_t], rho=tweedie_rho)
            if rank_loss_weight > 0:
                loss = loss + rank_loss_weight * _rank_loss_over_days(mu, pi, y_t)
            if topk_loss_weight > 0:
                loss = loss + topk_loss_weight * _topk_loss_over_days(mu, pi, y_t)
        elif zero_inflated:
            lam, pi = model(x_t, edge_index_t)
            loss = zero_inflated_poisson_nll(lam[mask_t], pi[mask_t], y_t[mask_t])
            if rank_loss_weight > 0:
                loss = loss + rank_loss_weight * _rank_loss_over_days(lam, pi, y_t)
            if topk_loss_weight > 0:
                loss = loss + topk_loss_weight * _topk_loss_over_days(lam, pi, y_t)
        else:
            pred = model(x_t, edge_index_t)
            loss = poisson_loss_fn(pred[mask_t], y_t[mask_t])
            if rank_loss_weight > 0:
                loss = loss + rank_loss_weight * _rank_loss_over_days(pred, None, y_t)
        return loss

    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        for x_t, y_t in instances_t:
            _compute_loss(x_t, y_t).backward()
        optimizer.step()

        if val_instances_t:
            model.eval()
            with torch.no_grad():
                val_loss = torch.stack([_compute_loss(x_t, y_t) for x_t, y_t in val_instances_t]).mean().item()
            model.train()
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def predict_gat_temporal(model: GATTemporal, x_seq: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    """Point-prediction array [N]. For a zero-inflated model this is the
    mixture's expected count, E[y] = (1 - pi) * lam - not just lam alone -
    so it stays comparable to every other model's point estimate. Same
    mixture-expectation logic applies to the ZINB decoder (E[y] =
    (1-pi)*mu - NB2's own mean parameter already *is* the count mean, r
    only shapes the variance, so the point estimate formula is identical
    to the ZIP case) and to the Tweedie/ZITD decoder (E[y] = (1-pi)*mu -
    the paper's own Eq. 6 states this exact formula, "the mean value of
    the ZITD distribution is E(y_k) = (1-pi)*mu")."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x_seq, dtype=torch.float32, device=device)
        edge_index_t = torch.tensor(edge_index, dtype=torch.long, device=device)
        pred = model(x_t, edge_index_t)
        if model.negative_binomial or model.tweedie:
            mu, pi, _dispersion = pred
            pred = (1 - pi) * mu
        elif model.zero_inflated:
            lam, pi = pred
            pred = (1 - pi) * lam
    return pred.cpu().numpy()
