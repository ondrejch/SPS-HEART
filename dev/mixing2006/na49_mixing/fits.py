# Not used by the HEART p+p fits (marked 2026-11-04): no production generation, fit, probe tool or paper script
# runs this module (an execution trace of the R137 generation and an import scan of the engine, the probe tools
# and the paper scripts). Code reviews of the p+p model can skip it. It stays until the cleanup and refactor after
# the p+A run.
"""Chi² fitting utilities — port of ``TFits`` / ``TMultiFit``.

:class:`Fitter`
    Compares data and MC difference spectra and computes optimal scale
    factors using the standard weighted least-squares formula.

:class:`MultiFitter`
    Fits multiple MC resonance contributions simultaneously to data using
    ``scipy.optimize.minimize``.
"""

from __future__ import annotations
import math
from typing import Optional, List, Dict
import numpy as np

from .histogram import Histogram1D
from .spectra import Spectra


class Fitter:
    """Compare one data spectrum against one MC spectrum.

    Reproduces the ``TFits::Get_fit`` / ``Get_error`` logic.
    """

    def __init__(
        self,
        nbins: int = 50,
        x_low: float = 1.1,
        x_high: float = 2.1,
    ) -> None:
        self._his_a = Histogram1D(nbins, x_low, x_high, name="data_diff")
        self._his_b = Histogram1D(nbins, x_low, x_high, name="mc_diff")
        self._his_data_mass = Histogram1D(nbins, x_low, x_high, name="data_mass")
        self._his_data_mix = Histogram1D(nbins, x_low, x_high, name="data_mix")
        self._his_mc_mass = Histogram1D(nbins, x_low, x_high, name="mc_mass")
        self._his_mc_mix = Histogram1D(nbins, x_low, x_high, name="mc_mix")
        self._n_mc_entries: int = 0

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------
    def add_spectra(self, spectra: Spectra) -> None:
        """Add histograms from *spectra* to the appropriate accumulators."""
        dif = spectra.get_dif_hist()
        mass = spectra.get_mass_hist()
        mix = spectra.get_mix_hist()
        if spectra.is_data():
            if dif is not None:
                self._his_a.add(dif)
            if mass is not None:
                self._his_data_mass.add(mass)
            if mix is not None:
                self._his_data_mix.add(mix)
        else:
            if dif is not None:
                self._his_b.add(dif)
            if mass is not None:
                self._his_mc_mass.add(mass)
            if mix is not None:
                self._his_mc_mix.add(mix)
            self._n_mc_entries += spectra.get_entries()

    def subtract_spectra(self, spectra: Spectra) -> None:
        """Subtract histograms from *spectra*."""
        dif = spectra.get_dif_hist()
        mass = spectra.get_mass_hist()
        mix = spectra.get_mix_hist()
        if spectra.is_data():
            if dif is not None:
                self._his_a.add(dif, scale=-1.0)
            if mass is not None:
                self._his_data_mass.add(mass, scale=-1.0)
            if mix is not None:
                self._his_data_mix.add(mix, scale=-1.0)
        else:
            if dif is not None:
                self._his_b.add(dif, scale=-1.0)
            if mass is not None:
                self._his_mc_mass.add(mass, scale=-1.0)
            if mix is not None:
                self._his_mc_mix.add(mix, scale=-1.0)
            self._n_mc_entries -= spectra.get_entries()

    def add_a(self, hist: Histogram1D) -> None:
        self._his_a.add(hist)

    def add_b(self, hist: Histogram1D) -> None:
        self._his_b.add(hist)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def get_fit(self) -> float:
        """Return α = Σ(a·b/σ²) / Σ(b²/σ²) over all bins."""
        n = self._his_a.nbins
        a = self._his_a._contents[1:n + 1].copy()
        b = self._his_b._contents[1:n + 1].copy()
        mask = a > 0.0
        if not mask.any():
            return 0.0
        am = a[mask]
        bm = b[mask]
        up = bm.sum()
        down = (bm * bm / am).sum()
        return float(up / down) if down != 0.0 else 0.0

    def get_error(self) -> float:
        """Return the statistical uncertainty on the scale factor."""
        n = self._his_a.nbins
        a = self._his_a._contents[1:n + 1].copy()
        b = self._his_b._contents[1:n + 1].copy()
        mask = a > 0.0
        if not mask.any():
            return 0.0
        sigma = (b[mask] ** 2 / a[mask]).sum()
        return float(1.0 / np.sqrt(sigma)) if sigma > 0.0 else 0.0

    def clear(self) -> None:
        for h in (
            self._his_a, self._his_b,
            self._his_data_mass, self._his_data_mix,
            self._his_mc_mass, self._his_mc_mix,
        ):
            h._contents[:] = 0.0
            h._cdf = None
        self._n_mc_entries = 0

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def data_diff(self) -> Histogram1D: return self._his_a
    @property
    def mc_diff(self) -> Histogram1D: return self._his_b
    @property
    def n_mc_entries(self) -> int: return self._n_mc_entries

    def __repr__(self) -> str:
        return (
            f"Fitter(fit={self.get_fit():.4f}, error={self.get_error():.4f})"
        )


# ---------------------------------------------------------------------------
# Multi-component fitter
# ---------------------------------------------------------------------------

class MultiFitter:
    """Fit multiple MC resonance contributions simultaneously to data.

    Parameters
    ----------
    data_hist : Histogram1D
        Measured (background-subtracted) mass spectrum.
    mc_hists : list of Histogram1D
        One histogram per resonance species (background subtracted).
    fit_range : tuple of (float, float)
        Mass range over which the chi² is evaluated.

    Notes
    -----
    The constructor deliberately stays permissive; every domain,
    compatibility, and optimizer-result check lives in :meth:`fit` and
    raises :class:`ValueError`. ``self._scales`` starts at ``None``, every
    :meth:`fit` call clears it on entry, and it is written back only after
    a fully successful fit, so :attr:`scales` never exposes the stale
    result of an earlier fit. Both :meth:`fit` and :attr:`scales` return
    a copy of the stored array, so a caller cannot mutate the fitter's
    result through the returned object. See that method's docstring for
    the exact contract.

    The diagnostic attribute :attr:`design_condition_number` — the 2-norm
    condition number of the weighted design matrix of the chi-square —
    follows the same stale-state discipline as ``self._scales``: it is
    ``None`` initially and on entry of every :meth:`fit` call, and it is
    written only after a fully successful fit. It is a plain float
    describing the conditioning of the fit problem, never a second scale
    array.
    """

    def __init__(
        self,
        data_hist: Histogram1D,
        mc_hists: List[Histogram1D],
        fit_range: tuple = (1.1, 2.1),
    ) -> None:
        self._data = data_hist
        self._mc = mc_hists
        self._fit_range = fit_range
        self._scales: Optional[np.ndarray] = None
        self.design_condition_number: Optional[float] = None

    def fit(self) -> np.ndarray:
        """Minimise χ² and return the optimal scale factors.

        Every call starts by discarding any previously stored result:
        ``self._scales`` resets to ``None`` before the validation steps
        below run, so a call that fails at validation, at the optimizer,
        or in the result checks never leaves the scales of an earlier
        successful fit visible through :attr:`scales`. The freshly
        optimized array is copied into ``self._scales`` only after every
        result check has succeeded. The array returned to the caller is
        a further copy, and :attr:`scales` likewise returns a copy, so
        mutating either view cannot change the stored result.

        Validation ordering — each step raises ``ValueError`` before any
        optimizer work starts:

        1. The MC component list must be non-empty (checked before the
           scipy import and before any bin access).
        2. Bin selection flows through ``Histogram1D.find_bin``, so the
           histogram non-finite policy keeps precedence: NaN/±inf
           endpoints raise from ``find_bin``, naming it. This ordering is
           pinned by ``test_nonfinite_endpoint_raises_valueerror_naming_find_bin``.
        3. Finite endpoints must satisfy strict ``low < high``; a
           reversed pair is rejected by value, regardless of what
           ``find_bin`` returned for it.
        4. Every MC histogram must carry exactly the data histogram's bin
           geometry (``nbins``, ``xlow``, ``xhigh``, compared exactly),
           so one bin index addresses the same interval in all of them;
           errors name the offending component index/name and both
           geometries.
        5. The window must cover at least one interior bin, i.e.
           ``[i_min, i_max] ∩ [1, nbins]`` non-empty. Windows may include
           under-/overflow bins *alongside* interior bins; a window whose
           selection touches only flow bins carries no interior-bin
           information and is rejected. Together with step 3 this also
           guarantees at least one selected bin in total.
        6. Identifiability over the selected domain ``[i_min..i_max]``,
           the very bins the χ² consumes. The general rule is the
           column rank of the weighted design matrix the χ² actually
           uses: ``design = (mc_vals / errors).T`` with shape
           ``(n_selected, n_comp)``, columns ``mc_vals[j] / errors``,
           so the objective is ``||data/errors - design @ scales||^2``.
           The fit is rejected when the numerical column rank of
           ``design`` is less than the number of components, computed
           with ``numpy.linalg.matrix_rank`` under its default SVD
           tolerance (``S.max() * max(design.shape) * eps``) — the one
           place a numerical tolerance is allowed in these checks. A
           rank-deficient design leaves at least one combination of
           scales undetermined; full column rank makes the χ² strictly
           convex in the scales, so the minimum over ``scales >= 0`` is
           unique and every returned scale is determined by the data.
           The rejection names the numerical rank, ``n_comp``, the
           selected bin interval, the fit range, and the index/name of
           every component that entered the design matrix. A design
           with non-finite entries is rejected by name instead of
           being silently ranked as zero.

           The zero-row and duplicate refusals below are faster
           exact-value special cases of the same rank rule; they run
           first and name a concrete component:

           - A component whose template is identically zero over the
             selected domain leaves its own scale undetermined while
             the remaining components stay identifiable. The fit is
             rejected and the error names the offending component
             index/name.
           - Exactly duplicated components (templates equal over the
             selected domain) fix only the sum of their scales; the
             individual split is arbitrary. The fit is rejected and
             the error names both offending components. Merge
             duplicates into one component, or reshape them until
             they differ inside the fit_range.

           The special-case comparisons use exact values rather than
           tolerances, so merely similar shapes reach the rank check
           and fit normally whenever the design has full numerical
           rank. A full-rank design can still be poorly conditioned;
           the 2-norm condition number of the design matrix is then
           available as the diagnostic attribute
           :attr:`design_condition_number` (a plain float, reset to
           ``None`` on every :meth:`fit` entry and written only after
           a fully successful fit; never a second scale array).

        After optimization the result must report success with a finite
        objective value and finite scales; otherwise a descriptive
        ``ValueError`` including the optimizer status/message is raised
        and no scale is stored.
        """
        # Finding 3: clearing precedes every validation step, so no
        # raising path and no failed refit can leave the scales of an
        # earlier successful fit visible through the scales property.
        # The design_condition_number diagnostic follows the same
        # discipline: reset here, written only after a fully successful
        # fit below.
        self._scales = None
        self.design_condition_number = None
        n_comp = len(self._mc)
        if n_comp < 1:
            raise ValueError(
                "MultiFitter.fit() requires at least one MC component "
                "to fit, got an empty mc_hists list"
            )

        low = self._fit_range[0]
        high = self._fit_range[1]
        # find_bin first: its non-finite policy owns NaN/±inf endpoints
        # (pinned behavior); strict ordering of finite endpoints follows.
        i_min = self._data.find_bin(low)
        i_max = self._data.find_bin(high)

        if not float(low) < float(high):
            raise ValueError(
                "MultiFitter.fit() requires fit_range[0] < fit_range[1], "
                f"got fit_range={self._fit_range!r}"
            )

        for j, hist in enumerate(self._mc):
            if (
                hist.nbins != self._data.nbins
                or hist.xlow != self._data.xlow
                or hist.xhigh != self._data.xhigh
            ):
                raise ValueError(
                    f"MultiFitter.fit() requires MC component {j} "
                    f"({hist.name!r}) to share the data histogram's "
                    "exact bin geometry (nbins, xlow, xhigh), got MC "
                    f"(nbins={hist.nbins}, xlow={hist.xlow!r}, "
                    f"xhigh={hist.xhigh!r}) versus data "
                    f"(nbins={self._data.nbins}, "
                    f"xlow={self._data.xlow!r}, "
                    f"xhigh={self._data.xhigh!r})"
                )

        if max(i_min, 1) > min(i_max, self._data.nbins):
            raise ValueError(
                f"MultiFitter.fit() requires fit_range={self._fit_range!r} "
                "to overlap at least one interior bin of the data "
                f"histogram (nbins={self._data.nbins}, "
                f"xlow={self._data.xlow!r}, xhigh={self._data.xhigh!r}); "
                f"the selected bins {i_min}..{i_max} touch only "
                "under-/overflow bins"
            )

        from scipy.optimize import minimize  # local import

        data_vals = np.array(
            [self._data.get_bin_content(i) for i in range(i_min, i_max + 1)]
        )
        mc_vals = np.array(
            [
                [h.get_bin_content(i) for i in range(i_min, i_max + 1)]
                for h in self._mc
            ]
        )
        errors = np.sqrt(np.maximum(data_vals, 1.0))

        # Finding 2: sensitivity checks over the selected domain, i.e.
        # exactly the bins the chi-square consumes. A matrix of
        # identically zero templates leaves every scale unconstrained; a
        # single zero row leaves that component's scale unconstrained;
        # exactly duplicated rows leave only the sum of the two scales
        # constrained. All three are rejected here, before any optimizer
        # work. Comparisons use exact values, never tolerances.
        component_has_info = (mc_vals != 0.0).any(axis=1)
        if not bool(component_has_info.any()):
            raise ValueError(
                "MultiFitter.fit() rejects the fit because no MC "
                "component contributes information in the fit domain "
                f"{self._fit_range!r}: all {n_comp} MC component "
                "templates are identically zero over the selected bins "
                f"{i_min}..{i_max}, leaving every scale unconstrained; "
                "supply MC components with nonzero template contents "
                "inside the fit_range"
            )
        zero_rows = [
            j for j in range(n_comp) if not component_has_info[j]
        ]
        if zero_rows:
            j = zero_rows[0]
            raise ValueError(
                f"MultiFitter.fit() rejects MC component {j} "
                f"({self._mc[j].name!r}): its template is identically "
                f"zero over the selected bins {i_min}..{i_max} of "
                f"fit_range {self._fit_range!r}, so this component's "
                "scale is unconstrained while the remaining components "
                "stay identifiable; remove the component or give it "
                "nonzero contents inside the fit_range"
            )
        for j1 in range(n_comp - 1):
            for j2 in range(j1 + 1, n_comp):
                if bool(np.array_equal(mc_vals[j1], mc_vals[j2])):
                    raise ValueError(
                        "MultiFitter.fit() rejects exactly duplicated MC "
                        f"components {j1} ({self._mc[j1].name!r}) and "
                        f"{j2} ({self._mc[j2].name!r}): their templates "
                        f"are identical over the selected bins "
                        f"{i_min}..{i_max} of fit_range "
                        f"{self._fit_range!r}, so the chi-square "
                        "determines only the sum of their scales and "
                        "the individual split is arbitrary; merge the "
                        "duplicate components into a single component"
                    )

        # Identifiability, general rule: the chi-square consumes the
        # weighted design matrix A with shape (n_selected, n_comp)
        # whose columns are mc_vals[j] / errors, so the objective is
        # ||data/errors - A @ scales||^2. Any exact linear dependence
        # among the columns (proportional templates, a template equal
        # to a combination of the others, more components than
        # selected bins) leaves at least one scale combination
        # undetermined, so the numerical column rank is checked with
        # numpy.linalg.matrix_rank under its default SVD tolerance
        # (S.max() * max(A.shape) * eps) — the one place a numerical
        # tolerance is allowed; the zero-row and duplicate checks
        # above are the faster exact-value special cases. The error
        # floor sqrt(max(data_vals, 1.0)) keeps the division finite in
        # normal operation; non-finite storage (reachable through
        # set_bin_content) is rejected by name instead of silently
        # ranking as zero.
        design = (mc_vals / errors).T          # (n_selected, n_comp)
        if not bool(np.isfinite(design).all()):
            raise ValueError(
                "MultiFitter.fit() rejects the fit because the "
                "weighted design matrix (mc_vals[j] / errors) over "
                f"the selected bins {i_min}..{i_max} of fit_range "
                f"{self._fit_range!r} contains non-finite values "
                "(non-finite histogram contents are reachable through "
                "set_bin_content), so its column rank cannot be "
                "computed reliably; supply finite data and MC "
                "template contents inside the fit_range"
            )
        rank = int(np.linalg.matrix_rank(design))
        if rank < n_comp:
            comp_list = ", ".join(
                f"{j} ({self._mc[j].name!r})" for j in range(n_comp)
            )
            raise ValueError(
                "MultiFitter.fit() rejects the fit because the "
                "weighted design matrix of the chi-square does not "
                f"have full column rank: numerical rank {rank} < "
                f"n_comp = {n_comp} over the selected bins "
                f"{i_min}..{i_max} of fit_range {self._fit_range!r}; "
                "the components entering the design matrix are "
                f"[{comp_list}], and this dependence leaves at least "
                "one combination of scales undetermined, so the "
                "individual scales are not identifiable; reshape the "
                "MC templates so they are linearly independent "
                "inside the fit_range"
            )

        def chi2(scales):
            pred = (np.array(scales)[:, None] * mc_vals).sum(axis=0)
            return float(np.sum(((data_vals - pred) / errors) ** 2))

        x0 = np.ones(n_comp)
        bounds = [(0.0, None)] * n_comp
        result = minimize(chi2, x0, method="L-BFGS-B", bounds=bounds)
        scales = np.asarray(result.x, dtype=float)
        objective = float(result.fun)
        opt_message = getattr(
            result, "message", "<optimizer reported no message>"
        )
        opt_status = getattr(result, "status", None)

        failure: Optional[str] = None
        if not result.success:
            failure = (
                "the optimizer did not converge "
                f"(success={result.success!r}, status={opt_status!r})"
            )
        elif not math.isfinite(objective):
            failure = (
                "the objective at the returned scales is non-finite "
                f"({objective!r})"
            )
        elif not bool(np.all(np.isfinite(scales))):
            bad = ", ".join(
                f"scales[{k}]={float(v)!r}"
                for k, v in enumerate(scales)
                if not math.isfinite(float(v))
            )
            failure = f"the returned scales are non-finite ({bad})"
        if failure is not None:
            raise ValueError(
                f"MultiFitter.fit() rejected the optimizer result: "
                f"{failure}; optimizer said {opt_message!r}; "
                "no scales were stored"
            )

        owned = np.array(scales, dtype=float, copy=True)
        self._scales = owned
        # Diagnostic only: the 2-norm condition number of the design
        # matrix validated above. It is written only after a fully
        # successful fit (reset on every fit() entry), so it is never
        # stale and never a second scale array.
        self.design_condition_number = float(np.linalg.cond(design, p=2))
        return owned.copy()

    @property
    def scales(self) -> Optional[np.ndarray]:
        if self._scales is None:
            return None
        return self._scales.copy()

    def __repr__(self) -> str:
        return f"MultiFitter(n_components={len(self._mc)})"
