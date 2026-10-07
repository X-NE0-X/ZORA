from ENV_MGMT.imports import *



class FactorLibrary:
    """
    ### What It Does
    Holds the predefined factor kernels (HMA, accumulated return, slope, quantile,
    convergence, divergence, ...) as a passive namespace of static/njit primitives.

    #### Responsibility
    Stores the predefined-factor compute primitives ONLY. It is not a manager and is never
    instantiated: `FactorManager.get()` computes predefined factors by calling these kernels
    (via `FactorManager._compute_customized`). Registering custom factors is `FactorManager`'s
    job, not this class's.

    #### How To Use
    Do not instantiate. Build the engine with `FactorManager(test_data)` and call `.get(...)`
    for predefined factors or `.register(...)` / `.register_vbt_factor(...)` for custom ones.
    Reference `FactorLibrary._hma(...)` etc. directly only when you need a raw kernel.

    #### Usage Example
    `FactorManager(test_data).get.single("HMA", {"period": 14})`
    """


    def __init__(self, *args: Any, **kwargs: Any) -> None:

        raise TypeError(
            "[WARNING] FactorLibrary is a passive predefined-factor kernel namespace and is not "
            "instantiable. Build the engine with FactorEngine.FactorManager(test_data) and call "
            ".get(...) for predefined factors or .register(...) for custom factors."
        )


    @staticmethod
    def _as_asset_param_array(value: Any, asset_count: int, dtype: Any) -> np.ndarray:

        arr = np.asarray(value, dtype = dtype).reshape(-1)

        if arr.size == 1 and asset_count > 1:
            arr = np.full(asset_count, arr[0], dtype = dtype)

        if arr.size != asset_count:

            raise ValueError(f"[WARNING] parameter length {arr.size} does not match asset count {asset_count}.")


        return arr


    @staticmethod
    @njit(cache = True)
    def _slope(arr_mat: np.ndarray, shift: int, mode: int) -> np.ndarray:

        T, M = arr_mat.shape

        # shift is a strictly-positive look-back. A negative shift reads arr_mat[t+|shift|] (a FUTURE bar
        # -> look-ahead) and indexes out of bounds (njit boundscheck=False -> garbage memory); shift==0
        # collapses to a zero/NaN diff. Reject invalid look-backs with an all-NaN factor.
        if shift < 1:

            return np.full((T, M), np.nan, dtype = arr_mat.dtype)

        slope_value = np.empty((T, M), dtype = arr_mat.dtype)

        for m in range(M):

            for t in range(T):

                if t < shift:
                    slope_value[t, m] = np.nan

                else:
                    diff = arr_mat[t, m] - arr_mat[t - shift, m]

                    if mode == 0:
                        slope_value[t, m] = diff

                    else:
                        slope_value[t, m] = diff / shift


        return slope_value


    @staticmethod
    @njit(cache = True)
    def _prev_diff(arr_mat: np.ndarray, lag_left: int, lag_right: int) -> np.ndarray:

        T, M = arr_mat.shape
        out = np.empty((T, M), dtype = arr_mat.dtype)
        lag_l = int(max(0, lag_left))
        lag_r = int(max(0, lag_right))

        for m in range(M):

            for t in range(T):
                left_idx = t - lag_l
                right_idx = t - lag_r

                if left_idx < 0 or right_idx < 0:
                    out[t, m] = np.nan

                else:
                    out[t, m] = arr_mat[left_idx, m] - arr_mat[right_idx, m]


        return out


    @staticmethod
    @njit(cache = True)
    def _prev_rollmax_safe(arr_mat: np.ndarray, window: int, warmup: int, shift: int, sentinel: float) -> np.ndarray:

        T, M = arr_mat.shape
        out = np.empty((T, M), dtype = arr_mat.dtype)
        rollmax = np.empty((T, M), dtype = arr_mat.dtype)
        w = int(max(1, window))
        warm = int(max(0, warmup))
        sh = int(max(0, shift))
        s_val = arr_mat.dtype.type(sentinel)

        for m in range(M):

            for t in range(T):
                start = t - w + 1

                if start < 0:
                    start = 0

                has_valid = False
                vmax = 0.0

                for i in range(start, t + 1):
                    v = arr_mat[i, m]

                    if np.isnan(v):

                        continue

                    if (not has_valid) or v > vmax:
                        vmax = v
                        has_valid = True

                if has_valid:
                    rollmax[t, m] = vmax

                else:
                    rollmax[t, m] = np.nan

            for t in range(T):
                src_t = t - sh

                if t < warm or src_t < 0 or np.isnan(rollmax[src_t, m]):
                    out[t, m] = s_val

                else:
                    out[t, m] = rollmax[src_t, m]


        return out


    @staticmethod
    @njit(cache = True)
    def _roll_quantile(data: np.ndarray, window: np.ndarray, quantiles: np.ndarray, scale: float = 1.0) -> np.ndarray:

        if data.ndim == 1:
            T = data.shape[0]
            M = quantiles.shape[0]
            arr_mat = np.empty((T, M), dtype = np.float32)

            for m in range(M):
                arr_mat[:, m] = data * scale

        elif data.ndim == 2:
            T, M = data.shape
            arr_mat = data * scale

            if quantiles.shape[0] != M:

                raise ValueError("[WARNING] quantiles size must match columns of 2D input")

        else:

            raise ValueError("[WARNING] Only 1D or 2D data supported.")

        quantile_mat = np.empty((T, M), dtype = np.float32)

        for m in range(M):
            col = arr_mat[:, m]
            w = int(window[m])
            q = quantiles[m] / 100.0

            for t in range(T):
                low = 0 if t < w else t - w
                seg = col[low: t]

                if seg.size == 0:
                    quantile_mat[t, m] = np.nan

                else:
                    quantile_mat[t, m] = np.nanquantile(seg, q)


        return quantile_mat


    @staticmethod
    @njit(cache = True)
    def _convergence(arr_mat: np.ndarray, pattern_window: np.ndarray) -> np.ndarray:

        T, M = arr_mat.shape
        conver_mat = np.zeros((11, T, M), dtype = np.float32)
        COUNT, PEAK_MAX, PEAK_MIN, PEAK_MEAN, PEAK_DIFF, BOTTOM_MAX, BOTTOM_MIN, BOTTOM_MEAN, BOTTOM_DIFF, PEAK_CNT, BOTTOM_CNT = range(11)

        # Level fields hold price magnitudes; a warmup / no-pattern bar left at the zeros-init 0.0 reads as a
        # real price of 0 and poisons any cross-sectional z-score or ranking downstream. Seed the eight level
        # rows with NaN (= "no pattern here"); the three count rows (COUNT / PEAK_CNT / BOTTOM_CNT) keep 0,
        # which is a true count of zero.
        conver_mat[PEAK_MAX, :, :] = np.nan
        conver_mat[PEAK_MIN, :, :] = np.nan
        conver_mat[PEAK_MEAN, :, :] = np.nan
        conver_mat[PEAK_DIFF, :, :] = np.nan
        conver_mat[BOTTOM_MAX, :, :] = np.nan
        conver_mat[BOTTOM_MIN, :, :] = np.nan
        conver_mat[BOTTOM_MEAN, :, :] = np.nan
        conver_mat[BOTTOM_DIFF, :, :] = np.nan

        for m in range(M):

            pw = pattern_window[m]
            arr = arr_mat[:, m]

            for t in range(T):

                count = 0
                peak_max = -1e9
                peak_min = 1e9
                peak_sum = 0.0
                peak_cnt = 0
                bottom_max = -1e9
                bottom_min = 1e9
                bottom_sum = 0.0
                bottom_cnt = 0

                if t < pw:
                    start = 0

                else:
                    start = t - pw

                sub = arr[start: t]
                L = sub.shape[0]

                if L >= 4:

                    for i in range(L - 3):

                        peak_hit = False
                        peak_val = 0.0

                        if sub[i] < sub[i + 1] and sub[i + 1] > sub[i + 2] and sub[i + 1] > sub[i + 3]:
                            # Require i+1 to beat the interior bar i+2 as well, else a still-rising run like
                            # [1,5,9,2] fires a false peak at 5 (skipping the true peak 9 at i+2); cond2 then
                            # correctly catches i+2. Only the demonstrated false-positive path is tightened;
                            # cond2/cond3's wider-lag tolerance is left as-designed.
                            peak_hit = True
                            peak_val = sub[i + 1]

                        elif sub[i] < sub[i + 2] and sub[i + 2] > sub[i + 3]:
                            peak_hit = True
                            peak_val = sub[i + 2]

                        elif (i + 4) < L and sub[i] < sub[i + 2] and sub[i + 2] > sub[i + 4]:
                            peak_hit = True
                            peak_val = sub[i + 2]

                        if peak_hit:

                            count += 1
                            peak_max = max(peak_max, peak_val)
                            peak_min = min(peak_min, peak_val)
                            peak_sum += peak_val
                            peak_cnt += 1

                        bottom_hit = False
                        bottom_val = 0.0

                        if sub[i] > sub[i + 1] and sub[i + 1] < sub[i + 2] and sub[i + 1] < sub[i + 3]:
                            # Symmetric to the peak fix: require i+1 to undercut the interior bar i+2 as well,
                            # else a still-falling run fires a false trough at i+1 and masks the true bottom at
                            # i+2 (caught by cond2). cond2/cond3 wider-lag tolerance left as-designed.
                            bottom_hit = True
                            bottom_val = sub[i + 1]

                        elif sub[i] > sub[i + 2] and sub[i + 2] < sub[i + 3]:
                            bottom_hit = True
                            bottom_val = sub[i + 2]

                        elif (i + 4) < L and sub[i] > sub[i + 2] and sub[i + 2] < sub[i + 4]:
                            bottom_hit = True
                            bottom_val = sub[i + 2]

                        if bottom_hit:

                            count += 1
                            bottom_max = max(bottom_max, bottom_val)
                            bottom_min = min(bottom_min, bottom_val)
                            bottom_sum += bottom_val
                            bottom_cnt += 1

                if peak_cnt > 0:

                    conver_mat[PEAK_MAX, t, m] = peak_max
                    conver_mat[PEAK_MIN, t, m] = peak_min
                    conver_mat[PEAK_MEAN, t, m] = peak_sum / peak_cnt
                    conver_mat[PEAK_DIFF, t, m] = peak_max - peak_min
                    conver_mat[PEAK_CNT, t, m] = peak_cnt

                if bottom_cnt > 0:

                    conver_mat[BOTTOM_MAX, t, m] = bottom_max
                    conver_mat[BOTTOM_MIN, t, m] = bottom_min
                    conver_mat[BOTTOM_MEAN, t, m] = bottom_sum / bottom_cnt
                    conver_mat[BOTTOM_DIFF, t, m] = bottom_max - bottom_min
                    conver_mat[BOTTOM_CNT, t, m] = bottom_cnt

                conver_mat[COUNT, t, m] = count


        return conver_mat


    @staticmethod
    @njit(cache = True)
    def _divergence(arr_mat: np.ndarray, price_mat: np.ndarray, pattern_window: np.ndarray, dd_period_2nd: np.ndarray) -> np.ndarray:

        T, M = arr_mat.shape
        diver_mat = np.zeros((6, T, M), dtype = np.bool_)
        PEAK_FAKE, PEAK_NEW, PEAK, BOTTOM_FAKE, BOTTOM_NEW, BOTTOM = range(6)

        for m in range(M):
            dd_period_2nd_arr = np.int16(dd_period_2nd[m])

            for t in range(T):

                low = int(max(0, t - pattern_window[m]))
                high = t
                max_idx = -1
                sec_max_idx = -1
                min_idx = -1
                sec_min_idx = -1

                for i in range(low, high):

                    if np.isnan(arr_mat[i, m]):

                        continue

                    if max_idx == -1 or arr_mat[i, m] > arr_mat[max_idx, m]:
                        max_idx = i

                if max_idx == -1:

                    continue

                for i in range(low, high):

                    if i == max_idx or np.isnan(arr_mat[i, m]):

                        continue

                    if sec_max_idx == -1 or arr_mat[i, m] > arr_mat[sec_max_idx, m]:
                        sec_max_idx = i

                if sec_max_idx == -1:

                    continue

                if price_mat[max_idx, m] < price_mat[sec_max_idx, m]:
                    delta = max_idx - sec_max_idx

                    if abs(delta) == 1:
                        diver_mat[PEAK_FAKE, t, m] = True

                    elif delta >= dd_period_2nd_arr:
                        diver_mat[PEAK_NEW, t, m] = True

                    elif -delta >= dd_period_2nd_arr:
                        diver_mat[PEAK, t, m] = True

                for i in range(low, high):

                    if np.isnan(arr_mat[i, m]):

                        continue

                    if min_idx == -1 or arr_mat[i, m] < arr_mat[min_idx, m]:
                        min_idx = i

                if min_idx == -1:

                    continue

                for i in range(low, high):

                    if i == min_idx or np.isnan(arr_mat[i, m]):

                        continue

                    if sec_min_idx == -1 or arr_mat[i, m] < arr_mat[sec_min_idx, m]:
                        sec_min_idx = i

                if sec_min_idx == -1:

                    continue

                if price_mat[min_idx, m] > price_mat[sec_min_idx, m]:
                    delta = min_idx - sec_min_idx

                    if abs(delta) == 1:
                        diver_mat[BOTTOM_FAKE, t, m] = True

                    elif delta >= dd_period_2nd_arr:
                        diver_mat[BOTTOM_NEW, t, m] = True

                    elif -delta >= dd_period_2nd_arr:
                        diver_mat[BOTTOM, t, m] = True


        return diver_mat


    @staticmethod
    def _hma(price: np.ndarray, period: int) -> np.ndarray:

        price_arr = price.astype(np.float64).reshape(-1)
        half = max(1, int(period // 2))
        wma_half = talib.WMA(price_arr, timeperiod = half)
        wma_full = talib.WMA(price_arr, timeperiod = int(period))
        raw = 2 * wma_half - wma_full
        sqrt_p = max(1, int(np.sqrt(period)))
        hma = talib.WMA(raw.astype(np.float64), timeperiod = sqrt_p)


        return hma.astype(np.float32)


    @staticmethod
    def _acc_return(price: np.ndarray, period: int) -> np.ndarray:

        series = pd.Series(np.asarray(price, dtype = np.float64).reshape(-1))
        gross = 1.0 + series.pct_change()


        return np.asarray(gross.rolling(int(period)).apply(np.prod, raw = True).sub(1.0).values, dtype = np.float32)


    class Barra:
        """
        #### What It Does
        Holds the general, asset-class-agnostic **Barra-style risk-factor** kernels (Beta, Momentum, Size, Non-linear Size, Residual Volatility, Non-linear Beta, Liquidity) as a passive namespace of static primitives, following the MSCI Barra USE4 descriptor conventions (winsorize -> cap-weighted-mean / equal-weighted-std standardization -> composite descriptor weighting -> orthogonalization).

        #### Responsibility
        Stores the Barra-exposure compute primitives ONLY. It is a nested namespace inside `FactorLibrary`, never instantiated: `FactorManager.get("BARRA_*")` computes a Barra exposure by dispatching (via the `kernel_class` / `calc_name` spec fields) to one of these kernels. Each kernel receives a NAMED PANEL DICT (`{field: T x N ndarray}`) declared by the spec's `input_fields`, so the same code runs on any asset class; a declared field that is entirely absent / all-NaN is a fail-loud error upstream, not a silent NaN exposure. **Weighting caliber**: the cross-sectional cap-weight for standardization (2), the market index / beta (3), and orthogonalization (4) is ADV = rolling-mean(Close x Volume) (`_adv`). Two signal calibers stay stocks: SIZE's notional is `ln(Close x OI)` (1) and LIQUIDITY's turnover denominator is `Volume / OI` (5). Standardization / winsorization / orthogonalization / ADV are shared operators here, not per-asset-class code.

        #### How To Use
        Do not instantiate. Build the engine with `FactorManager(test_data)` and call `.get.single("BARRA_BETA", ...)` etc.; the resulting exposures are the basis a caller residualizes its raw score against. Windows, half-lives, ADV window and composite weights live in the `FactorEngine.json` spec (`defaults` + the `kernel_config` block), so they are config, not hardcoded constants. (The upstream `Neutralize` / `Attribution` helper modules that consumed these exposures are not vendored here -- the harness scores its own free-form factor -- so a caller supplies its own residualization.)

        #### Usage Example
        ```python
        FMgr = FactorManager(test_data)
        art  = FMgr.get.single("BARRA_BETA")                            # CubeArtifact (also registered)
        beta = FMgr.get_frame(art.feature_id)                           # T x N exposure frame
        size = FMgr.get_frame(FMgr.get.single("BARRA_SIZE").feature_id)
        FMgr.register("MY_SIGNAL", raw_score.shift(1))                  # causal lag -> hand downstream to BacktestEngine
        ```
        """


        @staticmethod
        def _ewma_weights(window: int, halflife: float) -> np.ndarray:

            # exponential half-life weights over `window` points, index 0 = OLDEST, last = most recent.
            window = max(1, int(window))
            lam = 0.5 ** (1.0 / max(1e-9, float(halflife)))
            age = np.arange(window - 1, -1, -1, dtype = np.float64)   # oldest has largest age
            w = lam ** age
            total = w.sum()


            return (w / total) if total > 0 else np.full(window, 1.0 / window, dtype = np.float64)


        @staticmethod
        def _log_returns(price: np.ndarray) -> np.ndarray:

            # T x N log returns; first row NaN. Non-positive prices -> NaN (log-undefined).
            p = np.asarray(price, dtype = np.float64)
            safe = np.where(p > 0.0, p, np.nan)
            out = np.full_like(safe, np.nan)
            out[1:] = np.log(safe[1:] / safe[:-1])


            return out


        @staticmethod
        def _market_return(ret: np.ndarray, cap: np.ndarray) -> np.ndarray:

            # cap-weighted cross-sectional mean return per date -> (T,) market proxy.
            finite = np.isfinite(ret) & np.isfinite(cap) & (cap > 0.0)
            w = np.where(finite, cap, 0.0)
            wsum = w.sum(axis = 1)
            num = np.nansum(np.where(finite, w * ret, 0.0), axis = 1)


            return np.where(wsum > 0.0, num / np.where(wsum > 0.0, wsum, 1.0), np.nan)


        @staticmethod
        def _winsor(x: np.ndarray, n_std: float = 3.0) -> np.ndarray:

            # cross-sectional (per-row) clip to +/- n_std about the equal-weighted mean.
            xf = np.where(np.isfinite(x), x, np.nan)
            mu = np.nanmean(xf, axis = 1, keepdims = True)
            sd = np.nanstd(xf, axis = 1, keepdims = True)
            lo = mu - n_std * sd
            hi = mu + n_std * sd
            clipped = np.minimum(np.maximum(x, lo), hi)


            return np.where(np.isfinite(x) & np.isfinite(sd) & (sd > 0.0), clipped, x)


        @staticmethod
        def _cs_standardize(x: np.ndarray, cap: np.ndarray) -> np.ndarray:

            # Barra descriptor standardization: cap-weighted mean 0, equal-weighted std 1, per date.
            finite = np.isfinite(x)
            capw = np.where(finite & np.isfinite(cap) & (cap > 0.0), cap, 0.0)
            wsum = capw.sum(axis = 1, keepdims = True)
            mu = np.where(wsum > 0.0, np.nansum(np.where(finite, capw * x, 0.0), axis = 1, keepdims = True) / np.where(wsum > 0.0, wsum, 1.0), np.nan)
            xf = np.where(finite, x, np.nan)
            sd = np.nanstd(xf, axis = 1, keepdims = True)
            z = (x - mu) / np.where(sd > 0.0, sd, np.nan)


            return np.where(finite & (sd > 0.0) & np.isfinite(mu), z, np.nan).astype(np.float64)


        @staticmethod
        def _orthogonalize(target: np.ndarray, basis: np.ndarray, cap: np.ndarray) -> np.ndarray:

            # per-date cap-weighted (WLS) residual of target on [1, basis]; regression-weighted, Barra-style.
            out = np.full(target.shape, np.nan, dtype = np.float64)

            for t in range(target.shape[0]):
                y = target[t]
                b = basis[t]
                w = cap[t]
                mask = np.isfinite(y) & np.isfinite(b) & np.isfinite(w) & (w > 0.0)

                if int(mask.sum()) < 3:

                    continue

                rw = np.sqrt(w[mask])
                x_mat = np.column_stack([np.ones(int(mask.sum())), b[mask]])
                beta, *_ = np.linalg.lstsq(x_mat * rw[:, None], y[mask] * rw, rcond = None)
                out[t, np.flatnonzero(mask)] = y[mask] - x_mat @ beta


            return out


        @staticmethod
        def _wroll_mean(mat: np.ndarray, w: np.ndarray, min_frac: float = 0.75) -> np.ndarray:

            # NaN-aware fixed-window weighted rolling mean along axis 0; w oldest-first (len = window).
            # weights renormalized over the valid taps in each window; gated when coverage < min_frac.
            mat = np.asarray(mat, dtype = np.float64)
            window = int(w.shape[0])
            n_rows, n_cols = mat.shape
            finite = np.isfinite(mat)
            m0 = np.where(finite, mat, 0.0)
            vf = finite.astype(np.float64)
            kern = w[::-1]                                          # np.convolve flips the kernel back to oldest-first
            num = np.empty((n_rows, n_cols), dtype = np.float64)
            den = np.empty((n_rows, n_cols), dtype = np.float64)

            for j in range(n_cols):
                num[:, j] = np.convolve(m0[:, j], kern, mode = "full")[:n_rows]
                den[:, j] = np.convolve(vf[:, j], kern, mode = "full")[:n_rows]

            out = np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), np.nan)
            out = np.where(den >= float(min_frac), out, np.nan)

            if window > 1:
                out[: window - 1, :] = np.nan


            return out


        @staticmethod
        def _wroll_std(mat: np.ndarray, w: np.ndarray, min_frac: float = 0.75) -> np.ndarray:

            # NaN-aware fixed-window weighted rolling std (population) along axis 0.
            m1 = FactorLibrary.Barra._wroll_mean(mat, w, min_frac = min_frac)
            m2 = FactorLibrary.Barra._wroll_mean(np.asarray(mat, dtype = np.float64) ** 2, w, min_frac = min_frac)
            var = m2 - m1 ** 2


            return np.sqrt(np.where(np.isfinite(var) & (var > 0.0), var, np.nan))


        @staticmethod
        def _adv(price: np.ndarray, volume: np.ndarray, window: int) -> np.ndarray:

            # Dollar-volume ADV = trailing rolling mean of (price x volume). This is the cross-sectional
            # cap-weight under the ADV caliber: it feeds standardization (2), the market index / beta (3),
            # and orthogonalization (4) wherever a cap-weight was previously used. Rolling (trailing) => no
            # look-ahead; a ~window warmup band and a tilt toward liquid names are intrinsic to the caliber.
            p = np.asarray(price, dtype = np.float64)
            v = np.asarray(volume, dtype = np.float64)
            dv = np.where((p > 0.0) & np.isfinite(v) & (v >= 0.0), p * v, np.nan)
            w = max(1, int(window))


            return pd.DataFrame(dv).rolling(w, min_periods = max(1, int(w * 0.75))).mean().to_numpy()


        @staticmethod
        def _beta_raw(price: np.ndarray, cap: np.ndarray, window: int, halflife: float):

            # time-varying EWMA market beta + regression residual (both raw, T x N).
            # market = cap-weighted cross-sectional mean return; beta_t = ewm-cov(r, R) / ewm-var(R).
            ret = FactorLibrary.Barra._log_returns(price)
            mkt = FactorLibrary.Barra._market_return(ret, cap)
            r_df = pd.DataFrame(ret)
            r_mkt = pd.Series(mkt)
            minp = max(2, int(window))
            ewm_kw = dict(halflife = float(halflife), min_periods = minp, adjust = True)
            mean_r = r_df.ewm(**ewm_kw).mean()
            mean_m = r_mkt.ewm(**ewm_kw).mean()
            mean_rm = r_df.mul(r_mkt, axis = 0).ewm(**ewm_kw).mean()
            mean_mm = (r_mkt * r_mkt).ewm(**ewm_kw).mean()
            cov = mean_rm.sub(mean_r.mul(mean_m, axis = 0))
            var = (mean_mm - mean_m * mean_m).replace(0.0, np.nan)
            beta = cov.div(var, axis = 0)
            alpha = mean_r.sub(beta.mul(mean_m, axis = 0))
            resid = r_df.sub(alpha).sub(beta.mul(r_mkt, axis = 0))


            return beta.to_numpy(dtype = np.float64), resid.to_numpy(dtype = np.float64)


        @staticmethod
        def _beta(panels: dict, params: dict, config: dict) -> np.ndarray:

            # Barra BETA style: standardized time-varying market beta. Cross-sectional cap-weight = ADV (2/3).
            close = panels["Close"]
            adv = FactorLibrary.Barra._adv(close, panels["Volume"], int(config.get("adv_window", 21)))
            window = int(params.get("window", 252))
            halflife = float(params.get("halflife", 63.0))
            beta_raw, _ = FactorLibrary.Barra._beta_raw(close, adv, window, halflife)
            z = FactorLibrary.Barra._cs_standardize(FactorLibrary.Barra._winsor(beta_raw), adv)


            return np.asarray(z, dtype = np.float32)


        @staticmethod
        def _momentum(panels: dict, params: dict, config: dict) -> np.ndarray:

            # Barra MOMENTUM (RSTR) style: half-life-weighted trailing log-return sum, skipping the most recent `lag` days.
            close = panels["Close"]
            adv = FactorLibrary.Barra._adv(close, panels["Volume"], int(config.get("adv_window", 21)))
            window = int(params.get("window", 120))
            lag = int(params.get("lag", 5))
            halflife = float(params.get("halflife", 60.0))
            ret = FactorLibrary.Barra._log_returns(close)
            w = FactorLibrary.Barra._ewma_weights(window, halflife)
            rstr = FactorLibrary.Barra._wroll_mean(ret, w)

            if lag > 0:
                shifted = np.full_like(rstr, np.nan)
                shifted[lag:, :] = rstr[:-lag, :]
                rstr = shifted

            z = FactorLibrary.Barra._cs_standardize(FactorLibrary.Barra._winsor(rstr), adv)


            return np.asarray(z, dtype = np.float32)


        @staticmethod
        def _size(panels: dict, params: dict, config: dict) -> np.ndarray:

            # Barra SIZE (LNCAP): signal = ln(Close x OI-notional) [caliber (1) = OI, a stock]; cross-sectional
            # weight = ADV [caliber (2)]. The two calibers are DISTINCT panels and must never collapse to one.
            close = panels["Close"]
            p = np.asarray(close, dtype = np.float64)
            oi = np.asarray(panels["OpenInterest"], dtype = np.float64)
            adv = FactorLibrary.Barra._adv(close, panels["Volume"], int(config.get("adv_window", 21)))
            notional = np.where((p > 0.0) & (oi > 0.0), p * oi, np.nan)
            raw = np.log(notional)
            z = FactorLibrary.Barra._cs_standardize(FactorLibrary.Barra._winsor(raw), adv)


            return np.asarray(z, dtype = np.float32)


        @staticmethod
        def _nlsize(panels: dict, params: dict, config: dict) -> np.ndarray:

            # Barra NON-LINEAR SIZE (NLSIZE): standardized Size cubed, orthogonalized to Size, re-standardized.
            close = panels["Close"]
            adv = FactorLibrary.Barra._adv(close, panels["Volume"], int(config.get("adv_window", 21)))
            z_size = FactorLibrary.Barra._size(panels, params, config).astype(np.float64)
            cubed = FactorLibrary.Barra._winsor(z_size ** 3)
            ortho = FactorLibrary.Barra._orthogonalize(cubed, z_size, adv)
            # standardize-only after orthogonalization: the affine standardize preserves the cap-weighted
            # orthogonality exactly, whereas a post-ortho winsor (nonlinear clip) re-injects correlation to the base.
            z = FactorLibrary.Barra._cs_standardize(ortho, adv)


            return np.asarray(z, dtype = np.float32)


        @staticmethod
        def _nlbeta(panels: dict, params: dict, config: dict) -> np.ndarray:

            # Barra NON-LINEAR BETA (NLBETA, USE4 12th style): standardized Beta cubed, orthogonalized to Beta.
            close = panels["Close"]
            adv = FactorLibrary.Barra._adv(close, panels["Volume"], int(config.get("adv_window", 21)))
            z_beta = FactorLibrary.Barra._beta(panels, params, config).astype(np.float64)
            cubed = FactorLibrary.Barra._winsor(z_beta ** 3)
            ortho = FactorLibrary.Barra._orthogonalize(cubed, z_beta, adv)
            # standardize-only after orthogonalization (see _nlsize): preserves cap-weighted orthogonality to Beta.
            z = FactorLibrary.Barra._cs_standardize(ortho, adv)


            return np.asarray(z, dtype = np.float32)


        @staticmethod
        def _resid_vol(panels: dict, params: dict, config: dict) -> np.ndarray:

            # Barra RESIDUAL VOLATILITY: w0*DASTD + w1*CMRA + w2*HSIGMA, orthogonalized to Beta. Weight = ADV.
            close = panels["Close"]
            adv = FactorLibrary.Barra._adv(close, panels["Volume"], int(config.get("adv_window", 21)))
            _w = config.get("weights")
            window = int(params.get("window", 252))
            hl_beta = float(params.get("halflife", 63.0))
            hl_dastd = float(params.get("hl_dastd", 42.0))
            hl_hsigma = float(params.get("hl_hsigma", 63.0))
            w_comp = list(_w) if _w is not None and len(list(_w)) == 3 else [0.75, 0.15, 0.10]

            ret = FactorLibrary.Barra._log_returns(close)
            beta_raw, resid = FactorLibrary.Barra._beta_raw(close, adv, window, hl_beta)

            dastd = FactorLibrary.Barra._wroll_std(ret, FactorLibrary.Barra._ewma_weights(window, hl_dastd))
            hsigma = FactorLibrary.Barra._wroll_std(resid, FactorLibrary.Barra._ewma_weights(window, hl_hsigma))

            # CMRA: cumulative-range of trailing 1..12 monthly (21d) cumulative log returns.
            n_rows, n_cols = ret.shape
            months = 12
            step = 21
            r_df = pd.DataFrame(ret)
            z_stack = np.full((months, n_rows, n_cols), np.nan, dtype = np.float64)

            for k in range(1, months + 1):
                win = step * k
                z_stack[k - 1] = r_df.rolling(win, min_periods = max(2, int(win * 0.75))).sum().to_numpy()

            cmra = np.nanmax(z_stack, axis = 0) - np.nanmin(z_stack, axis = 0)
            cmra = np.where(np.all(~np.isfinite(z_stack), axis = 0), np.nan, cmra)

            z_dastd = FactorLibrary.Barra._cs_standardize(FactorLibrary.Barra._winsor(dastd), adv)
            z_cmra = FactorLibrary.Barra._cs_standardize(FactorLibrary.Barra._winsor(cmra), adv)
            z_hsigma = FactorLibrary.Barra._cs_standardize(FactorLibrary.Barra._winsor(hsigma), adv)

            stack = np.stack([z_dastd, z_cmra, z_hsigma], axis = 0)
            wvec = np.asarray(w_comp, dtype = np.float64).reshape(3, 1, 1)
            avail = np.isfinite(stack)
            wnum = np.nansum(np.where(avail, stack * wvec, 0.0), axis = 0)
            wden = np.nansum(np.where(avail, wvec, 0.0), axis = 0)
            composite = np.where(wden > 0.0, wnum / np.where(wden > 0.0, wden, 1.0), np.nan)

            z_beta = FactorLibrary.Barra._cs_standardize(FactorLibrary.Barra._winsor(beta_raw), adv)
            ortho = FactorLibrary.Barra._orthogonalize(composite, z_beta, adv)
            # standardize-only after orthogonalization (see _nlsize): preserves cap-weighted orthogonality to Beta.
            z = FactorLibrary.Barra._cs_standardize(ortho, adv)


            return np.asarray(z, dtype = np.float32)


        @staticmethod
        def _liquidity(panels: dict, params: dict, config: dict) -> np.ndarray:

            # Barra LIQUIDITY: w0*STOM + w1*STOQ + w2*STOA, share-turnover on 1/3/12-month horizons.
            # turnover DENOMINATOR = OI [caliber (5), a stock]; cross-sectional weight = ADV [caliber (2)].
            close = panels["Close"]
            adv = FactorLibrary.Barra._adv(close, panels["Volume"], int(config.get("adv_window", 21)))
            _w = config.get("weights")
            w_comp = list(_w) if _w is not None and len(list(_w)) == 3 else [0.35, 0.35, 0.30]
            v = np.asarray(panels["Volume"], dtype = np.float64)
            s = np.asarray(panels["OpenInterest"], dtype = np.float64)
            turnover = np.where((s > 0.0) & np.isfinite(v) & (v >= 0.0), v / s, np.nan)
            t_df = pd.DataFrame(turnover)

            sum_1 = t_df.rolling(21, min_periods = 16).sum().to_numpy()
            sum_3 = t_df.rolling(63, min_periods = 47).sum().to_numpy()
            sum_12 = t_df.rolling(252, min_periods = 189).sum().to_numpy()

            stom = np.log(np.where(sum_1 > 0.0, sum_1, np.nan))
            stoq = np.log(np.where(sum_3 > 0.0, sum_3 / 3.0, np.nan))
            stoa = np.log(np.where(sum_12 > 0.0, sum_12 / 12.0, np.nan))

            z_stom = FactorLibrary.Barra._cs_standardize(FactorLibrary.Barra._winsor(stom), adv)
            z_stoq = FactorLibrary.Barra._cs_standardize(FactorLibrary.Barra._winsor(stoq), adv)
            z_stoa = FactorLibrary.Barra._cs_standardize(FactorLibrary.Barra._winsor(stoa), adv)

            stack = np.stack([z_stom, z_stoq, z_stoa], axis = 0)
            wvec = np.asarray(w_comp, dtype = np.float64).reshape(3, 1, 1)
            avail = np.isfinite(stack)
            wnum = np.nansum(np.where(avail, stack * wvec, 0.0), axis = 0)
            wden = np.nansum(np.where(avail, wvec, 0.0), axis = 0)
            composite = np.where(wden > 0.0, wnum / np.where(wden > 0.0, wden, 1.0), np.nan)
            z = FactorLibrary.Barra._cs_standardize(FactorLibrary.Barra._winsor(composite), adv)


            return np.asarray(z, dtype = np.float32)
