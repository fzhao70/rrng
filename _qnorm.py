"""R's qnorm5 (normal quantile function), Wichura (1988) algorithm AS 241.

Bit-for-bit with R's src/nmath/qnorm.c for finite p in (0,1), mu, sigma. Vectorized
over a numpy array of probabilities. Used by norm_rand (R's default Inversion rnorm).
"""
import numpy as np


def qnorm(p, mu=0.0, sigma=1.0):
    """Vectorized R qnorm(p, mu, sigma, lower_tail=TRUE, log_p=FALSE)."""
    p = np.asarray(p, dtype=np.float64)
    q = p - 0.5
    val = np.empty_like(p)

    # --- central region |q| <= 0.425 ---
    central = np.abs(q) <= 0.425
    if np.any(central):
        qc = q[central]
        r = 0.180625 - qc * qc
        num = (((((((r * 2509.0809287301226727 +
                    33430.575583588128105) * r + 67265.770927008700853) * r +
                    45921.953931549871457) * r + 13731.693765509461125) * r +
                    1971.5909503065514427) * r + 133.14166789178437745) * r +
                    3.387132872796366608)
        den = (((((((r * 5226.495278852854561 +
                    28729.085735721942674) * r + 39307.89580009271061) * r +
                    21213.794301586595867) * r + 5394.1960214247511077) * r +
                    687.1870074920579083) * r + 42.313330701600911252) * r + 1.0)
        val[central] = qc * num / den

    # --- tails ---
    tail = ~central
    if np.any(tail):
        qt = q[tail]
        pt = p[tail]
        # r = sqrt(-log( q<0 ? p : 1-p ))
        rr = np.where(qt < 0.0, pt, 1.0 - pt)
        rr = np.sqrt(-np.log(rr))

        out = np.empty_like(qt)
        near = rr <= 5.0
        far = ~near

        if np.any(near):
            r = rr[near] - 1.6
            num = (((((((r * 7.7454501427834140764e-4 +
                        0.0227238449892691845833) * r + 0.24178072517745061177) * r +
                        1.27045825245236838258) * r + 3.64784832476320460504) * r +
                        5.7694972214606914055) * r + 4.6303378461565452959) * r +
                        1.42343711074968357734)
            den = (((((((r * 1.05075007164441684324e-9 +
                        5.475938084995344946e-4) * r + 0.0151986665636164571966) * r +
                        0.14810397642748007459) * r + 0.68976733498510000455) * r +
                        1.6763848301838038494) * r + 2.05319162663775882187) * r + 1.0)
            out[near] = num / den

        if np.any(far):
            r = rr[far] - 5.0
            num = (((((((r * 2.01033439929228813265e-7 +
                        2.71155556874348757815e-5) * r + 0.0012426609473880784386) * r +
                        0.026532189526576123093) * r + 0.29656057182850489123) * r +
                        1.7848265399172913358) * r + 5.4637849111641143699) * r +
                        6.6579046435011037772)
            den = (((((((r * 2.04426310338993978564e-15 +
                        1.4215117583164458887e-7) * r + 1.8463183175100546818e-5) * r +
                        7.868691311456132591e-4) * r + 0.0148753612908506148525) * r +
                        0.13692988092273580531) * r + 0.59983220655588793769) * r + 1.0)
            out[far] = num / den

        out = np.where(qt < 0.0, -out, out)
        val[tail] = out

    return mu + sigma * val
