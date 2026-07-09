"""
    MovingAverage

Julia backend with moving average computations for pysatl_tsp.

Functions in this module operate on a single window of observations and are
called from Python handlers (see `julia/sma_handler.py`). Missing observations
are represented as `nothing` (Python `None` is converted automatically by
juliacall).
"""
module MovingAverage

"""
    sma(values, min_periods) -> Union{Float64, Nothing}

Simple moving average of `values`, ignoring `nothing` entries.

Returns `nothing` when the number of valid (non-`nothing`) observations in the
window is less than `min_periods`, mirroring the semantics of the pure-Python
`SMAHandler`.
"""
function sma(values::AbstractVector, min_periods::Integer)
    total = 0.0
    count = 0
    for v in values
        if v !== nothing
            total += Float64(v)
            count += 1
        end
    end
    count < min_periods && return nothing
    return total / count
end

end # module
