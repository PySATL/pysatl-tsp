"""
    TSPBackend

Julia backend for pysatl_tsp.

The module mirrors the library's streaming architecture on the Julia side: each
operation is a small stateful object that consumes observations one at a time
(`step!`) and immediately produces a result, exactly like an `InductiveHandler`
does in Python. Operations can be linked into a `Chain`, which lets a sequence
of adjacent Julia-backed handlers be evaluated with a single call per
observation instead of one call per handler.

Missing observations are represented on the wire as a `(value, is_missing)`
pair rather than `nothing`, so that the Python/Julia boundary never depends on
implicit `None` conversion and a genuine `NaN` observation stays
distinguishable from "not enough data yet".
"""
module TSPBackend

export Op, Chain, sma, median_filter, chain_of, step_chain

"""
    Op

Base type for a stateful streaming operation.

Every concrete subtype is expected to declare the fields `len::Int`,
`min_periods::Int`, `values::Vector{Float64}` and `valid::Vector{Bool}`, which
together describe a sliding window of the most recent observations and a mask
marking which of them are present. Shared window bookkeeping is implemented
once in `_push!`.
"""
abstract type Op end

"""Result returned when an operation cannot produce a value yet."""
const MISSING_OUT = (NaN, true)

"""
    _push!(op, x, is_missing)

Append an observation to the operation's window, discarding the oldest one when
the window is full.
"""
function _push!(op::Op, x::Float64, is_missing::Bool)
    push!(op.values, x)
    push!(op.valid, !is_missing)
    if length(op.values) > op.len
        popfirst!(op.values)
        popfirst!(op.valid)
    end
    return nothing
end

# --------------------------------------------------------------------------
# Simple moving average
# --------------------------------------------------------------------------

"""Sliding-window arithmetic mean, ignoring missing observations."""
mutable struct SMA <: Op
    len::Int
    min_periods::Int
    values::Vector{Float64}
    valid::Vector{Bool}
end

"""
    sma(len, min_periods) -> SMA

Construct a simple moving average operation over a window of `len` points that
stays missing until at least `min_periods` present observations are available.
"""
sma(len::Integer, min_periods::Integer) = SMA(Int(len), Int(min_periods), Float64[], Bool[])

function step!(op::SMA, x::Float64, is_missing::Bool)
    _push!(op, x, is_missing)

    total = 0.0
    count = 0
    @inbounds for i in eachindex(op.values)
        if op.valid[i]
            total += op.values[i]
            count += 1
        end
    end

    count < op.min_periods && return MISSING_OUT
    return (total / count, false)
end

# --------------------------------------------------------------------------
# Median filter
# --------------------------------------------------------------------------

"""Sliding-window median, ignoring missing observations."""
mutable struct MedianFilter <: Op
    len::Int
    min_periods::Int
    values::Vector{Float64}
    valid::Vector{Bool}
    scratch::Vector{Float64}
end

"""
    median_filter(len, min_periods) -> MedianFilter

Construct a median filter over a window of `len` points that stays missing
until at least `min_periods` present observations are available.
"""
function median_filter(len::Integer, min_periods::Integer)
    return MedianFilter(Int(len), Int(min_periods), Float64[], Bool[], Float64[])
end

function step!(op::MedianFilter, x::Float64, is_missing::Bool)
    _push!(op, x, is_missing)

    empty!(op.scratch)
    @inbounds for i in eachindex(op.values)
        op.valid[i] && push!(op.scratch, op.values[i])
    end

    n = length(op.scratch)
    n < op.min_periods && return MISSING_OUT

    sort!(op.scratch)
    if isodd(n)
        return (op.scratch[div(n + 1, 2)], false)
    end
    return ((op.scratch[div(n, 2)] + op.scratch[div(n, 2) + 1]) / 2, false)
end

# --------------------------------------------------------------------------
# Chaining
# --------------------------------------------------------------------------

"""
    Chain

An ordered sequence of operations evaluated as a single unit, so that a run of
adjacent Julia-backed handlers costs one language boundary crossing per
observation instead of one per handler.
"""
struct Chain
    ops::Vector{Op}
end

"""
    chain_of(ops...) -> Chain

Build a chain from operations given in upstream-to-downstream order.
"""
chain_of(ops::Op...) = Chain(collect(Op, ops))

"""
    step_chain(chain, x, is_missing) -> (Float64, Bool)

Feed one observation through every operation of `chain` and return the final
`(value, is_missing)` pair.

Intermediate missing results are passed on to the next operation as missing
observations rather than skipped, which reproduces exactly what happens when
the same handlers are connected as separate Python handlers and one of them
yields `None`.
"""
function step_chain(chain::Chain, x::Real, is_missing::Bool)
    value = Float64(x)
    missing_flag = is_missing
    for op in chain.ops
        value, missing_flag = step!(op, value, missing_flag)
    end
    return (value, missing_flag)
end

end # module
