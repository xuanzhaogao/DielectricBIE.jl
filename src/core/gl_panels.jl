# Gauss-Legendre × Gauss-Legendre panels and their aspect ratio.
#
# A GL×GL panel is a `FlatPanel`, or a `TensorPanel`/`CurvedPanel` whose rules in u and v are both
# Gauss-Legendre (decided by the rules, not by `kind`: a GL corner panel is GL×GL too). The near quadrature
# treats each one as one SCTL element, accurate only for moderate aspect ratios: the meshers split GL×GL
# panels to aspect ratio ≤ GL_MAX_ASPECT and the near quadrature refuses larger ones (`check_gl_aspect`).
# GCQ panels (a rule with several fine segments) have no limit and are never split.
#
# The aspect ratio is longer side / shorter side; a curved panel's sides are its mean chords across u and
# across v at 1/4, 1/2 and 3/4 of the other parameter (the measure of the SCTL extension's `_curved_split`).

"""
    GL_MAX_ASPECT

Largest aspect ratio (`panel_aspect`, longer side / shorter side) of a Gauss-Legendre × Gauss-Legendre panel,
4.0: the meshers split GL×GL panels until they are within it, and `check_gl_aspect` refuses larger ones (with a
relative slack of 1e-8).
"""
const GL_MAX_ASPECT = 4.0

# a panel passes when its aspect ratio is at most this (GL_MAX_ASPECT up to rounding)
const _GL_ASPECT_LIMIT = GL_MAX_ASPECT * (1 + 1e-8)

"""
    is_gl_panel(p) -> Bool

Whether `p` is a Gauss-Legendre × Gauss-Legendre panel: every `FlatPanel`; a `TensorPanel` or `CurvedPanel`
when neither of its rules `ru`, `rv` is a GCQ rule (whatever its `kind`, e.g. a GL corner panel).
"""
is_gl_panel(::FlatPanel) = true
is_gl_panel(p::Union{TensorPanel, CurvedPanel}) = !is_gcq(p.ru) && !is_gcq(p.rv)

_aspect(a, b) = Float64(max(a, b) / min(a, b))

# mean chords of the map y(u, v) of [0, 1]²: across u (y(1, t) - y(0, t)) and across v, at t = 1/4, 1/2, 3/4
function _mean_chords(y)
    su = sum(norm(y(1.0, t) .- y(0.0, t)) for t in (0.25, 0.5, 0.75)) / 3
    sv = sum(norm(y(t, 1.0) .- y(t, 0.0)) for t in (0.25, 0.5, 0.75)) / 3
    return su, sv
end

"""
    panel_aspect(p) -> Float64

Aspect ratio of a panel, longer side / shorter side: for a 3D `FlatPanel` from its corners (lb, rb, rt, lt; the
mean of opposite sides), for a `TensorPanel` from `Lu` and `Lv`, for a `CurvedPanel` from its mean chords across u
and across v at 1/4, 1/2 and 3/4 of the other parameter.
"""
function panel_aspect(p::FlatPanel{T, 3}) where T
    c = p.corners
    return _aspect(norm(c[2] .- c[1]) + norm(c[3] .- c[4]), norm(c[4] .- c[1]) + norm(c[3] .- c[2]))
end

panel_aspect(p::TensorPanel) = _aspect(p.Lu, p.Lv)

panel_aspect(p::CurvedPanel) = _aspect(_mean_chords((u, v) -> curved_lpoint(p, u, v))...)

"""
    check_gl_aspect(p, k::Int)

Throw an `ArgumentError` naming panel `k` when the GL×GL panel `p` has `panel_aspect(p)` above `GL_MAX_ASPECT`
(relative slack 1e-8; a degenerate panel, of aspect `Inf` or `NaN`, fails too). Returns `nothing`; a GCQ panel
(`is_gl_panel(p)` false) has no limit and always passes.
"""
function check_gl_aspect(p, k::Int)
    is_gl_panel(p) || return nothing
    a = panel_aspect(p)
    a <= _GL_ASPECT_LIMIT ||
        throw(ArgumentError("panel $k: GL×GL aspect ratio $(a) exceeds GL_MAX_ASPECT = 4 (large aspect ratios are not supported yet)"))
    return nothing
end

# the number of equal pieces a GL×GL panel of aspect ratio a is cut into along its longer side: ceil(a / GL_MAX_ASPECT),
# with the slack of check_gl_aspect (a = 8 (1 + 1e-15) gives two pieces, not three)
_gl_pieces(a) = ceil(Int, a / _GL_ASPECT_LIMIT)

# every GL×GL TensorPanel above the limit cut along its longer side into _gl_pieces equal GL panels (same normal,
# anchor, plane, rules; kind :gl); other panels unchanged
function _split_gl_aspect(panels::Vector{TensorPanel{T}}) where T
    out = TensorPanel{T}[]
    for p in panels
        a = panel_aspect(p)
        if !is_gl_panel(p) || !(a > _GL_ASPECT_LIMIT) || !isfinite(a)
            push!(out, p); continue
        end
        n = _gl_pieces(a)
        for i in 0:n-1
            if p.Lu >= p.Lv
                o = p.o .+ (p.Lu * i / n) .* p.eu; Lu = p.Lu / n; Lv = p.Lv
            else
                o = p.o .+ (p.Lv * i / n) .* p.ev; Lu = p.Lu; Lv = p.Lv / n
            end
            push!(out, TensorPanel(p.normal, p.anchor, o, p.eu, p.ev, Lu, Lv, p.ru, p.rv, :gl, p.plane))
        end
    end
    return out
end

# the parameter rectangles (U0, U1, V0, V1) that the GL×GL panel on [U0, U1] × [V0, V1] of the patch map f is cut into:
# along its longer physical direction (mean chords of f, the measure of panel_aspect) into _gl_pieces equal parts, and
# again until every piece is within the limit (a curved panel's pieces need not have exactly 1/n of its aspect ratio)
function _gl_aspect_rects(f, U0, U1, V0, V1; out = NTuple{4, Float64}[], depth = 0)
    su, sv = _mean_chords((u, v) -> f(U0 + u * (U1 - U0), V0 + v * (V1 - V0)))
    a = _aspect(su, sv)
    if !(a > _GL_ASPECT_LIMIT) || !isfinite(a) || depth >= 16      # degenerate panels are left to check_gl_aspect
        push!(out, (U0, U1, V0, V1)); return out
    end
    n = _gl_pieces(a)
    cut(x0, x1) = [i == 0 ? x0 : i == n ? x1 : x0 + (x1 - x0) * i / n for i in 0:n]    # exact ends
    if su >= sv
        b = cut(U0, U1)
        foreach(i -> _gl_aspect_rects(f, b[i], b[i+1], V0, V1; out, depth = depth + 1), 1:n)
    else
        b = cut(V0, V1)
        foreach(i -> _gl_aspect_rects(f, U0, U1, b[i], b[i+1]; out, depth = depth + 1), 1:n)
    end
    return out
end
