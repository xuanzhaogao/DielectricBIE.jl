# FFTW's threads callback. FFTW.jl runs FFTW's parallel loops on Julia tasks (its spawnloop), but its caller waits in
# @sync and so yields while the jobs run, and the jobs read data on the caller's C stack (FFTW's spawn_loop and the
# parent plan's apply): that data was found overwritten mid-loop, a segfault in FFTW's spawn_apply (FINUFFT's
# threaded FFTs share FFTW.jl's libfftw3, so FINUFFT calls with nthreads > 1 go through it). The replacement below
# never yields: the caller claims jobs itself and spins until the ones taken by helper tasks are done (the approach of
# FFTW.jl PR #344, unreleased as of FFTW.jl 1.10). Installed by __init__, after FFTW.jl's own (lazy) installation.
function _fftw_spawnloop(f::Ptr{Cvoid}, fdata::Ptr{Cvoid}, elsize::Csize_t, num::Cint, ::Ptr{Cvoid})
    next = Threads.Atomic{Int}(0); done = Threads.Atomic{Int}(0)
    work() = while true
        i = Threads.atomic_add!(next, 1)
        i >= num && break
        ccall(f, Ptr{Cvoid}, (Ptr{Cvoid},), fdata + elsize * i)
        Threads.atomic_add!(done, 1)
    end
    for _ in 2:num
        Threads.@spawn work()
    end
    work()
    while done[] < num
        ccall(:jl_cpu_pause, Cvoid, ())
        GC.safepoint()
    end
    return nothing
end

function _install_fftw_spawnloop()
    (FFTW.fftw_provider == "fftw" && Threads.nthreads() > 1) || return
    # FFTW.jl's lazy init: loading either library (re)installs its callback in both, so load both first
    Base.Libc.Libdl.dlopen(FFTW.libfftw3); Base.Libc.Libdl.dlopen(FFTW.libfftw3f)
    cb = @cfunction(_fftw_spawnloop, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Cint, Ptr{Cvoid}))
    ccall((:fftw_threads_set_callback, FFTW.libfftw3), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), cb, C_NULL)
    ccall((:fftwf_threads_set_callback, FFTW.libfftw3f), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), cb, C_NULL)
    return
end
