using Random 
using LinearAlgebra
using CSV, DataFrames
using ArgParse
import Statistics
using Dates
const sqrt3 = sqrt(3.0)
@inline function unitrand(::Type{Float64})
    return sqrt3 * (2.0 * rand(Float64) - 1.0)
end
@inline function randbetween(a, b)
    a,b=promote(a,b)
    return rand(typeof(a))*(b-a)+a 
end

function bd!(x, y, x0, y0, amp)
    N = length(x)
    msdx = 0.0
    msdy = 0.0
    meanx = 0.0
    meany = 0.0

    for j in 1:N
        
        x[j] += amp * unitrand(Float64)
        y[j] += amp * unitrand(Float64)

        msdx += (x[j] - x0[j])^2
        msdy += (y[j] - y0[j])^2
        meanx += x[j] - x0[j]
        meany += y[j] - y0[j]
        
        # xx = x[j]
        # yy = y[j]
        # #periodic bc in x and y
        # if xx > L
        #     xx -= L
        # elseif xx < 0
        #     xx += L
        # end
        # if yy > L
        #     yy -= L
        # elseif yy < 0
        #     yy += L
        # end
        # x[j], y[j] = xx, yy
    end

    return msdx / N, msdy / N, meanx / N, meany / N
end
function simulate(Dt, N, tf, dt)
    L = 1.0
    Nstep = Int64(floor(tf/dt))
    amp = Float64(sqrt(2*Dt*dt))
    x0 = [randbetween(0, L) for _ in 1:N]
    y0 = [randbetween(0, L) for _ in 1:N]

    x = copy(x0)
    y = copy(y0)

    t = 0.0
    t_arr = zeros(Nstep)
    msdx = zeros(Nstep)
    msdy = zeros(Nstep)
    meanx = zeros(Nstep)
    meany = zeros(Nstep)
    for i in 1:Nstep
        if i % 10000 == 0
            @info "At timestep $i time  $(dt*i)"
        end
        t_arr[i] = t
        msdx[i], msdy[i], meanx[i], meany[i] = bd!(x, y, x0, y0, amp)
        t += dt
    end
    return t_arr , msdx, msdy, meanx, meany
end
function parse_command_line()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--Dt"
            help = "Diffusion constant"
            arg_type = Float64
            default = 1.00
        "--dt"
            help = "time step"
            arg_type = Float64
            default = 0.001
        "--tf"
            help = "total time"
            arg_type = Float64
            default = 100.0
        "--N"
            help = "No of particles"
            arg_type = Int
            default = 100000
    end
    return parse_args(ARGS,s)
end
function main()
    args = parse_command_line()
    Dt = args["Dt"]
    dt = args["dt"]
    tf = args["tf"]
    N = args["N"]

    start_time = now()
    t_arr, msdx, msdy, meanx, meany = simulate(Dt, N, tf, dt)
    end_time = now()


    df = DataFrame(time = t_arr, meanx = meanx ,meany = meany, msd_x = msdx, msd_y = msdy)
    CSV.write("msd_data_D_$(Dt)_dt_$(dt)_N_$(N).csv", df)
    elapsed = end_time - start_time
    println("Elapsed time: $elapsed")
end


main()
