using NuclearSurfaceVibrations
using .Classical

function λlist(Elist, Blist=0.55, Dlist=0.4)
    for B in Blist, D in Dlist
        @progress "λs" for i = 1:length(Elist)
            prefix = "../../output/classical/B$B-D$D/E$(Elist[i])"
            λs = λmap(Elist[i], B=B, D=D)
            plt = histogram(λs, nbins=50)
            savefig(plt, "$prefix/lyapunov_hist.pdf")
            coloredpoincare(Elist[i], λs, name="lyapunov", B=B, D=D)
        end
    end
end

function galilist(Elist, Blist=0.55, Dlist=0.4)
    for B in Blist, D in Dlist
        pmap((i)->(chaoticity = Gali.galimap(Elist[i], B=B, D=D);
            prefix = "../../output/classical/B$B-D$D/E$(Elist[i])";
            plt = histogram(chaoticity, nbins=50);
            savefig(plt, "$prefix/gali_hist.pdf")),
            PmapProgressMeter.Progress(length(Elist)), 1:length(Elist))
        @progress "poincare gali" for i in 1:length(Elist)
            chaoticity = Gali.galimap(Elist[i], B=B, D=D)
            plt = coloredpoincare(Elist[i], chaoticity, name="gali", B=B, D=D)
        end
    end
end


B = [0.25]
λlist(25:25.:700, B)
galilist(100:10.:150)


df = concat(r"z0.csv", location="classical/B$(B[1])-D0.4", re=r"E[0-9]+\.[0-9]+", filter=[:E, :λs])

by(df, :E, df->DataFrame(λ = maximum(df[:λs]))) |> @orderby(_.E) |>
    @df plot(:E, :λ, m=4, xlabel="E", ylabel="\\lambda", label="B = $(B[1])")
savefig("../../output/classical/B$(B[1])-D0.4/lambda(E).pdf")


######

λs = λmap(10., B=0.55)
histogram(λs, nbins=50)

coloredpoincare(E, λs, name="lyapunov", B=B)
coloredpoincare(E, chaoticity, name="gali", B=B)

plt = energy_error(sim, E, (A, B, D))
plt2 = scatter(sim, vars=(4, 2), msc=nothing, ms=2.)

q0, p0, N = generateInitialConditions(E, params=(A,B,D))
sim = poincaremap(q0, p0, N, prefix, B=B)
scatter(sim, vars=(4,2), msc=nothing, zcolor=(x,y)->V([x,y],(A,B,D)))

scatter(rand(10), rand(10), zcolor=(x,y,z)->x^2)
