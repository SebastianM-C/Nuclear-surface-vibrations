#!/usr/bin/env julia

using JLD
using Plots
using DataFrames, CSV

prefix = "../../output/quantum/"
re = r"n[0-9]+-b[0-9]+\.[0-9]+-d[0-9]+\.[0-9]+"
has_data(f) = any(contains.(readdir(joinpath(prefix, f)), "perf_data.csv"))
folders = filter(f->ismatch(re, f) && has_data(f), readdir(prefix))

filenames = [joinpath(prefix, folders[i], "perf_data.csv") for i=1:length(folders)]

df = CSV.read(filenames[1])
for i=2:length(filenames)
    append!(df, CSV.read(filenames[i]))
end



print_timer()
