#!/usr/bin/env julia

module Utils

export concat, files

using DataFrames, CSV

"""
    files(name; location="quantum")

Return all the files in the output folder for the given `location` that
have the given `name`.

## Arguments
- `name`: the name of the files
- `location="quantum"`: the location in the output folder. Can be `"classical"`
or `"qunatum"`

"""
function files(name; location="quantum")
    prefix = "../../output/$location/"
    re = r"n[0-9]+-b[0-9]+\.[0-9]+-d[0-9]+\.[0-9]+"
    has_data(f) = any(contains.(readdir(joinpath(prefix, f)), name))
    folders = filter(f->ismatch(re, f) && has_data(f), readdir(prefix))

    [joinpath(prefix, folders[i], name) for i=1:length(folders)]
end

"""
    concat(name)

Concatenate the `DataFrame`s in the output folder with the given `name`
and return a `DataFrame` containing all the data. See also [`files`](@ref).

## Arguments
- `name`: the name of the files
- `location="quantum"`: the location in the output folder. Can be `"classical"`
or `"qunatum"`

"""
function concat(name; location="quantum")
    filenames = files(name; location=location)
    df = CSV.read(filenames[1])
    for i=2:length(filenames)
        append!(df, CSV.read(filenames[i]))
    end

    return df
end

end  # module Utils
