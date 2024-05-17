Units = Dict{String, Union{Unitful.FreeUnits, Unitful.Quantity}}("eV" => u"eV", "fs" => u"fs", "K" => u"K", "cm^-1" => u"h*c*cm^-1", "u" => u"u", "Angstrom" => u"Å", "A" => u"Å", "kg" => u"kg", "m" => u"m")
au2SI(unit::Unitful.FreeUnits, x::Number) = ustrip(auconvert(unit, x))
au2SI(unit::Unitful.Quantity, x::Number) = x/austrip(unit)

function ParseCommandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--input"
            help = "Input file"
            arg_type = String
            default = "input.in"
    end
    return parse_args(s)
end

function Parameters()
    args = ParseCommandline()
    for line in eachrow(readdlm(args["input"], '='; comments = true))
        key = strip(line[1])
        val = line[2]
        if key == "observables"
            args[key] = Symbol.(strip.(split(val, ',')))
        elseif  key == "representation"
            args[key] = Symbol(strip(val))
        else
            if isa(val, Number)
                args[key] = val
            else
                val = strip.(split(val, ','))
                args[key] = austrip(parse(Float64, string(val[1])) *  Units[val[2]])
            end
        end
    end
    return args
end
