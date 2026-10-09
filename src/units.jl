const PowersType = SVector{2, Rational{Int}}
argument_error(msg::String) = throw(ArgumentError(msg))


"""
Based on Wflow.jl https://github.com/Deltares/Wflow.jl

Store a unit as a product of powers, for instance:
m/s -> Unit(; m = 1, s = -1)

Positive and negative powers can be split to support e.g.
m²m⁻² -> Unit(; m = (2, 2))

The `absolute_temperature` flag indicates whether °C => K
requires a +273.15 shift.

Adding support for a new unit is easy:
- Add a field to the `Unit` struct
- Specify how it translates to SI standard units in `to_SI_data`
- If the symbol for the unit is different from its field name in `Unit`,
    add it to `UnitStrings`
"""
struct Unit
    # Temperature
    absolute_temperature::Bool # Expected to be first field!
    K::PowersType # Kelvin, SI standard
    degC::PowersType # degree Celsius
    # Time
    s::PowersType # second, SI standard
    ms::PowersType # millisecond
    min::PowersType # minute
    h::PowersType # hour
    d::PowersType # day
    y::PowersType #year
    dt::PowersType # time step
    # Length
    m::PowersType # meter, SI standard
    cm::PowersType # centimeter
    mm::PowersType # millimeter
    μm::PowersType # micrometer
    # Volume
    L::PowersType # liter
    # Mass
    kg::PowersType # kilogram, SI standard
    g::PowersType # gram
    t::PowersType # tonne
    # Energy
    J::PowersType # joule
    # Fraction
    percentage::PowersType # percentage, converted to unitless fraction in the SI standard
    ppm::PowersType # parts per million, converted to unitless fraction in the SI standard
    # degree
    deg::PowersType # degree (latitude and longitude)
    # Factor for converting a value in this unit to the above mentioned standard SI units (apart from dt)
    to_SI_factor_without_dt::Float64 # Expected to be last field!
end

const Units = fieldnames(Unit)[2:(end - 1)]
const N_UNITS = length(Units)
const STANDARD_UNITS = [:K, :s, :m, :kg, :J]

"""
Based on Wflow.jl https://github.com/Deltares/Wflow.jl
"""
function Unit(absolute_temperature, powers_all...)
    to_SI_factor_without_dt = 1.0
    for (i, powers) in enumerate(powers_all)
        unit = Units[i]
        (unit == :dt) && continue
        @assert all(≥(0), powers) "Expected non-negative input, got $powers for $unit."
        net_power = powers[2] - powers[1]
        if (unit ∉ STANDARD_UNITS) && !iszero(net_power)
            factor = to_SI_data[i].factor
            to_SI_factor_without_dt *= factor^net_power
        end
    end
    return Unit(absolute_temperature, powers_all..., to_SI_factor_without_dt)
end

"""
Based on Wflow.jl https://github.com/Deltares/Wflow.jl
"""
@generated function get_powers_tuple(unit::Unit)
    exprs = [:(getfield(unit, $(i + 1))) for i in 1:N_UNITS]  # +1 to skip absolute_temperature
    return :(tuple($(exprs...)))
end

"""
Based on Wflow.jl https://github.com/Deltares/Wflow.jl
"""
function Unit(; absolute_temperature = false, kwargs...)
    powers = ntuple(
        i -> begin
            unit = Units[i]
            powers = return if unit in keys(kwargs)
                val = kwargs[unit]

                if val isa Number
                    val > 0 ? (0, val) : (-val, 0)
                else
                    val
                end
            else
                (0 // 1, 0 // 1)
            end
            PowersType(powers)
        end, N_UNITS
    )
    for unit in keys(kwargs)
        (unit ∉ Units) && argument_error("Unrecognized unit $unit.")
    end
    return Unit(absolute_temperature, powers...)
end

"""
Based on Wflow.jl https://github.com/Deltares/Wflow.jl
"""
const to_SI_data = @NamedTuple{factor::Float64, unit_SI::Unit}[
    (factor = 1.0, unit_SI = Unit(; K = 1)), # K
    (factor = 1.0, unit_SI = Unit(; K = 1)), # degC
    (factor = 1.0, unit_SI = Unit(; s = 1)), # s
    (factor = 1.0e-3, unit_SI = Unit(; s = 1)), # ms
    (factor = 60.0, unit_SI = Unit(; s = 1)), # min
    (factor = 3600, unit_SI = Unit(; s = 1)), # h
    (factor = 86400.0, unit_SI = Unit(; s = 1)), # d
    (factor = 31536000.0, unit_SI = Unit(; s = 1)), # y
    (factor = NaN, unit_SI = Unit(; s = 1)), # dt
    (factor = 1.0, unit_SI = Unit(; m = 1)), # m
    (factor = 1.0e-2, unit_SI = Unit(; m = 1)), # cm
    (factor = 1.0e-3, unit_SI = Unit(; m = 1)), # mm
    (factor = 1.0e-6, unit_SI = Unit(; m = 1)), # μm
    (factor = 1.0e-3, unit_SI = Unit(; m = 3)), # L
    (factor = 1.0, unit_SI = Unit(; kg = 1)), # kg
    (factor = 1.0e-3, unit_SI = Unit(; kg = 1)), # g
    (factor = 1.0e3, unit_SI = Unit(; kg = 1)), # t
    (factor = 1.0, unit_SI = Unit(; kg = 1, m = 2, s = -2)), # J
    (factor = 1.0e-2, unit_SI = Unit()), # percentage
    (factor = 1.0e-6, unit_SI = Unit()), # ppm
    (factor = 1.0, unit_SI = Unit()), # degree
]

"""
Based on Wflow.jl https://github.com/Deltares/Wflow.jl
"""
function Base.:*(u1::Unit, u2::Unit)
    absolute_temperature_new = u1.absolute_temperature || u2.absolute_temperature
    powers1 = get_powers_tuple(u1)
    powers2 = get_powers_tuple(u2)
    powers_new = ntuple(i -> powers1[i] + powers2[i], N_UNITS)
    return Unit(absolute_temperature_new, powers_new...)
end

"""
Based on Wflow.jl https://github.com/Deltares/Wflow.jl
"""
function Base.:^(u::Unit, n::Rational{Int})
    powers = get_powers_tuple(u)
    powers_new = if (n > 0)
        ntuple(i -> powers[i] * n, N_UNITS)
    else
        ntuple(i -> reverse(powers[i] * -n), N_UNITS)
    end
    return Unit(u.absolute_temperature, powers_new...)
end

const UnitStrings = Dict{Symbol, String}(
    :degC => "°C",
    :percentage => "%",
    :d => "day",
    :dt => "Δt",
    :t => "ton",
    :deg => "°"
)

"""
Based on Wflow.jl https://github.com/Deltares/Wflow.jl
"""
function power_string(power::Rational{Int}, BMI_standard::Bool)
    (; num, den) = power
    return if isinteger(power)
        n = string(Int(power))
        BMI_standard ? n : replace(n, SUPERSCRIPT_NUMBERS...)
    else
        if BMI_standard
            "$num/$den"
        else
            num_str = replace(string(num), SUPERSCRIPT_NUMBERS...)
            den_str = replace(string(den), SUPERSCRIPT_NUMBERS...)
            "$(num_str)ᐟ$den_str"
        end
    end
end

"""
Represent the Unit as a string,
following the BMI standard if `BMI_standard = true`:
https://bmi.csdms.io/en/stable/bmi.var_funcs.html#get-var-units
Based on Wflow.jl https://github.com/Deltares/Wflow.jl
"""
function to_string(unit::Unit; BMI_standard = false)
    n_units = length(Units)
    symbols = ntuple(i -> get(UnitStrings, Units[i], String(Units[i])), Val(n_units))
    powers = ntuple(i -> getfield(unit, Units[i]), Val(n_units))
    out = String[]

    # Positive powers
    for (symbol, powers_) in zip(symbols, powers)
        power = powers_[2]
        if !iszero(power)
            term = isone(power) ? symbol : "$symbol$(power_string(power, BMI_standard))"
            push!(out, term)
        end
    end

    # Negative powers
    for (symbol, powers_) in zip(symbols, powers)
        power = powers_[1]
        if !iszero(power)
            push!(out, "$symbol$(power_string(-power, BMI_standard))")
        end
    end

    out = join(out, " ")
    # Dash if unitless
    return isempty(out) ? "-" : out
end

"""
Based on Wflow.jl https://github.com/Deltares/Wflow.jl
Convert a general unit to an SI standard unit
"""
function to_SI(unit::Unit)
    powers_all = get_powers_tuple(unit)
    unit_new = Unit(
        unit.absolute_temperature,
        ntuple(Returns(SVector(0 // 1, 0 // 1)), N_UNITS)...,
    )
    for (i, powers) in enumerate(powers_all)
        net_power = powers[2] - powers[1]
        if !iszero(net_power)
            unit_new *= to_SI_data[i].unit_SI^net_power
        end
    end
    return unit_new
end
# Fallback method for non-Floats, e.g. nothing, missing, Bool, Int
to_SI(x, unit::Unit; kwargs...) = x

"""
Based on Wflow.jl https://github.com/Deltares/Wflow.jl
In-place conversion of an array of values to the corresponding SI unit
"""
function to_SI!(x::AbstractArray, unit::Unit; dt_val::Union{Nothing, Number} = nothing)
    unit_ref = Ref(unit)
    @. x = to_SI(x, unit_ref; dt_val)
    return x
end

"""
Convert a value in SI unit to the given unit
Based on Wflow.jl https://github.com/Deltares/Wflow.jl
"""
function to_SI(x::AbstractArray, unit::Unit; dt_val::Union{Nothing, Number} = nothing)
    out = copy(x)
    return to_SI!(out, unit; dt_val)
end