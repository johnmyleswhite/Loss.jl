export mean_absolute_deviation
export median_absolute_deviation
export maximum_absolute_deviation
export minimum_absolute_deviation
export root_mean_absolute_deviation
export sum_absolute_deviation
export mean_squared_error
export median_squared_error
export maximum_squared_error
export minimum_squared_error
export root_mean_squared_error
export sum_squared_error
export mean_squared_log_error
export median_squared_log_error
export maximum_squared_log_error
export minimum_squared_log_error
export root_mean_squared_log_error
export sum_squared_log_error
export mean_hinge_loss
export median_hinge_loss
export maximum_hinge_loss
export minimum_hinge_loss
export root_mean_hinge_loss
export sum_hinge_loss
export mean_zero_one_loss
export median_zero_one_loss
export maximum_zero_one_loss
export minimum_zero_one_loss
export root_mean_zero_one_loss
export sum_zero_one_loss
export mean_log_loss
export median_log_loss
export maximum_log_loss
export minimum_log_loss
export root_mean_log_loss
export sum_log_loss

module Loss
    function absolute_deviation(true_value::Real, predicted_value::Real)
        return abs(true_value - predicted_value)
    end

    function squared_error(true_value::Real, predicted_value::Real)
        return (true_value - predicted_value)^2
    end

    function squared_log_error(true_value::Real, predicted_value::Real)
        return (log1p(true_value) - log1p(predicted_value))^2
    end

    function hinge_loss(true_value::Real, predicted_value::Real)
        return max(0, 1 - true_value * predicted_value)
    end

    function zero_one_loss(true_value::Union(Real, String), predicted_value::Union(Real, String))
        return float64(true_value == predicted_value)
    end

    function log_loss(true_value::Real, predicted_value::Real)
        return -(true_value * log(predicted_value) + (1 - true_value) * log(1 - predicted_value))
    end

    function root_mean{T <: Real}(values::Array{T})
        return sqrt(mean(values))
    end

    error_fs = (:absolute_deviation, :squared_error, :squared_log_error,
                :hinge_loss, :zero_one_loss, :log_loss)
    aggregation_fs = (:mean, :median, :maximum, :minimum, :root_mean, :sum)

    for error_f in error_fs
        for aggregation_f in aggregation_fs
            f_name = symbol(string(aggregation_f, "_", error_f))
            @eval begin
                function $(f_name){S <: Real, T <: Real}(a::Array{S}, b::Array{T})
                    if size(a) != size(b)
                        error("Sizes of true values and predicted values must match")
                    end
                    n = length(a)
                    elementwise_errors = Array(Float64, n)
                    for i in 1:n
                        elementwise_errors[i] = $(error_f)(a[i], b[i])
                    end
                    return $(aggregation_f)(elementwise_errors)
                end
            end
        end
    end
end
