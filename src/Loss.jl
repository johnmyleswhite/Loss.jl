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
    res = 0.0
    n = numel(values)
    for i in 1:n
      res += values[i]
    end
    return sqrt(res / n)
  end

  error_fs = (:absolute_deviation, squared_error, :squared_log_error,
              :hinge_loss, :zero_one_loss, :log_loss)
  aggregation_fs = (:mean, :median, :max, :min, :root_mean, :sum)

  # TODO: Generate export statements?
  for error_f in error_fs
    for aggregation_f in aggregation_fs
      f_name = symbol(strcat(string(aggregation_f), "_", string(error_f)))
      @eval begin
        function $(f_name){S <: Real, T <: Real}(a::Array{S}, b::Array{T})
          if size(a) != size(b)
            error("Sizes of true values and predicted values must match")
          end
          n = numel(a)
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
