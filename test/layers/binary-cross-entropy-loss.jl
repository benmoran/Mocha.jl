function test_binary_crossentropy_loss_layer(backend::Backend, tensor_dim, T, epsilon)
  println("-- Testing BinaryCrossEntropyLossLayer on $(typeof(backend)){$T}...")

  dims = abs(rand(Int,tensor_dim)) % 6 + 2


  println("    > $dims")

  #dims_label = copy(dims); dims_label[op_dim] = 1; dims_label = tuple(dims_label...)
  dims = tuple(dims...)

  prob = rand(T, dims)
  # ensure we don't give nans when we predict exact 0's or 1's
  prob[1] = 0.0
  prob[2] = 1.0

  label = rand(Int, dims) .> 0.5
  label = convert(Array{T}, label)

  prob_blob = make_blob(backend, prob)
  label_blob = make_blob(backend, label)
  diff_blob1  = make_blob(backend, prob)
  diff_blob2  = make_blob(backend, prob)

  # Now we've made the blob, we clip the predictions for calculating the expected loss
  prob[1] = eps(T)
  prob[2] = 1 - eps(T)
  inputs = Blob[prob_blob, label_blob]
  weight = 0.25
  layer = BinaryCrossEntropyLossLayer(bottoms=[:pred, :labels], weight=weight)
  state = setup(backend, layer, inputs, Blob[])

  forward(backend, state, inputs)

  @test !isnan(state.loss)
  expected_loss = convert(T, 0)

  for i = 1:prod(dims)
      expected_loss += -log(vec(prob)[i])*vec(label)[i]
      expected_loss += -log(1 - vec(prob)[i])*vec(1 - label)[i]
  end

  expected_loss /= dims[end]
  expected_loss *= weight

  @test -epsilon < 1 - state.loss/expected_loss < epsilon


  diffs = Blob[diff_blob1, diff_blob2]
  backward(backend, state, inputs, diffs)
  grad_pred = -weight * (label./prob - (1-label)./(1-prob) ) / dims[end]
  diff = similar(grad_pred)

  copy!(diff, diffs[1])

  @test !any(isnan(diff))
  @test all(-epsilon .< 1 - grad_pred./diff .< epsilon)

  grad_label = -weight * log(prob./(1-prob)) / dims[end]
  diff = similar(grad_pred)
  copy!(diff, diffs[2])

  @test all(-epsilon .< grad_label - diff .< epsilon)

  shutdown(backend, state)
end

function test_binary_crossentropy_loss_layer(backend::Backend, T, epsilon)
  for i in [2,4,5]
      test_binary_crossentropy_loss_layer(backend, i, T, epsilon)
  end
end

function test_binary_crossentropy_loss_layer(backend::Backend)
  test_binary_crossentropy_loss_layer(backend, Float32, 1e-5)
  test_binary_crossentropy_loss_layer(backend, Float64, 1e-6)
end

if test_cpu
  test_binary_crossentropy_loss_layer(backend_cpu)
end
if test_gpu
  test_binary_crossentropy_loss_layer(backend_gpu)
end
