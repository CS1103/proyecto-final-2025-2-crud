TEST(DenseTest, Backward) {
    Dense<double> layer(2, 1, ...);
    
    Tensor<double, 2> input(1, 2);
    input = {1.0, 2.0};
    
    auto output = layer.forward(input);
    
    Tensor<double, 2> grad_output(1, 1);
    grad_output = {1.0};
    
    auto grad_input = layer.backward(grad_output);
    EXPECT_EQ(grad_input.shape()[0], 1);
    EXPECT_EQ(grad_input.shape()[1], 2);
}