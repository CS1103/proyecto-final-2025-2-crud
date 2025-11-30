TEST(DenseTest, ForwardPass) {
    Dense<double> layer(2, 3,
        [](auto& w) { w.fill(0.5); },
        [](auto& b) { b.fill(0.1); }
    );
    
    Tensor<double, 2> input(1, 2);
    input = {1.0, 2.0};
    
    auto output = layer.forward(input);
    // output debería ser (1*0.5 + 2*0.5 + 0.1) para cada neurona
    EXPECT_NEAR(output(0,0), 1.6, 1e-6);
}