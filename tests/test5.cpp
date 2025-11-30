TEST(ActivationTest, ReLU) {
    ReLU<double> relu;
    
    Tensor<double, 2> input(2, 2);
    input = {{-1.0, 2.0}, {-3.0, 4.0}};
    
    auto output = relu.forward(input);
    EXPECT_EQ(output(0,0), 0.0);
    EXPECT_EQ(output(0,1), 2.0);
    EXPECT_EQ(output(1,0), 0.0);
    EXPECT_EQ(output(1,1), 4.0);
}