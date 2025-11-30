TEST(LossTest, MSELoss) {
    Tensor<double, 2> pred(2, 1);
    pred = {0.8, 0.2};
    
    Tensor<double, 2> target(2, 1);
    target = {1.0, 0.0};
    
    MSELoss<double> loss(pred, target);
    double loss_value = loss.loss();
    
    // MSE = ((0.8-1)^2 + (0.2-0)^2) / 2 = 0.04/2 = 0.02
    EXPECT_NEAR(loss_value, 0.04, 1e-6);
}