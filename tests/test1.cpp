TEST(TensorTest, BasicOperations) {
    Tensor<double, 2> t1(2, 3);
    t1.fill(1.0);
    
    Tensor<double, 2> t2(2, 3);
    t2.fill(2.0);
    
    auto result = t1 + t2;
    EXPECT_EQ(result(0,0), 3.0);
}