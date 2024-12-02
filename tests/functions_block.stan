functions{
real poisson_real_lpdf(vector k, vector lambda) {
  return sum(k .* log(lambda) - lambda - lgamma(k + 1.));
}

vector term_interp(real alpha, vector x, tuple(vector, vector) lu) {
  if (alpha > 1.) {
    return alpha * (lu.2 - x);
  }
  
  if (alpha < -1.) {
    return alpha * (x - lu.1);
  }
  
  vector[size(x)] s = 0.5 * (lu.2 - lu.1);
  vector[size(x)] a = 0.0625 * (lu.2 + lu.1 - 2. * x);
  real r = alpha ^ 2 * (alpha ^ 2 * (alpha ^ 2 * 3. - 10.) + 15.);
  
  return r * a + alpha * s;
}

real factor_interp(real alpha, tuple(real, real) lu) {
  if (alpha > 1.) {
    return lu.2 ^ alpha;
  }
  
  if (alpha < -1.) {
    return (1. / lu.1) ^ alpha;
  }
  
  vector[6] b = [lu.2 - 1., 1. / lu.1 - 1., log(lu.2) * lu.2,
                 log(lu.1) / lu.1, log(lu.2) ^ 2 * lu.2,
                 log(lu.1) ^ 2 / lu.1]';
  
  matrix[6, 6] m = [[15. / 16., -15. / 16., -7. / 16., -7. / 16., 1. / 16.,
                     -1. / 16.],
                    [3. / (2. ^ 2), 3. / (2. ^ 2), -9. / 16., 9. / 16.,
                     1. / 16., 1. / 16.],
                    [-5. / (8. ^ 3), 5. / (8. ^ 3), 5. / (8. ^ 2),
                     5. / (8. ^ 2), -1. / 8., 1. / 8.],
                    [3. / (-2. ^ 4), 3. / (-2. ^ 4), -7. / (-8. ^ 3),
                     7. / (-8. ^ 3), -1. / (8. ^ 2), -1. / (8. ^ 2)],
                    [3. / (16. ^ 5), -3. / (16. ^ 5), -3. / (16. ^ 4),
                     -3. / (16. ^ 4), 1. / (16. ^ 3), -1. / (16. ^ 3)],
                    [1. / (2. ^ 6), 1. / (2. ^ 6), -5. / (16. ^ 5),
                     5. / (16. ^ 5), 1. / (16. ^ 4), 1. / (16. ^ 4)]];
  
  return 1. + alpha ^ linspaced_row_vector(6, 1, 6) * m * b;
}

}