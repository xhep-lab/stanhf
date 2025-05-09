functions{
 
real poisson_real_lpdf(data vector k, vector lambda) {
  return sum(k .* log(lambda) - lambda - lgamma(k + 1.));
}

vector term_interp(real alpha, data vector x, data tuple(vector, vector) lu) {
  if (alpha > 1.) {
    return alpha * (lu.2 - x);
  }
  
  if (alpha < -1.) {
    return alpha * (x - lu.1);
  }
  
  vector[size(x)] s = 0.5 * (lu.2 - lu.1);
  vector[size(x)] a = 0.0625 * (lu.2 + lu.1 - 2. * x);
  real alpha_square = square(alpha);
  real r = alpha_square * (alpha_square * (alpha_square * 3. - 10.) + 15.);
  
  return r * a + alpha * s;
}

real factor_interp(real alpha, data tuple(real, real) lu) {
  if (alpha > 1.) {
    return lu.2 ^ alpha;
  }
  
  if (alpha < -1.) {
    return lu.1 ^ (-alpha);
  }
  
  real log_l = log(lu.1);
  real log_u = log(lu.2);
  
  vector[6] b = [lu.2 - 1., lu.1 - 1., log_u * lu.2, -log_l * lu.1,
                 square(log_u) * lu.2, square(log_l) * lu.1]';
  
  matrix[6, 6] m = [[0.9375, -0.9375, -0.4375, -0.4375, 0.0625, -0.0625],
                    [1.5, 1.5, -0.5625, 0.5625, 0.0625, 0.0625],
                    [-0.625, 0.625, 0.625, 0.625, -0.125, 0.125],
                    [-1.5, -1.5, 0.875, -0.875, -0.125, -0.125],
                    [0.1875, -0.1875, -0.1875, -0.1875, 0.0625, -0.0625],
                    [0.5, 0.5, -0.3125, 0.3125, 0.0625, 0.0625]];
  
  return 1. + alpha ^ linspaced_row_vector(6, 1, 6) * m * b;
}

}