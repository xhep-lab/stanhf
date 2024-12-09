#include <ostream>
#include <stan/math.hpp>

template <typename T, typename U>
U poisson_real_lpdf(const T &k, const T &lambda, std::ostream *pstream__) {
  U r = 0;
  for (int i = 0; i < k.size(); ++i) {
    r += k[i] * stan::math::log(lambda[i]) - lambda[i] -
         stan::math::lgamma(k[i] + 1.);
  }
  return r;
}

template <typename T> using Vector = Eigen::VectorX<T>;

template <typename T> using tuple = std::tuple<T, T>;

template <typename T> using map = Eigen::Map<Vector<T>>;

template <typename T, typename U, typename V>
Vector<T> term_interp(const T &alpha, const map<U> &x,
                      const tuple<Vector<V>> &lu, std::ostream *pstream__) {

  const auto l = std::get<0>(lu);
  const auto u = std::get<1>(lu);

  if (alpha > 1.) {
    return alpha * (u - x);
  }

  if (alpha < -1.) {
    return alpha * (x - l);
  }

  const auto s = 0.5 * (u - l);
  const auto a = 0.0625 * (u + l - 2. * x);
  const auto alpha_squared = alpha * alpha;
  const auto r =
      alpha_squared * (alpha_squared * (alpha_squared * 3. - 10.) + 15.);

  return r * a + alpha * s;
}

template <typename U, typename V>
U factor_interp(U alpha, tuple<V> lu, std::ostream *pstream__) {

  const auto l = std::get<0>(lu);
  const auto u = std::get<1>(lu);

  if (alpha > 1.) {
    return stan::math::pow(u, alpha);
  }

  if (alpha < -1.) {
    return stan::math::pow(l, -alpha);
  }

  const auto log_u = stan::math::log(u);
  const auto log_l = stan::math::log(l);

  Eigen::Vector<U, 6> b;
  b << u - 1., 1. / l - 1., log_u * u, log_l / l, log_u * log_u * u,
      log_l * log_l / l;

  static const Eigen::Matrix<U, 6, 6> m =
      (Eigen::Matrix<U, 6, 6>() << 15. / 16., -15. / 16., -7. / 16., -7. / 16.,
       1. / 16., -1. / 16., 3. / std::pow(2, 2), 3. / std::pow(2, 2), -9. / 16.,
       9. / 16., 1. / 16., 1. / 16., -5. / std::pow(8, 3), 5. / std::pow(8, 3),
       5. / std::pow(8, 2), 5. / std::pow(8, 2), -1. / 8., 1. / 8.,
       3. / (-std::pow(2, 4)), 3. / (-std::pow(2, 4)), -7. / (-std::pow(8, 3)),
       7. / (-std::pow(8, 3)), -1. / std::pow(8, 2), -1. / std::pow(8, 2),
       3. / std::pow(16, 5), -3. / std::pow(16, 5), -3. / std::pow(16, 4),
       -3. / std::pow(16, 4), 1. / std::pow(16, 3), -1. / std::pow(16, 3),
       1. / std::pow(2, 6), 1. / std::pow(2, 6), -5. / std::pow(16, 5),
       5. / std::pow(16, 5), 1. / std::pow(16, 4), 1. / std::pow(16, 4))
          .finished();

  Eigen::Vector<U, 6> r;
  r[0] = alpha;
  for (int i = 1; i < r.size(); i++) {
    r[i] = r[i - 1] * alpha;
  }

  return 1. + r.transpose() * m * b;
}
