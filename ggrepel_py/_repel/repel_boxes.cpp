// Port of ggrepel's src/repel_boxes.cpp.
// Upstream: https://github.com/slowkow/ggrepel
// Pybind11 binding layer replaces Rcpp. Inner algorithm is verbatim.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;

// Exported convenience functions ---------------------------------------------

// Euclidean distance between two 2D points given as length-2 arrays.
static double euclid_vec(py::array_t<double> a, py::array_t<double> b) {
    auto ra = a.unchecked<1>();
    auto rb = b.unchecked<1>();
    return std::sqrt((ra(0) - rb(0)) * (ra(0) - rb(0)) +
                     (ra(1) - rb(1)) * (ra(1) - rb(1)));
}

static py::array_t<double> centroid_vec(py::array_t<double> b, double hjust,
                                        double vjust) {
    auto rb = b.unchecked<1>();
    py::array_t<double> out(2);
    auto ro = out.mutable_unchecked<1>();
    ro(0) = rb(0) + (rb(2) - rb(0)) * hjust;
    ro(1) = rb(1) + (rb(3) - rb(1)) * vjust;
    return out;
}

static bool intersect_circle_rectangle_vec(py::array_t<double> c,
                                           py::array_t<double> r) {
    auto rc = c.unchecked<1>();
    auto rr = r.unchecked<1>();
    double c_x = rc(0);
    double c_radius = rc(2);
    double r_x = (rr(2) + rr(0)) / 2;
    double r_halfwidth = std::abs(rr(0) - r_x);
    double cx = std::abs(c_x - r_x);
    double xDist = r_halfwidth + c_radius;
    if (cx > xDist) return false;
    double c_y = rc(1);
    double r_y = (rr(3) + rr(1)) / 2;
    double r_halfheight = std::abs(rr(1) - r_y);
    double cy = std::abs(c_y - r_y);
    double yDist = r_halfheight + c_radius;
    if (cy > yDist) return false;
    if (cx <= r_halfwidth || cy <= r_halfheight) return true;
    double xCornerDist = cx - r_halfwidth;
    double yCornerDist = cy - r_halfheight;
    double xCornerDistSq = xCornerDist * xCornerDist;
    double yCornerDistSq = yCornerDist * yCornerDist;
    double maxCornerDistSq = c_radius * c_radius;
    return xCornerDistSq + yCornerDistSq <= maxCornerDistSq;
}

static py::array_t<double> intersect_line_circle_vec(py::array_t<double> p1,
                                                     py::array_t<double> p2,
                                                     double r) {
    auto rp1 = p1.unchecked<1>();
    auto rp2 = p2.unchecked<1>();
    double theta = std::atan2(rp1(1) - rp2(1), rp1(0) - rp2(0));
    py::array_t<double> out(2);
    auto ro = out.mutable_unchecked<1>();
    ro(0) = rp2(0) + r * std::cos(theta);
    ro(1) = rp2(1) + r * std::sin(theta);
    return out;
}

static double euclid_pair(double x1, double y1, double x2, double y2) {
    double dx = x1 - x2, dy = y1 - y2;
    return std::sqrt(dx * dx + dy * dy);
}

static py::array_t<double> intersect_line_rectangle_vec(py::array_t<double> p1,
                                                        py::array_t<double> p2,
                                                        py::array_t<double> b) {
    auto rp1 = p1.unchecked<1>();
    auto rp2 = p2.unchecked<1>();
    auto rb = b.unchecked<1>();

    double dy = rp2(1) - rp1(1);
    double dx = rp2(0) - rp1(0);
    double slope = dy / dx;
    double intercept = rp2(1) - rp2(0) * slope;

    double retval[4][2];
    for (int i = 0; i < 4; i++) {
        retval[i][0] = -std::numeric_limits<double>::infinity();
        retval[i][1] = -std::numeric_limits<double>::infinity();
    }

    double x, y;
    if (dx != 0) {
        x = rb(0);
        y = dy == 0 ? rp1(1) : slope * x + intercept;
        if (rb(1) <= y && y <= rb(3)) {
            retval[0][0] = x;
            retval[0][1] = y;
        }
        x = rb(2);
        y = dy == 0 ? rp1(1) : slope * x + intercept;
        if (rb(1) <= y && y <= rb(3)) {
            retval[1][0] = x;
            retval[1][1] = y;
        }
    }
    if (dy != 0) {
        y = rb(1);
        x = dx == 0 ? rp1(0) : (y - intercept) / slope;
        if (rb(0) <= x && x <= rb(2)) {
            retval[2][0] = x;
            retval[2][1] = y;
        }
        y = rb(3);
        x = dx == 0 ? rp1(0) : (y - intercept) / slope;
        if (rb(0) <= x && x <= rb(2)) {
            retval[3][0] = x;
            retval[3][1] = y;
        }
    }
    int imin = 0;
    double dmin = std::numeric_limits<double>::infinity();
    for (int i = 0; i < 4; i++) {
        double d = euclid_pair(retval[i][0], retval[i][1], rp1(0), rp1(1));
        if (d < dmin) {
            dmin = d;
            imin = i;
        }
    }
    py::array_t<double> out(2);
    auto ro = out.mutable_unchecked<1>();
    ro(0) = retval[imin][0];
    ro(1) = retval[imin][1];
    return out;
}

static py::array_t<double> select_line_connection_vec(py::array_t<double> p1,
                                                      py::array_t<double> b) {
    auto rp1 = p1.unchecked<1>();
    auto rb = b.unchecked<1>();

    py::array_t<double> out(2);
    auto ro = out.mutable_unchecked<1>();

    bool top = false, left = false, right = false, bottom = false;

    if ((rp1(0) >= rb(0)) && (rp1(0) <= rb(2))) {
        ro(0) = rp1(0);
    } else if (rp1(0) > rb(2)) {
        ro(0) = rb(2);
        right = true;
    } else {
        ro(0) = rb(0);
        left = true;
    }

    if ((rp1(1) >= rb(1)) && (rp1(1) <= rb(3))) {
        ro(1) = rp1(1);
    } else if (rp1(1) > rb(3)) {
        ro(1) = rb(3);
        top = true;
    } else {
        ro(1) = rb(1);
        bottom = true;
    }

    double midx = (rb(0) + rb(2)) * 0.5;
    double midy = (rb(3) + rb(1)) * 0.5;
    double d = std::sqrt(std::pow(rp1(0) - ro(0), 2) +
                         std::pow(rp1(1) - ro(1), 2));

    if ((top || bottom) && !(left || right)) {
        double altd = std::sqrt(std::pow(rp1(0) - midx, 2) +
                                std::pow(rp1(1) - ro(1), 2));
        ro(0) = ro(0) + (midx - ro(0)) * d / altd;
    } else if ((left || right) && !(top || bottom)) {
        double altd = std::sqrt(std::pow(rp1(0) - ro(0), 2) +
                                std::pow(rp1(1) - midy, 2));
        ro(1) = ro(1) + (midy - ro(1)) * d / altd;
    } else if ((left || right) && (top || bottom)) {
        double altd1 = std::sqrt(std::pow(rp1(0) - midx, 2) +
                                 std::pow(rp1(1) - ro(1), 2));
        double altd2 = std::sqrt(std::pow(rp1(0) - ro(0), 2) +
                                 std::pow(rp1(1) - midy, 2));
        if (altd1 < altd2) {
            ro(0) = ro(0) + (midx - ro(0)) * d / altd1;
        } else {
            ro(1) = ro(1) + (midy - ro(1)) * d / altd2;
        }
    }
    return out;
}

// Main code for text label placement -----------------------------------------

struct Point {
    double x, y;
};

static Point operator-(const Point& a, const Point& b) { return {a.x - b.x, a.y - b.y}; }
static Point operator+(const Point& a, const Point& b) { return {a.x + b.x, a.y + b.y}; }
static Point operator/(const Point& a, const double& b) { return {a.x / b, a.y / b}; }
static Point operator*(const double& b, const Point& a) { return {a.x * b, a.y * b}; }
static Point operator*(const Point& a, const double& b) { return {a.x * b, a.y * b}; }

struct Box {
    double x1, y1, x2, y2;
};

static Box operator+(const Box& b, const Point& p) {
    return {b.x1 + p.x, b.y1 + p.y, b.x2 + p.x, b.y2 + p.y};
}

struct Circle {
    double x, y, r;
};

static double euclid(Point a, Point b) {
    Point dist = a - b;
    return std::sqrt(dist.x * dist.x + dist.y * dist.y);
}

static bool approximately_equal_fn(double x1, double x2) {
    return std::abs(x2 - x1) < (std::numeric_limits<double>::epsilon() * 100);
}

static bool line_intersect(Point p1, Point q1, Point p2, Point q2) {
    if (std::isnan(p1.x) || std::isnan(p1.y) || std::isnan(q1.x) ||
        std::isnan(q1.y) || std::isnan(p2.x) || std::isnan(p2.y) ||
        std::isnan(q2.x) || std::isnan(q2.y)) {
        return false;
    }
    if (q1.x == q2.x && q1.y == q2.y) return false;
    if (p1.x == q1.x && p1.y == q1.y) return false;
    if (p2.x == q2.x && p2.y == q2.y) return false;

    double dy1 = q1.y - p1.y;
    double dx1 = q1.x - p1.x;
    double slope1 = dy1 / dx1;
    double intercept1 = q1.y - q1.x * slope1;

    double dy2 = q2.y - p2.y;
    double dx2 = q2.x - p2.x;
    double slope2 = dy2 / dx2;
    double intercept2 = q2.y - q2.x * slope2;

    double x, y;

    if (approximately_equal_fn(dx1, 0.0)) {
        if (approximately_equal_fn(dx2, 0.0)) {
            return false;
        } else {
            x = p1.x;
            y = slope2 * x + intercept2;
        }
    } else if (approximately_equal_fn(dx2, 0.0)) {
        x = p2.x;
        y = slope1 * x + intercept1;
    } else {
        if (approximately_equal_fn(slope1, slope2)) {
            return false;
        }
        x = (intercept2 - intercept1) / (slope1 - slope2);
        y = slope1 * x + intercept1;
    }

    if (x < p1.x && x < q1.x) return false;
    if (x > p1.x && x > q1.x) return false;
    if (y < p1.y && y < q1.y) return false;
    if (y > p1.y && y > q1.y) return false;
    if (x < p2.x && x < q2.x) return false;
    if (x > p2.x && x > q2.x) return false;
    if (y < p2.y && y < q2.y) return false;
    if (y > p2.y && y > q2.y) return false;
    return true;
}

static Box put_within_bounds(Box b, Point xlim, Point ylim) {
    double width = std::fabs(b.x1 - b.x2);
    double height = std::fabs(b.y1 - b.y2);
    if (b.x1 < xlim.x) {
        b.x1 = xlim.x;
        b.x2 = b.x1 + width;
    } else if (b.x2 > xlim.y) {
        b.x2 = xlim.y;
        b.x1 = b.x2 - width;
    }
    if (b.y1 < ylim.x) {
        b.y1 = ylim.x;
        b.y2 = b.y1 + height;
    } else if (b.y2 > ylim.y) {
        b.y2 = ylim.y;
        b.y1 = b.y2 - height;
    }
    return b;
}

static Point centroid(Box b, double hjust, double vjust) {
    return {b.x1 + (b.x2 - b.x1) * hjust, b.y1 + (b.y2 - b.y1) * vjust};
}

static bool overlaps(Box a, Box b) {
    if (std::isnan(a.x1) || std::isnan(a.y1) || std::isnan(a.x2) ||
        std::isnan(a.y2) || std::isnan(b.x1) || std::isnan(b.y1) ||
        std::isnan(b.x2) || std::isnan(b.y2)) {
        return false;
    }
    return b.x1 <= a.x2 && b.y1 <= a.y2 && b.x2 >= a.x1 && b.y2 >= a.y1;
}

static bool overlaps(Circle c, Box r) {
    if (std::isnan(c.x) || std::isnan(c.y) || std::isnan(c.r) ||
        std::isnan(r.x1) || std::isnan(r.y1) || std::isnan(r.x2) ||
        std::isnan(r.y2)) {
        return false;
    }
    double c_x = c.x;
    double c_radius = c.r;
    double r_x = (r.x1 + r.x2) / 2;
    double r_halfwidth = std::abs(r.x1 - r_x);
    double cx = std::abs(c_x - r_x);
    double xDist = r_halfwidth + c_radius;
    if (cx > xDist) return false;
    double c_y = c.y;
    double r_y = (r.y1 + r.y2) / 2;
    double r_halfheight = std::abs(r.y1 - r_y);
    double cy = std::abs(c_y - r_y);
    double yDist = r_halfheight + c_radius;
    if (cy > yDist) return false;
    if (cx <= r_halfwidth || cy <= r_halfheight) return true;
    double xCornerDist = cx - r_halfwidth;
    double yCornerDist = cy - r_halfheight;
    double xCornerDistSq = xCornerDist * xCornerDist;
    double yCornerDistSq = yCornerDist * yCornerDist;
    double maxCornerDistSq = c_radius * c_radius;
    return xCornerDistSq + yCornerDistSq <= maxCornerDistSq;
}

static Point repel_force_both(Point a, Point b, double force = 0.000001) {
    double dx = std::fabs(a.x - b.x);
    double dy = std::fabs(a.y - b.y);
    double d2 = std::max(dx * dx + dy * dy, 0.0004);
    Point v = (a - b) / std::sqrt(d2);
    Point f = force * v / d2;
    if (dx > dy) {
        f.y = f.y * 2;
    } else {
        f.x = f.x * 2;
    }
    return f;
}

static Point repel_force_y(Point a, Point b, double force = 0.000001) {
    double dx = std::fabs(a.x - b.x);
    double dy = std::fabs(a.y - b.y);
    double d2 = std::max(dx * dx + dy * dy, 0.0004);
    Point v = {0, 1};
    if (a.y < b.y) v.y = -1;
    Point f = force * v / d2 * 2;
    return f;
}

static Point repel_force_x(Point a, Point b, double force = 0.000001) {
    double dx = std::fabs(a.x - b.x);
    double dy = std::fabs(a.y - b.y);
    double d2 = std::max(dx * dx + dy * dy, 0.0004);
    Point v = {1, 0};
    if (a.x < b.x) v.x = -1;
    Point f = force * v / d2 * 2;
    return f;
}

static Point repel_force(Point a, Point b, double force = 0.000001,
                         const std::string& direction = "both") {
    if (std::isnan(a.x) || std::isnan(a.y) || std::isnan(b.x) ||
        std::isnan(b.y)) {
        return {0, 0};
    }
    if (direction == "x") return repel_force_x(a, b, force);
    if (direction == "y") return repel_force_y(a, b, force);
    return repel_force_both(a, b, force);
}

static Point spring_force_both(Point a, Point b, double force) {
    Point v = (a - b);
    return force * v;
}

static Point spring_force_y(Point a, Point b, double force) {
    Point v = {0, (a.y - b.y)};
    return force * v;
}

static Point spring_force_x(Point a, Point b, double force) {
    Point v = {(a.x - b.x), 0};
    return force * v;
}

static Point spring_force(Point a, Point b, double force,
                          const std::string& direction) {
    if (std::isnan(a.x) || std::isnan(a.y) || std::isnan(b.x) ||
        std::isnan(b.y)) {
        return {0, 0};
    }
    if (direction == "x") return spring_force_x(a, b, force);
    if (direction == "y") return spring_force_y(a, b, force);
    return spring_force_both(a, b, force);
}

static std::vector<double> rescale(std::vector<double> v) {
    double min_value = *std::min_element(v.begin(), v.end());
    double max_value = *std::max_element(v.begin(), v.end());
    for (std::size_t i = 0; i < v.size(); i++) {
        v[i] = (v[i] - min_value) / max_value;
    }
    return v;
}

static py::dict repel_boxes2(py::array_t<double, py::array::c_style | py::array::forcecast> data_points,
                             py::array_t<double> point_size,
                             double point_padding_x, double point_padding_y,
                             py::array_t<double, py::array::c_style | py::array::forcecast> boxes,
                             py::array_t<double> xlim, py::array_t<double> ylim,
                             py::array_t<double> hjust, py::array_t<double> vjust,
                             double force_push, double force_pull,
                             double max_time, double max_overlaps,
                             int max_iter, std::string direction, int verbose,
                             py::object seed) {
    auto rdp = data_points.unchecked<2>();
    auto rps = point_size.unchecked<1>();
    auto rb = boxes.unchecked<2>();
    auto rxl = xlim.unchecked<1>();
    auto ryl = ylim.unchecked<1>();
    auto rhj = hjust.unchecked<1>();
    auto rvj = vjust.unchecked<1>();

    int n_points = static_cast<int>(rdp.shape(0));
    int n_texts = static_cast<int>(rb.shape(0));

    double force_point_size = 100.0;

    if (std::isnan(force_push)) force_push = 1e-6;
    if (std::isnan(force_pull)) force_pull = 1e-6;

    if (force_push == 0) max_iter = 0;

    if (n_texts > n_points) {
        throw std::invalid_argument("n_texts > n_points");
    }
    if (rps.shape(0) != n_points) {
        throw std::invalid_argument("point_size.length() != n_points");
    }
    if (rhj.shape(0) < n_texts) {
        throw std::invalid_argument("hjust.length() < n_texts");
    }
    if (rvj.shape(0) < n_texts) {
        throw std::invalid_argument("vjust.length() < n_texts");
    }
    if (rxl.shape(0) != 2) {
        throw std::invalid_argument("xlim.length() != 2");
    }
    if (ryl.shape(0) != 2) {
        throw std::invalid_argument("ylim.length() != 2");
    }

    Point xbounds = {rxl(0), rxl(1)};
    Point ybounds = {ryl(0), ryl(1)};

    std::vector<Point> Points(n_points);
    std::vector<Circle> DataCircles(n_points);
    for (int i = 0; i < n_points; i++) {
        DataCircles[i].x = rdp(i, 0);
        DataCircles[i].y = rdp(i, 1);
        DataCircles[i].r = rps(i) + (point_padding_x + point_padding_y) / 4.0;
        Points[i].x = rdp(i, 0);
        Points[i].y = rdp(i, 1);
    }

    // RNG: std::mt19937, seeded either from std::random_device or the supplied
    // int. Deviation from R's Marsaglia-Bray Gaussian is documented.
    std::mt19937 rng;
    if (seed.is_none()) {
        std::random_device rd;
        rng.seed(rd());
    } else {
        rng.seed(seed.cast<std::uint32_t>());
    }
    std::normal_distribution<double> jitter(0.0, force_push);
    std::vector<double> r(n_texts);
    for (int i = 0; i < n_texts; i++) {
        r[i] = jitter(rng);
    }

    std::vector<Box> TextBoxes(n_texts);
    std::vector<Point> original_centroids(n_texts);
    std::vector<double> TextBoxWidths(n_texts, 0);
    for (int i = 0; i < n_texts; i++) {
        TextBoxes[i].x1 = rb(i, 0);
        TextBoxes[i].x2 = rb(i, 2);
        TextBoxes[i].y1 = rb(i, 1);
        TextBoxes[i].y2 = rb(i, 3);
        TextBoxWidths[i] = std::abs(TextBoxes[i].x2 - TextBoxes[i].x1);
        if (direction != "y") {
            TextBoxes[i].x1 += r[i];
            TextBoxes[i].x2 += r[i];
        }
        if (direction != "x") {
            TextBoxes[i].y1 += r[i];
            TextBoxes[i].y2 += r[i];
        }
        original_centroids[i] = centroid(TextBoxes[i], rhj(i), rvj(i));
    }
    TextBoxWidths = rescale(TextBoxWidths);

    std::vector<Point> velocities(n_texts, {0, 0});
    double velocity_decay = 0.7;

    Point f{}, ci{}, cj{};

    auto start_time = std::chrono::steady_clock::now();
    long long elapsed_ns = 0;
    double max_time_ns = max_time * 1e9;

    std::vector<double> total_overlaps(n_texts, 0);
    std::vector<bool> too_many_overlaps(n_texts, false);

    int iter = 0;
    int n_overlaps = 1;
    int p_overlaps = 1;
    bool i_overlaps = true;

    while (n_overlaps && iter < max_iter) {
        iter += 1;
        p_overlaps = n_overlaps;
        n_overlaps = 0;

        if (iter % 10 == 0) {
            elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - start_time)
                             .count();
            if (static_cast<double>(elapsed_ns) > max_time_ns) break;
        }

        force_push *= 0.99999;
        force_pull *= 0.9999;

        for (int i = 0; i < n_texts; i++) {
            if (iter == 2 && total_overlaps[i] > max_overlaps) {
                too_many_overlaps[i] = true;
            }
            if (too_many_overlaps[i]) continue;
            total_overlaps[i] = 0;
            i_overlaps = false;
            f.x = 0;
            f.y = 0;
            ci = centroid(TextBoxes[i], rhj(i), rvj(i));

            for (int j = 0; j < n_points; j++) {
                if (i == j) {
                    if (rps(i) == 0 && point_padding_x == 0 &&
                        point_padding_y == 0) {
                        continue;
                    }
                    if (overlaps(DataCircles[i], TextBoxes[i])) {
                        n_overlaps += 1;
                        i_overlaps = true;
                        total_overlaps[i] += 1;
                        f = f + repel_force(ci, Points[i],
                                            rps(i) * force_point_size * force_push,
                                            direction);
                    }
                } else if (j < n_texts && too_many_overlaps[j]) {
                    if (rps(j) == 0 && point_padding_x == 0 &&
                        point_padding_y == 0) {
                        continue;
                    }
                    if (overlaps(DataCircles[j], TextBoxes[i])) {
                        n_overlaps += 1;
                        i_overlaps = true;
                        total_overlaps[i] += 1;
                        f = f + repel_force(ci, Points[j],
                                            rps(i) * force_point_size * force_push,
                                            direction);
                    }
                } else {
                    if (j < n_texts) {
                        cj = centroid(TextBoxes[j], rhj(j), rvj(j));
                        if (overlaps(TextBoxes[i], TextBoxes[j])) {
                            n_overlaps += 1;
                            i_overlaps = true;
                            total_overlaps[i] += 1;
                            f = f + repel_force(ci, cj, force_push, direction);
                        }
                    }
                    if (rps(j) == 0 && point_padding_x == 0 &&
                        point_padding_y == 0) {
                        continue;
                    }
                    if (overlaps(DataCircles[j], TextBoxes[i])) {
                        n_overlaps += 1;
                        i_overlaps = true;
                        total_overlaps[i] += 1;
                        f = f + repel_force(ci, Points[j],
                                            rps(i) * force_point_size * force_push,
                                            direction);
                    }
                }
            }

            if (!i_overlaps) {
                f = f + spring_force(original_centroids[i], ci, force_pull,
                                     direction);
            }

            double overlap_multiplier = 1.0;
            if (total_overlaps[i] > 10) {
                overlap_multiplier += 0.5;
            } else {
                overlap_multiplier += 0.05 * total_overlaps[i];
            }

            velocities[i] = overlap_multiplier * velocities[i] *
                                (TextBoxWidths[i] + 1e-6) * velocity_decay +
                            f;
            TextBoxes[i] = TextBoxes[i] + velocities[i];
            TextBoxes[i] = put_within_bounds(TextBoxes[i], xbounds, ybounds);

            if (n_overlaps == 0 || iter % 5 == 0) {
                for (int j = 0; j < n_texts; j++) {
                    cj = centroid(TextBoxes[j], rhj(j), rvj(j));
                    ci = centroid(TextBoxes[i], rhj(i), rvj(i));
                    if (i != j && line_intersect(ci, Points[i], cj, Points[j])) {
                        n_overlaps += 1;
                        TextBoxes[i] = TextBoxes[i] + spring_force(cj, ci, 1, direction);
                        TextBoxes[j] = TextBoxes[j] + spring_force(ci, cj, 1, direction);
                        ci = centroid(TextBoxes[i], rhj(i), rvj(i));
                        cj = centroid(TextBoxes[j], rhj(j), rvj(j));
                        if (line_intersect(ci, Points[i], cj, Points[j])) {
                            TextBoxes[i] = TextBoxes[i] + spring_force(cj, ci, 1.25, direction);
                            TextBoxes[j] = TextBoxes[j] + spring_force(ci, cj, 1.25, direction);
                        }
                    }
                }
            }
        }
    }

    if (verbose) {
        std::string msg;
        if (static_cast<double>(elapsed_ns) > max_time_ns) {
            msg = "ggrepel: " + std::to_string(max_time_ns / 1e9) +
                  "s elapsed for " + std::to_string(iter) + " iterations, " +
                  std::to_string(p_overlaps) +
                  " overlaps. Consider increasing 'max_time'.";
        } else if (iter >= max_iter) {
            msg = "ggrepel: " + std::to_string(max_iter) + " iterations in " +
                  std::to_string(elapsed_ns / 1e9) + "s, " +
                  std::to_string(p_overlaps) +
                  " overlaps. Consider increasing 'max_iter'.";
        } else {
            msg = "ggrepel: text repel complete in " + std::to_string(iter) +
                  " iterations (" + std::to_string(elapsed_ns / 1e9) + "s), " +
                  std::to_string(p_overlaps) + " overlaps";
        }
        // Mirror R's ``rlang::inform`` (stderr-only diagnostic, not a
        // warning): route through ggrepel_py's named logger instead of
        // ``warnings.warn`` to avoid being intercepted by ``-W error``
        // or ``pytest.warns`` filters.
        py::module_::import("ggrepel_py._options").attr("inform")(msg);
    }

    py::array_t<double> xs(n_texts);
    py::array_t<double> ys(n_texts);
    py::array_t<bool> too_many(n_texts);
    auto rxs = xs.mutable_unchecked<1>();
    auto rys = ys.mutable_unchecked<1>();
    auto rtm = too_many.mutable_unchecked<1>();
    for (int i = 0; i < n_texts; i++) {
        rxs(i) = (TextBoxes[i].x1 + TextBoxes[i].x2) / 2;
        rys(i) = (TextBoxes[i].y1 + TextBoxes[i].y2) / 2;
        rtm(i) = too_many_overlaps[i];
    }

    py::dict out;
    out["x"] = xs;
    out["y"] = ys;
    out["too_many_overlaps"] = too_many;
    return out;
}

PYBIND11_MODULE(_repel, m) {
    m.doc() = "Force-directed text repulsion kernel for ggrepel_py.";

    m.def("euclid", &euclid_vec, "a"_a, "b"_a,
          "Euclidean distance between two 2D points.");
    m.def("centroid", &centroid_vec, "b"_a, "hjust"_a, "vjust"_a,
          "Centroid of a box with hjust/vjust weights.");
    m.def("intersect_circle_rectangle", &intersect_circle_rectangle_vec, "c"_a,
          "r"_a);
    m.def("intersect_line_circle", &intersect_line_circle_vec, "p1"_a, "p2"_a,
          "r"_a);
    m.def("intersect_line_rectangle", &intersect_line_rectangle_vec, "p1"_a,
          "p2"_a, "b"_a);
    m.def("select_line_connection", &select_line_connection_vec, "p1"_a,
          "b"_a);
    m.def("approximately_equal", &approximately_equal_fn, "x1"_a, "x2"_a);

    m.def("repel_boxes2", &repel_boxes2,
          "data_points"_a, "point_size"_a, "point_padding_x"_a,
          "point_padding_y"_a, "boxes"_a, "xlim"_a, "ylim"_a, "hjust"_a,
          "vjust"_a, "force_push"_a = 1e-7, "force_pull"_a = 1e-7,
          "max_time"_a = 0.1, "max_overlaps"_a = 10.0, "max_iter"_a = 2000,
          "direction"_a = std::string("both"), "verbose"_a = 0,
          "seed"_a = py::none(),
          "Run the force-directed repulsion loop.");
}
