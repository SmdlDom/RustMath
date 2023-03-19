use std::ops::{Index,IndexMut};
use crate::linear_algebra::float_mat::FloatMat;

//#region FloatN

#[derive(Debug, Clone)]
pub struct FloatN {
    size: usize,
    data: Vec<f64>
}

impl Index<usize> for FloatN {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(
            index < self.size,
            "Error: Out of Bound."
        );
        &self.data[index]
    }
}

impl IndexMut<usize> for FloatN {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(
            index < self.size,
            "Error: Out of Bound."
        );
        &mut self.data[index]
    }
}

impl FloatN {
    pub fn new(data: Vec<f64>) -> Self {
        let size = data.len();
        Self { data, size }
    }

    pub fn zero(size: usize) -> Self {
        Self {data: vec!(0.0; size), size }
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn magnitude(&self) -> f64 {
        let mut sum = 0.0;
        for i in 0..self.size {
            sum += self[i] * self[i];
        }
        sum.sqrt()
    }

    pub fn normalize(&self) -> Self {
        let mag = self.magnitude();
        if mag == 0.0 {
            return self.clone();
        }
        let mut normalize_data = FloatN::zero(self.size);
        for i in 0..self.size {
            normalize_data[i] = self[i] / mag;
        }
        normalize_data
    }

    pub fn add(&self, other: &Self) -> Self {
        let mut result = FloatN::zero(self.size);
        for i in 0..self.size {
            result[i] = self[i] + other[i];
            //deal with imprecision for 0.0
            if result[i].abs() < f64::EPSILON * 10.0 {
                result[i] = 0.0;
            }
        }
        result
    }

    pub fn scale(&self, factor: f64) -> Self {
        let mut result = FloatN::zero(self.size);
        for i in 0..self.size {
            result[i] = self[i] * factor;
        }
        result
    }

    pub fn add_scalar(&self, scalar: f64) -> Self {
        let mut result = self.clone();
        for i in 0..self.size {
            result[i] += scalar;
        }
        result
    }

    pub fn sub(&self, other: &Self) -> Self {
        self.add(&other.scale(-1.0))
    }

    pub fn dot(&self, other: &Self) -> f64 {
        let mut res = 0.0;
        for i in 0..self.size {
            res += self[i] * other[i];
        }
        res
    }

    pub fn kronecker(&self, other: &Self) -> FloatMat {
        let mut result = FloatMat::zero(self.size, self.size);
        for i in 0..self.size {
            for j in 0..self.size {
                result[i][j] = self[i] * other[j];
            }
        }
        result
    }

    pub fn distance(&self, other: &Self) -> f64 {
        let mut sum = 0.0;
        for i in 0..self.size {
            sum += (self[i] - other[i]).powi(2);
        }
        sum.sqrt()
    }

    pub fn angle_between(&self, other: &Self) -> f64 {
        let dot_product = self.dot(other);
        let magnitudes_product = self.magnitude() * other.magnitude();
        if magnitudes_product == 0.0 {
            return 0.0;
        }
        (dot_product / magnitudes_product).acos()
    }

    pub fn reflect(&self, normal: &Self) -> Self {
        let dot = self.dot(normal);
        let normal_scaled = normal.scale(2.0 * dot);
        self.sub(&normal_scaled)
    }

    pub fn project(&self, normal: &Self) -> Self {
        let dot = self.dot(normal);
        let normal_normalized = normal.normalize();
        normal_normalized.scale(dot)
    }

    pub fn reject(&self, normal: &Self) -> Self {
        let projection = self.project(normal);
        self.sub(&projection)
    }

    pub fn cross(&self, other: &Self) -> Self {
        assert_eq!(
            self.size, 3,
            "Error: Cross product only defined for 3-dimensional vectors."
        );
        assert_eq!(
            other.size, 3,
            "Error: Cross product only defined for 3-dimensional vectors."
        );

        let mut result = FloatN::zero(3);
        result[0] = self[1] * other[2] - self[2] * other[1];
        result[1] = self[2] * other[0] - self[0] * other[2];
        result[2] = self[0] * other[1] - self[1] * other[0];
        result
    }
}

//#endregion FloatN

//#region Tests

#[cfg(test)]
mod float_n_tests {
    use super::*;

    #[test]
    fn float_n_new_test() {
        let x = FloatN::new(vec!(1.0, 2.0));
        assert_eq!(x[0], 1.0);
        assert_eq!(x[1], 2.0);
    }

    #[test]
    fn float_n_zero_test() {
        let x = FloatN::zero(3);
        for i in 0..3 {
            assert_eq!(x[i], 0.0);
        }
    }

    #[test]
    fn float_n_magnitude_test() {
        let x = FloatN::new(vec!(3.0,4.0));
        let mag = x.magnitude();
        assert_eq!(mag, 5.0);
    }

    #[test]
    fn float_n_normalize_test() {
        let x = FloatN::new(vec!(3.0,4.0));
        let normalize_x = x.normalize();
        assert_eq!(normalize_x[0], 3.0/5.0);
        assert_eq!(normalize_x[1], 4.0/5.0);
    }

    #[test]
    fn float_n_add_test() {
        let x = FloatN::new(vec!(1.0,2.0));
        let y = FloatN::new(vec!(2.0, 3.0));
        let z = x.add(&y);
        assert_eq!(z[0], 3.0);
        assert_eq!(z[1], 5.0);
    }

    #[test]
    fn float_n_scale_test() {
        let x = FloatN::new(vec!(1.0,2.0));
        let y = x.scale(2.0);
        assert_eq!(y[0], 2.0);
        assert_eq!(y[1],4.0);
    }

    #[test]
    fn float_n_sub_test() {
        let x = FloatN::new(vec!(1.0, 2.0));
        let y = FloatN::new(vec!(2.0,1.0));
        let z = x.sub(&y);
        assert_eq!(z[0], -1.0);
        assert_eq!(z[1], 1.0);
    }

    #[test]
    fn float_n_dot_test() {
        let x = FloatN::new(vec!(1.0,2.0));
        let y = x.scale(2.0);
        let dot = x.dot(&y);
        assert_eq!(dot, 10.0);
    }

    #[test]
    fn float_n_cross_test() {
        let x = FloatN::new(vec!(1.0, 2.0, 3.0));
        let y = x.scale(2.0);
        let cross_a = x.cross(&y);
        assert_eq!(cross_a[0], 0.0);
        let z = y.add(&FloatN::new(vec!(3.0,2.0,1.0)));
        let cross_b = y.cross(&z);
        assert_eq!(cross_b[0], -8.0);
    }

    #[test]
    fn float_n_distance_test() {
        let x = FloatN::new(vec!(1.0,2.0));
        let y = x.scale(3.0);
        let distance = x.distance(&y);
        assert_eq!(format!("{:.2}", distance), "4.47");
    }

    #[test]
    fn float_n_angle_between_test() {
        let x = FloatN::new(vec!(1.0,2.0));
        let y = FloatN::new(vec!(2.0, 1.0));
        let angle = x.angle_between(&y);
        assert_eq!(format!("{:.2}", angle), "0.64");
    }

    #[test]
    fn float_n_reflect_test() {
        let x = FloatN::new(vec!(1.0,2.0));
        let y = FloatN:: new(vec!(1.0,1.0));
        let reflection = x.reflect(&y);
        assert_eq!(format!("{:.2}", reflection[0]), "-5.00");
        assert_eq!(format!("{:.2}", reflection[1]), "-4.00");
    }

    #[test]
    fn float_n_project_reject_test() {
        let x = FloatN::new(vec!(1.0,2.0));
        let y = FloatN::new(vec!(-1.0, -1.0));
        let proj_x_on_y = x.project(&y);
        let rej_x_on_y = x.reject(&y);
        let proj_plus_rej = proj_x_on_y.add(&rej_x_on_y);
        assert_eq!(proj_plus_rej[0], 1.0);
        assert_eq!(proj_plus_rej[1], 2.0);
    }

    #[test]
    fn float_n_kronecker_product_test() {
        let x = FloatN::new(vec!(1.0,2.0,3.0));
        let y= FloatN::new(vec!(9.0,8.0,7.0));
        let outer_prod = x.kronecker(&y);
        assert_eq!(outer_prod[0][0], 9.0);
        assert_eq!(outer_prod[1][1], 16.0);
        assert_eq!(outer_prod[2][2], 21.0);
    }
}

//#endregion Tests