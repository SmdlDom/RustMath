use std::ops::{Index,IndexMut};

//#region FloatN

#[derive(Debug, Copy, Clone)]
pub struct FloatN<const N: usize> {
    data: [f64; N]
}

impl<const N: usize> Index<usize> for FloatN<N> {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(
            index < N,
            "Error: Out of Bound."
        );
        &self.data[index]
    }
}

impl<const N: usize> IndexMut<usize> for FloatN<N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(
            index < N,
            "Error: Out of Bound."
        );
        &mut self.data[index]
    }
}

impl<const N: usize> FloatN<N> {
    pub fn new(data: [f64; N]) -> Self {
        Self { data }
    }

    pub fn zero() -> Self {
        Self {data: [0.0; N] }
    }

    pub fn len(&self) -> usize {
        N
    }

    pub fn magnitude(&self) -> f64 {
        let mut sum = 0.0;
        for i in 0..N {
            sum += self[i] * self[i];
        }
        sum.sqrt()
    }

    pub fn normalize(&self) -> Self {
        let mag = self.magnitude();
        if mag == 0.0 {
            return self.clone();
        }
        let mut normalize_data = FloatN::zero();
        for i in 0..N {
            normalize_data[i] = self[i] / mag;
        }
        normalize_data
    }

    pub fn add(&self, other: &Self) -> Self {
        let mut result = FloatN::zero();
        for i in 0..N {
            result[i] = self[i] + other[i];
            //deal with imprecision for 0.0
            if result[i].abs() < f64::EPSILON * 10.0 {
                result[i] = 0.0;
            }
        }
        result
    }

    pub fn scale(&self, factor: f64) -> Self {
        let mut result = FloatN::zero();
        for i in 0..N {
            result[i] = self[i] * factor;
        }
        result
    }

    pub fn sub(&self, other: &Self) -> Self {
        self.add(&other.scale(-1.0))
    }

    pub fn dot(&self, other: &Self) -> f64 {
        let mut res = 0.0;
        for i in 0..N {
            res += self[i] * other[i];
        }
        res
    }

    pub fn distance(&self, other: &Self) -> f64 {
        let mut sum = 0.0;
        for i in 0..N {
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
}

impl FloatN<3> {
    pub fn cross(&self, other: &Self) -> Self {
        let mut result = [0.0; 3];
        result[0] = self[1] * other[2] - self[2] * other[1];
        result[1] = self[2] * other[0] - self[0] * other[2];
        result[2] = self[0] * other[1] - self[1] * other[0];
        FloatN::new(result)
    }
}

//#endregion FloatN

//#region Tests

#[cfg(test)]
mod float_n_tests {
    use super::*;

    #[test]
    fn float_n_new_test() {
        let x = FloatN::new([1.0, 2.0]);
        assert_eq!(x[0], 1.0);
        assert_eq!(x[1], 2.0);
    }

    #[test]
    fn float_n_zero_test() {
        let x = FloatN::<3>::zero();
        for i in 0..3 {
            assert_eq!(x[i], 0.0);
        }
    }

    #[test]
    fn float_n_magnitude_test() {
        let x = FloatN::new([3.0,4.0]);
        let mag = x.magnitude();
        assert_eq!(mag, 5.0);
    }

    #[test]
    fn float_n_normalize_test() {
        let x = FloatN::new([3.0,4.0]);
        let normalize_x = x.normalize();
        assert_eq!(normalize_x[0], 3.0/5.0);
        assert_eq!(normalize_x[1], 4.0/5.0);
    }

    #[test]
    fn float_n_add_test() {
        let x = FloatN::new([1.0,2.0]);
        let y = FloatN::new([2.0, 3.0]);
        let z = x.add(&y);
        assert_eq!(z[0], 3.0);
        assert_eq!(z[1], 5.0);
    }

    #[test]
    fn float_n_scale_test() {
        let x = FloatN::new([1.0,2.0]);
        let y = x.scale(2.0);
        assert_eq!(y[0], 2.0);
        assert_eq!(y[1],4.0);
    }

    #[test]
    fn float_n_sub_test() {
        let x = FloatN::new([1.0, 2.0]);
        let y = FloatN::new([2.0,1.0]);
        let z = x.sub(&y);
        assert_eq!(z[0], -1.0);
        assert_eq!(z[1], 1.0);
    }

    #[test]
    fn float_n_dot_test() {
        let x = FloatN::new([1.0,2.0]);
        let y = x.scale(2.0);
        let dot = x.dot(&y);
        assert_eq!(dot, 10.0);
    }

    #[test]
    fn float_n_cross_test() {
        let x = FloatN::new([1.0, 2.0, 3.0]);
        let y = x.scale(2.0);
        let cross_a = x.cross(&y);
        assert_eq!(cross_a[0], 0.0);
        let z = y.add(&FloatN::new([3.0,2.0,1.0]));
        let cross_b = y.cross(&z);
        assert_eq!(cross_b[0], -8.0);
    }

    #[test]
    fn float_n_distance_test() {
        let x = FloatN::new([1.0,2.0]);
        let y = x.scale(3.0);
        let distance = x.distance(&y);
        assert_eq!(format!("{:.2}", distance), "4.47");
    }

    #[test]
    fn float_n_angle_between_test() {
        let x = FloatN::new([1.0,2.0]);
        let y = FloatN::new([2.0, 1.0]);
        let angle = x.angle_between(&y);
        assert_eq!(format!("{:.2}", angle), "0.64");
    }

    #[test]
    fn float_n_reflect_test() {
        let x = FloatN::new([1.0,2.0]);
        let y = FloatN:: new([1.0,1.0]);
        let reflection = x.reflect(&y);
        assert_eq!(format!("{:.2}", reflection[0]), "-5.00");
        assert_eq!(format!("{:.2}", reflection[1]), "-4.00");
    }

    #[test]
    fn float_n_project_reject_test() {
        let x = FloatN::new([1.0,2.0]);
        let y = FloatN::new([-1.0, -1.0]);
        let proj_x_on_y = x.project(&y);
        let rej_x_on_y = x.reject(&y);
        let proj_plus_rej = proj_x_on_y.add(&rej_x_on_y);
        assert_eq!(proj_plus_rej[0], 1.0);
        assert_eq!(proj_plus_rej[1], 2.0);
    }
}

//#endregion Tests