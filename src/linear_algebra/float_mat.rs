use std::ops::{Index, IndexMut};
use crate::linear_algebra::float_vec::FloatN;

//#region FloatMat

#[derive(Debug, Copy, Clone)]
pub struct FloatMat<const N: usize, const M: usize> {
    //Row major
    data: [FloatN<M>; N],
}

impl<const N: usize, const M: usize> Index<usize> for FloatMat<N,M> {
    type Output = FloatN<M>;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(
            index < N,
            "Error: Out of Bound."
        );
        &self.data[index]
    }
}

impl<const N: usize, const M: usize> IndexMut<usize> for FloatMat<N,M> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(
            index < N,
            "Error: Out of Bound."
        );
        &mut self.data[index]
    }
}

impl<const N: usize, const M: usize> FloatMat<N,M> {

    pub fn new(data: [FloatN<M>; N]) -> Self {
        Self { data }
    }

    pub fn zero() -> Self {
        Self::new([FloatN::<M>::zero(); N])
    }

    pub fn rows_count(&self) -> usize {
        N
    }

    pub fn cols_count(&self) -> usize {
        M
    }

    pub fn transpose(&self) -> FloatMat<M, N> {
        let mut data = [FloatN::<N>::zero(); M];
        for i in 0..N {
            for j in 0..M {
                data[j][i] = self[i][j];
            }
        }
        FloatMat::new(data)
    }

    pub fn scale(&self, factor: f64) -> Self {
        let mut result = Self::zero();
        for i in 0..N {
            for j in 0..M {
                result[i][j] = self[i][j] * factor;
            }
        }
        result
    }

    pub fn add(&self, other: &Self) -> Self {
        let mut result = Self::zero();
        for i in 0..N {
            for j in 0..M {
                result[i][j] = self[i][j] + other[i][j];
            }
        }
        result
    }

    pub fn mul<const P: usize>(&self, other: &FloatMat<P, M>) -> FloatMat<N,P> {
        let mut result = FloatMat::<N,P>::zero();

        for i in 0..N {
            for j in 0..P {
                let mut sum = 0.0;
                for k in 0..M {
                    sum += self[i][k] * other[k][j];
                }
                result[i][j] = sum;
            }
        }

        result
    }

    //#region Row manipulation

    fn swap_rows(&mut self, i: usize, j: usize) {
        let row_i = self[i];
        self[i] = std::mem::replace(&mut self[j], row_i);
    }

    fn scale_row(&mut self, i: usize, factor: f64) {
        self[i] = self[i].scale(factor);
    }

    fn add_scaled_row(&mut self, i: usize, j: usize, factor: f64) {
        self[i] = self[i].add(&self[j].scale(factor));
    }

    //#endregion Row manipulation

    //#region algo on matrix itself

    pub fn row_echelon_form(&self) -> Self {
        let mut clone = self.clone();
        let mut pivot = 0;

        for i in 0..N {
            // Find pivot row
            let mut max_row = i;
            for j in i+1..N {
                if clone[j][pivot].abs() > clone[max_row][pivot].abs() {
                    max_row = j;
                }
            }

            // Swap pivot row with current row
            if max_row != i {
                clone.swap_rows(i, max_row);
            }

            // Scale pivot row to have a leading coefficient of 1
            let pivot_coeff = clone[i][pivot];
            if pivot_coeff != 0.0 {
                for k in 0..M {
                    clone[i][k] /= pivot_coeff;
                }
            }

            // Make all rows below pivot row zero in current column
            for j in i+1..N {
                let factor = clone[j][pivot];
                clone.add_scaled_row(j, i, -factor);
            }

            pivot += 1;
            if pivot >= M {
                break;
            }
        }
        clone
    }

    pub fn reduced_row_echelon_form(&self) -> Self {
        let mut rref = self.row_echelon_form();
        let mut pivot = 0;

        for i in 0..N {
            // Find pivot row
            let mut pivot_row = None;
            for j in pivot..M {
                if rref[i][j] != 0.0 {
                    pivot_row = Some(j);
                    break;
                }
            }

            // If there is a pivot in this row
            if let Some(pivot_col) = pivot_row {
                // Make all other rows zero in current column
                for j in 0..N {
                    if i != j && rref[j][pivot_col] != 0.0 {
                        let factor = rref[j][pivot_col];
                        rref.add_scaled_row(j, i, -factor);
                    }
                }

                pivot += 1;
                if pivot >= M {
                    break;
                }
            }
        }
        rref
    }

    //#endregion algo on matrix itself
}

impl<const N: usize> FloatMat<N,N> {
    pub fn identity() -> Self {
        let mut data = Self::zero();

        for i in 0..N {
            for j in 0..N {
                if i == j {
                    data[i][j] = 1.0;
                }
            }
        }

       data
    }

    pub fn triangulate_gaussian(&self) -> Self {
        let mut clone = self.clone();
        //Gaussian elimination with maximum selection in column
        for i in 0..N {
            // Find the row with maximum absolute value in the i-th column
            let mut j_max = i;
            for j in (i + 1)..N {
                if clone[j][i].abs() > clone[j_max][i].abs() {
                    j_max = j;
                }
            }

            // Swap rows i and j_max if necessary
            if j_max != i {
                clone.swap_rows(i, j_max);
            }

            // Scale row i to make the i-th column elements below the i-th row zero
            if clone[i][i] == 0.0 {
                continue;
            }

            // Subtract multiples of row i from the rows below it to make the i-th column elements below the i-th row zero
            for j in (i + 1)..N {
                let factor = clone[j][i] / clone[i][i];
                clone.add_scaled_row(j, i, -factor);
            }
        }

        clone
    }

    pub fn lu_decomposition_crout(&self) -> (FloatMat<N, N>, FloatMat<N, N>) {
        let mut upper = FloatMat::identity();
        let mut lower = FloatMat::zero();

        for j in 0..N {
            for i in j..N {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += lower[i][k] * upper[k][j];
                }
                lower[i][j] = self[i][j] - sum;
            }

            for i in j..N {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += lower[j][k] * upper[k][i];
                }
                //TODO should handle cases where the det is 0
                upper[j][i] = (self[j][i] - sum) / lower[j][j];
            }
        }

        (lower, upper)
    }

    pub fn det(&self) -> f64 {
        let mut det = if N % 2 == 0 {1.0} else {-1.0};

        // Triangulize the matrix
        let mat_triangulated = self.triangulate_gaussian();

        // Compute the determinant as the product of the diagonal elements
        for i in 0..N {
            det *= mat_triangulated[i][i];
        }

        det
    }
}

//#endregion FloatMat

//#region Tests

#[cfg(test)]
mod float_mat_tests {
    use super::*;

    #[test]
    fn float_mat_new_test() {
        let row_a = FloatN::new([1.0,2.0,3.0]);
        let row_b = row_a.scale(2.0);
        let row_c = row_a.scale(3.0);
        let mat = FloatMat::new([row_a, row_b, row_c]);
        assert_eq!(mat[0][0], 1.0);
        assert_eq!(mat[1][1], 4.0);
        assert_eq!(mat[2][2], 9.0);
    }

    #[test]
    fn float_mat_identity_test() {
        let identity = FloatMat::<3,3>::identity();
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert_eq!(identity[i][j], 1.0);
                } else {
                    assert_eq!(identity[i][j], 0.0);
                }
            }
        }
    }

    #[test]
    fn float_mat_transpose_test() {
        let row_a = FloatN::new([1.0,2.0,3.0]);
        let row_b = row_a.scale(2.0);
        let mat = FloatMat::new([row_a, row_b]);
        let mat_prime = mat.transpose();
        assert_eq!(mat_prime[0][0], 1.0);
        assert_eq!(mat_prime[0][1], 2.0);
        assert_eq!(mat_prime[2][0], 3.0);
    }

    #[test]
    fn float_mat_scale_test() {
        let row_a = FloatN::new([1.0,2.0]);
        let row_b = row_a.scale(2.0);
        let mat = FloatMat::new([row_a, row_b]);
        let mat_scaled = mat.scale(2.0);
        assert_eq!(mat_scaled[0][0], 2.0);
        assert_eq!(mat_scaled[1][1], 8.0);
    }

    #[test]
    fn float_mat_add_test() {
        let row_a = FloatN::new([1.0,2.0]);
        let row_b = FloatN::new([2.0,3.0]);
        let mat_a = FloatMat::new([row_a, row_b]);
        let mat_b = FloatMat::new([row_b, row_a]);
        let mat_add = mat_a.add(&mat_b);
        assert_eq!(mat_add[0][0], 3.0);
        assert_eq!(mat_add[1][1], 5.0);
    }

    #[test]
    fn float_mat_mul_test() {
        let row_a = FloatN::new([1.0,2.0]);
        let row_b = row_a.scale(2.0);
        let mat_a = FloatMat::new([row_a, row_b]);
        let mat_b = mat_a.scale(2.0);
        let mat_mul = mat_a.mul(&mat_b);
        assert_eq!(mat_mul[0][0], 10.0);
        assert_eq!(mat_mul[1][1], 40.0);
    }

    #[test]
    fn float_mat_swap_rows_test() {
        let row_a = FloatN::new([1.0,2.0]);
        let row_b = row_a.scale(2.0);
        let mut mat_a = FloatMat::new([row_a, row_b]);
        mat_a.swap_rows(0,1);
        assert_eq!(mat_a[0][0], 2.0);
        assert_eq!(mat_a[1][1], 2.0);
    }

    #[test]
    fn float_mat_scale_row_test() {
        let row_a = FloatN::new([1.0,2.0]);
        let row_b = row_a.scale(2.0);
        let mut mat_a = FloatMat::new([row_a, row_b]);
        mat_a.scale_row(0,2.0);
        assert_eq!(mat_a[0][0], 2.0);
    }

    #[test]
    fn float_mat_add_scale_row_test() {
        let row_a = FloatN::new([1.0,2.0]);
        let row_b = row_a.scale(2.0);
        let mut mat_a = FloatMat::new([row_a, row_b]);
        mat_a.add_scaled_row(0,1,2.0);
        assert_eq!(mat_a[0][0], 5.0);
    }

    #[test]
    fn float_mat_triangulate_by_gaussian_test() {
        let row_a = FloatN::new([1.0,2.0,3.0]);
        let row_b = FloatN::new([3.0,7.0,1.0]);
        let row_c = FloatN::new([5.0,1.0,2.0]);
        let mat_a = FloatMat::new([row_a, row_b, row_c]);
        let triangulated =  mat_a.triangulate_gaussian();
        assert_eq!(triangulated[0][0], 5.0);
        assert_eq!(triangulated[1][1], 6.4);
        assert_eq!(triangulated[2][2], 2.65625);
    }

    #[test]
    fn float_mat_lu_decomposition_by_crout_test() {
        let row_a = FloatN::new([1.0,2.0,3.0]);
        let row_b = FloatN::new([3.0,7.0,1.0]);
        let row_c = FloatN::new([5.0,1.0,2.0]);
        let mat_a = FloatMat::new([row_a, row_b, row_c]);
        let (lower, upper) =  mat_a.lu_decomposition_crout();
        assert_eq!(lower[0][0], 1.0);
        assert_eq!(lower[1][0], 3.0);
        assert_eq!(lower[1][1], 1.0);
        assert_eq!(lower[2][0], 5.0);
        assert_eq!(lower[2][1], -9.0);
        assert_eq!(lower[2][2], -85.0);
        assert_eq!(upper[0][0], 1.0);
        assert_eq!(upper[0][1], 2.0);
        assert_eq!(upper[0][2], 3.0);
        assert_eq!(upper[1][2], -8.0);
    }

    #[test]
    fn float_mat_det_test() {
        let row_a = FloatN::new([1.0,2.0,3.0]);
        let row_b = FloatN::new([3.0,7.0,1.0]);
        let row_c = FloatN::new([5.0,1.0,2.0]);
        let mat_a = FloatMat::new([row_a, row_b, row_c]);
        let det = mat_a.det();
        assert_eq!(det, -85.0);
    }

    #[test]
    fn float_mat_row_echelon_form_test() {
        let row_a = FloatN::new([1.0,2.0,3.0,4.0]);
        let row_b = FloatN::new([3.0,7.0,1.0,2.0]);
        let row_c = FloatN::new([5.0,1.0,2.0,1.0]);
        let mat_a = FloatMat::new([row_a, row_b, row_c]);
        let echelon = mat_a.reduced_row_echelon_form();
        assert_eq!(format!("{:.2}", echelon[0][3]), "-0.36");
        assert_eq!(format!("{:.2}", echelon[1][3]), "0.26");
        assert_eq!(format!("{:.2}", echelon[2][3]), "1.28");
    }
}

//#endregion Tests