use std::ops::{Index, IndexMut};
use crate::linear_algebra::float_vec::FloatN;

//#region FloatMat

#[derive(Debug, Copy, Clone)]
pub struct FloatMat<const N: usize, const M: usize> {
    //Row major
    data: [FloatN<M>; N],
}

//#region std traits

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

//#endregion std traits

impl<const N: usize, const M: usize> FloatMat<N,M> {

    //#region ctor

    pub fn new(data: [FloatN<M>; N]) -> Self {
        Self { data }
    }

    pub fn zero() -> Self {
        Self::new([FloatN::<M>::zero(); N])
    }

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

    //#endregion ctor

    //#region getters

    pub fn rows_count(&self) -> usize {
        N
    }

    pub fn cols_count(&self) -> usize {
        M
    }

    pub fn get_column(&self, col_idx: usize) -> FloatN<N> {
        let mut column = FloatN::<N>::zero();
        for row_idx in 0..N {
            column[row_idx] = self[row_idx][col_idx];
        }
        column
    }

    pub fn get_sub_mat<const O: usize,const P: usize>(&self, i: usize, j: usize)
                                                      -> Option<FloatMat<O,P>> {
        if O + i > N || P + j > M {
            return None;
        }

        let mut sub_mat = FloatMat::<O, P>::zero();
        for k in 0..O {
            for l in 0..P {
                sub_mat[k][l] = self[i + k][j + l];
            }
        }
        Some(sub_mat)
    }

    pub fn get_sub_mat_on<const O: usize, const P: usize>(&self, i: usize, j: usize, mut k: usize,
                                                          mut l: usize)
                                                          -> Option<FloatMat<O,P>> {
        // first ensure i and j are within a proper range
        if i > N || j > M {
            return None;
        }

        // then we might want to clamp k and l
        let delta_k = (N) as i32 - (i + k) as i32;
        let delta_l = (M) as i32 - (j + l) as i32 ;
        if delta_k < 0 {
            k -= delta_k.abs() as usize;
        }
        if delta_l < 0 {
            l -= delta_l.abs() as usize;
        }

        //Then we want to ensure O,P is big enought to fit data in
        if i + k > O || j + l > P {
            return None;
        }

        let mut sub_mat_on = FloatMat::<O, P>::zero();
        for a in i..k {
            for b in j..l {
                sub_mat_on[a][b] = self[a][b];
            }
        }
        Some(sub_mat_on)
    }

    //#endregion getters

    //#region data manipulation

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

    fn swap_entries(&mut self, row1: usize, col1: usize, row2: usize, col2: usize) {
        let temp = self[row1][col1];
        self[row1][col1] = self[row2][col2];
        self[row2][col2] = temp;
    }

    fn set_row(&mut self, index: usize, new_row: FloatN<M>) {
        self.data[index] = new_row;
    }

    pub fn set_column(&mut self, col_idx: usize, column: FloatN<N>) {
        for row_idx in 0..N {
            self[row_idx][col_idx] = column[row_idx];
        }
    }

    //#endregion data manipulation

    //#region checks

    pub fn is_upper_triangular(&self) -> bool {
        for i in 0..N {
            for j in 0..i {
                if self[i][j] != 0.0 {
                    return false;
                }
            }
        }
        true
    }

    pub fn is_lower_triangular(&self) -> bool {
        for i in 0..N {
            for j in (i + 1)..M {
                if self[i][j] != 0.0 {
                    return false;
                }
            }
        }
        true
    }

    pub fn is_orthogonal(&self) -> bool {
        let transpose = self.transpose();
        let product = self.mul(&transpose);
        product.is_identity()
    }

    pub fn is_equal(&self, other: &Self) -> bool {
        for i in 0..N {
            for j in 0..M {
                if (self[i][j] - other[i][j]).abs() > f64::EPSILON * 10.0 {
                    return false;
                }
            }
        }
        true
    }

    //#endregion checks

    //#region matrix operation

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
                //deal with imprecision for 0.0
                if result[i][j].abs() < f64::EPSILON * 10.0 {
                    result[i][j] = 0.0;
                }
            }
        }
        result
    }

    pub fn sub(&self, other: &Self) -> Self {
        self.add(&other.scale(-1.0))
    }

    pub fn mul<const P: usize>(&self, other: &FloatMat<M, P>) -> FloatMat<N,P> {
        let mut result = FloatMat::<N,P>::zero();

        for i in 0..N {
            for j in 0..P {
                let mut sum = 0.0;
                for k in 0..M {
                    sum += self[i][k] * other[k][j];
                }
                result[i][j] = sum;
                //deal with imprecision for 0.0
                if result[i][j].abs() < f64::EPSILON * 10.0 {
                    result[i][j] = 0.0;
                }
            }
        }

        result
    }

    //#endregion matrix operation

    //#region matrix algorithm

    pub fn row_echelon_form(&self) -> Self {
        let mut rref = self.clone();
        let mut pivot = 0;

        for i in 0..N {
            // Find pivot row
            let mut max_row = i;
            for j in i+1..N {
                if rref[j][pivot].abs() > rref[max_row][pivot].abs() {
                    max_row = j;
                }
            }

            // Swap pivot row with current row
            if max_row != i {
                rref.swap_rows(i, max_row);
            }

            // Scale pivot row to have a leading coefficient of 1
            let pivot_coeff = rref[i][pivot];
            if pivot_coeff != 0.0 {
                for k in 0..M {
                    rref[i][k] /= pivot_coeff;
                }
            }

            // Make all rows below pivot row zero in current column
            for j in i+1..N {
                let factor = rref[j][pivot];
                rref.add_scaled_row(j, i, -factor);
            }

            pivot += 1;
            if pivot >= M {
                break;
            }
        }
        rref
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

    pub fn rank(&self) -> i32 {
        let ref rref = self.row_echelon_form();
        let mut rank = 0;
        for i in 0..N {
            let mut is_zero_row = true;
            for j in 0..M {
                if rref[i][j] != 0.0 {
                    is_zero_row = false;
                    break;
                }
            }
            if !is_zero_row {
                rank += 1;
            }
        }

        rank
    }

    pub fn qr_decomposition_gs(&self) -> (FloatMat<N, M>, FloatMat<M, M>) {
        let mut q = FloatMat::<N, M>::zero();
        let mut r = FloatMat::<M, M>::zero();

        for j in 0..M {
            let mut v = self.get_column(j);
            for i in 0..j {
                let q_i = q.get_column(i);
                let r_ij = v.dot(&q_i);
                r[i][j] = r_ij;
                v = v.sub(&q_i.scale(r_ij));
            }
            r[j][j] = v.magnitude();
            q.set_column(j, v.normalize());
        }

        (q, r)
    }

    pub fn lq_decomposition_gs(&self) -> (FloatMat<N,N>, FloatMat<N,M>) {
        let (q_transpose, r_transpose) =
            self.transpose().qr_decomposition_gs();
        let q = q_transpose.transpose();
        let r = r_transpose.transpose();
        (r, q)
    }

    pub fn qr_decomposition_householder(&self) -> (FloatMat<N, M>, FloatMat<M, M>) {
        let mut q = FloatMat::<N, M>::identity();
        let mut r: FloatMat<M,M> = self.get_sub_mat_on::<M,M>(0,0, M, M).unwrap();
        for j in 0..M.min(N-1) {
            let mut v = r.get_column(j);
            for k in 0..j {
                v[k] = 0.0;
            }
            let norm = v.magnitude();
            let sign = if v[0] > 0.0 { 1.0 } else { -1.0 };
            v[j] = v[j] + sign * norm;
            let dot = v.dot(&v);
            let h = FloatMat::<M, M>::identity().sub(&v.kronecker(&v.scale(2.0 / dot)));
            r = h.mul(&r);
            q = q.mul(&h);
        }

        (q, r)
    }

    //#endregion matrix algorithm
}

impl<const N: usize> FloatMat<N,N> {

    //#region ctor

    //#endregion ctor

    //#region checks

    pub fn is_symmetric(&self) -> bool {
        for i in 0..N {
            for j in 0..N {
                if self[i][j] != self[j][i] {
                    return false;
                }
            }
        }
        true
    }

    pub fn is_identity(&self) -> bool {
        for i in 0..N {
            for j in 0..N {
                if i == j {
                    if (self[i][j] - 1.0).abs() > f64::EPSILON {
                        return false;
                    }
                } else {
                    if self[i][j].abs() > f64::EPSILON {
                        return false;
                    }
                }
            }
        }

        true
    }

    //#endregion checks

    //#region matrix algorithms

    pub fn triangulate_gaussian(&self) -> Self {
        let mut rref = self.clone();
        //Gaussian elimination with maximum selection in column
        for i in 0..N {
            // Find the row with maximum absolute value in the i-th column
            let mut j_max = i;
            for j in (i + 1)..N {
                if rref[j][i].abs() > rref[j_max][i].abs() {
                    j_max = j;
                }
            }

            // Swap rows i and j_max if necessary
            if j_max != i {
                rref.swap_rows(i, j_max);
            }

            // Scale row i to make the i-th column elements below the i-th row zero
            if rref[i][i] == 0.0 {
                continue;
            }

            // Subtract multiples of row i from the rows below it to make the i-th column elements
            // below the i-th row zero
            for j in (i + 1)..N {
                let factor = rref[j][i] / rref[i][i];
                rref.add_scaled_row(j, i, -factor);
            }
        }

        rref
    }

    pub fn lu_decomposition_gaussian_elimination(&self) -> Option<(FloatMat<N, N>, FloatMat<N, N>)> {
        let mut a = self.clone();
        let mut l = FloatMat::identity();
        let mut u = FloatMat::zero();

        for i in 0..N {
            // Perform row swaps to avoid zero pivots
            for j in i..N {
                if a[j][i] != 0.0 {
                    a.swap_rows(i, j);
                    l.swap_rows(i, j);
                    if i > 0 {
                        for k in 0..i-1 {
                            l.swap_entries(k, i, k, j);
                        }
                    }
                    break;
                }
                if j == N - 1 {
                    // Matrix is singular
                    return None;
                }
            }

            // Perform Gaussian elimination on current column
            for j in i+1..N {
                let pivot = a[i][i];
                if pivot == 0.0 {
                    // Matrix is singular
                    return None;
                }
                let factor = a[j][i] / pivot;
                a.add_scaled_row(j, i, -factor);
                l[j][i] = factor;
            }
        }

        for i in 0..N {
            u.set_row(i, a[i]);
        }

        Some((l, u))
    }

    pub fn lu_decomposition_doolittle(&self) -> Option<(FloatMat<N,N>, FloatMat<N,N>)> {
        let mut l = FloatMat::identity();
        let mut u = FloatMat::zero();

        for j in 0..N {
            for i in j..N {
                let sum = (0..j).fold(0.0, |acc, k| acc + l[j][k] * u[k][i]);
                u[j][i] = self[j][i] - sum;

                if j == i && l[j][i] == 0.0 {
                    return None;
                }
            }

            for i in (j + 1)..N {
                let sum = (0..j).fold(0.0, |acc, k| acc + l[i][k] * u[k][j]);
                l[i][j] = (self[i][j] - sum) / u[j][j];
            }
        }

        Some((l, u))
    }

    pub fn lu_decomposition_crout(&self) -> Option<(FloatMat<N, N>, FloatMat<N, N>)> {
        let mut u = FloatMat::identity();
        let mut l = FloatMat::zero();

        for j in 0..N {
            for i in j..N {
                let sum = (0..j).fold(0.0, |acc, k| acc + l[i][k] * u[k][j]);
                l[i][j] = self[i][j] - sum;

                if i == j && l[i][j] == 0.0 {
                    return None;
                }
            }

            for i in j..N {
                let sum = (0..j).fold(0.0, |acc, k| acc + l[j][k] * u[k][i]);
                u[j][i] = (self[j][i] - sum) / l[j][j];
            }
        }

        Some((l, u))
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

    //#endregion matric algorithms
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
    fn float_mat_triangulate_gaussian_test() {
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
    fn float_mat_lu_decomposition_gaussian_elimination_test() {
        let row_a = FloatN::new([1.0,2.0,3.0]);
        let row_b = FloatN::new([3.0,7.0,1.0]);
        let row_c = FloatN::new([5.0,1.0,2.0]);
        let mat_a = FloatMat::new([row_a, row_b, row_c]);
        let opts =  mat_a.lu_decomposition_gaussian_elimination();
        match opts {
            None => assert!(false),
            Some((lower, upper)) => {
                assert_eq!(lower[0][0], 1.0);
                assert_eq!(lower[1][0], 3.0);
                assert_eq!(lower[1][1], 1.0);
                assert_eq!(lower[2][0], 5.0);
                assert_eq!(lower[2][1], -9.0);
                assert_eq!(lower[2][2], 1.0);
                assert_eq!(upper[0][0], 1.0);
                assert_eq!(upper[0][1], 2.0);
                assert_eq!(upper[0][2], 3.0);
                assert_eq!(upper[1][2], -8.0);
                assert_eq!(upper[1][1], 1.0);
                assert_eq!(upper[2][2], -85.0);
            }
        }
    }

    #[test]
    fn float_mat_lu_decomposition_doolittle_test() {
        let row_a = FloatN::new([1.0,2.0,3.0]);
        let row_b = FloatN::new([3.0,7.0,1.0]);
        let row_c = FloatN::new([5.0,1.0,2.0]);
        let mat_a = FloatMat::new([row_a, row_b, row_c]);
        let opts =  mat_a.lu_decomposition_doolittle();
        match opts {
            None => assert!(false),
            Some((lower, upper)) => {
                assert_eq!(lower[0][0], 1.0);
                assert_eq!(lower[1][0], 3.0);
                assert_eq!(lower[1][1], 1.0);
                assert_eq!(lower[2][0], 5.0);
                assert_eq!(lower[2][1], -9.0);
                assert_eq!(lower[2][2], 1.0);
                assert_eq!(upper[0][0], 1.0);
                assert_eq!(upper[0][1], 2.0);
                assert_eq!(upper[0][2], 3.0);
                assert_eq!(upper[1][2], -8.0);
                assert_eq!(upper[1][1], 1.0);
                assert_eq!(upper[2][2], -85.0);
            }
        }
    }

    #[test]
    fn float_mat_lu_decomposition_crout_test() {
        let row_a = FloatN::new([1.0,2.0,3.0]);
        let row_b = FloatN::new([3.0,7.0,1.0]);
        let row_c = FloatN::new([5.0,1.0,2.0]);
        let mat_a = FloatMat::new([row_a, row_b, row_c]);
        let opts =  mat_a.lu_decomposition_crout();
        match opts {
            None => assert!(false),
            Some((lower, upper)) => {
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
                assert_eq!(upper[1][1], 1.0);
                assert_eq!(upper[2][2], 1.0);
            }
        }
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

    #[test]
    fn float_mat_is_symmetric_test() {
        let row_a = FloatN::new([1.0, 2.0]);
        let row_b = FloatN::new([2.0, 1.0]);
        let mat_a = FloatMat::new([row_a, row_b]);
        let mat_b = FloatMat::new([row_a, row_a]);
        assert_eq!(mat_a.is_symmetric(), true);
        assert_eq!(mat_b.is_symmetric(), false);
    }

    #[test]
    fn float_mat_rank_test() {
        let row_a = FloatN::new([1.0,2.0,3.0,4.0]);
        let row_b = FloatN::new([3.0,7.0,1.0,2.0]);
        let row_c = FloatN::new([5.0,1.0,2.0,1.0]);
        let mat_a = FloatMat::new([row_a, row_b, row_c, row_a]);
        let rank = mat_a.rank();
        assert_eq!(rank, 3);
    }

    #[test]
    fn float_mat_is_upper_lower_triangular_test() {
        let row_a = FloatN::new([1.0,2.0,3.0]);
        let row_b = FloatN::new([0.0,1.0,2.0]);
        let mat_a = FloatMat::new([row_a, row_b]);
        assert_eq!(mat_a.is_upper_triangular(), true);
        assert_eq!(mat_a.is_lower_triangular(), false);
        let row_c = FloatN::new([1.0, 0.0, 0.0]);
        let row_d = FloatN::new([2.0, 1.0, 0.0]);
        let mat_b = FloatMat::new([row_c, row_d]);
        assert_eq!(mat_b.is_upper_triangular(), false);
        assert_eq!(mat_b.is_lower_triangular(), true);
    }

    #[test]
    fn float_mat_is_identity_test() {
        let row_a = FloatN::new([1.0,0.0,0.0]);
        let row_b = FloatN::new([0.0,1.0,0.0]);
        let row_c = FloatN::new([0.0, 0.0, 1.0]);
        let mut mat_a = FloatMat::new([row_a, row_b, row_c]);
        assert_eq!(mat_a.is_identity(), true);
        mat_a.swap_entries(0,0,0,1);
        assert_eq!(mat_a.is_identity(), false);
    }

    #[test]
    fn float_mat_is_orthogonal_test() {
        let row_a = FloatN::new([1.0,0.0,0.0]);
        let row_b = FloatN::new([0.0,0.0,1.0]);
        let row_c = FloatN::new([0.0,-1.0,0.0]);
        let mat_a = FloatMat::new([row_a, row_b, row_c]);
        assert_eq!(mat_a.is_orthogonal(), true);
        let row_d = FloatN::new([1.0,2.0,3.0]);
        let row_e = FloatN::new([4.0,5.0,6.0]);
        let row_f = FloatN::new([7.0,8.0,9.0]);
        let mat_b = FloatMat::new([row_d, row_e, row_f]);
        assert_eq!(mat_b.is_orthogonal(), false);
    }

    #[test]
    fn float_mat_is_equal_test() {
        let mat_a = FloatMat::<2,2>::identity();
        let mat_b = FloatMat::<2,2>::identity();
        assert_eq!(mat_a.is_equal(&mat_b), true);
        let mut mat_c = FloatMat::<2,2>::identity();
        mat_c.swap_entries(0,0,0,1);
        assert_eq!(mat_a.is_equal(&mat_c), false);
    }

    #[test]
    fn float_mat_qr_decomposition_gs_test() {
        let row_a = FloatN::new([1.0,2.0,3.0,4.0]);
        let row_b = FloatN::new([3.0,7.0,1.0,2.0]);
        let row_c = FloatN::new([5.0,1.0,2.0,1.0]);
        let mat_a = FloatMat::new([row_a, row_b, row_c]);
        let (q, r) = mat_a.qr_decomposition_gs();
        assert_eq!(q.is_orthogonal(), true);
        assert_eq!(r.is_upper_triangular(), true);
        let mul = q.mul(&r);
        assert_eq!(mul.is_equal(&mat_a), true);
    }

    #[test]
    fn float_mat_lq_decomposition_gs_test() {
        let row_a = FloatN::new([1.0,2.0,3.0,4.0]);
        let row_b = FloatN::new([3.0,7.0,1.0,2.0]);
        let row_c = FloatN::new([5.0,1.0,2.0,1.0]);
        let mat_a = FloatMat::new([row_a, row_b, row_c]);
        let (l, q) = mat_a.lq_decomposition_gs();
        assert_eq!(q.is_orthogonal(), true);
        assert_eq!(l.is_lower_triangular(), true);
        let mul = l.mul(&q);
        assert_eq!(mul.is_equal(&mat_a), true);
    }

    #[test]
    fn float_mat_qr_decomposition_householder_test() {
        let row_a = FloatN::new([1.0,2.0,3.0,4.0]);
        let row_b = FloatN::new([3.0,7.0,1.0,2.0]);
        let row_c = FloatN::new([5.0,1.0,2.0,1.0]);
        let mat_a = FloatMat::new([row_a, row_b, row_c]);
        let (q, r) = mat_a.qr_decomposition_householder();
        assert_eq!(q.is_orthogonal(), true);
        assert_eq!(r.is_upper_triangular(), true);
        let mul = q.mul(&r);
        assert_eq!(mul.is_equal(&mat_a), true);
    }
}

//#endregion Tests