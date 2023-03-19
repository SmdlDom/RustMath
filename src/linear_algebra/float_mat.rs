use std::ops::{Index, IndexMut};
use crate::functions::polynomial::Polynomial;
use crate::linear_algebra::float_vec::FloatN;

//#region FloatMat

#[derive(Debug, Clone)]
pub struct FloatMat {
    //Row major
    rows: usize,
    cols: usize,
    data: Vec<FloatN>,
}

//#region std traits

impl Index<usize> for FloatMat {
    type Output = FloatN;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(
            index < self.rows,
            "Error: Out of Bound."
        );
        &self.data[index]
    }
}

impl IndexMut<usize> for FloatMat {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(
            index < self.rows,
            "Error: Out of Bound."
        );
        &mut self.data[index]
    }
}

//#endregion std traits

impl FloatMat {

    //#region ctor

    pub fn new(data: Vec<FloatN>) -> Self {
        let rows = data.len();
        let cols = data[0].len();
        for i in 0..rows {
            assert_eq!(data[i].len(), cols, "Error: Not all rows are of the same dimension")
        }
        Self { data, rows, cols }
    }

    pub fn zero(rows: usize, cols: usize) -> Self {
        Self::new(vec!(FloatN::zero(cols); rows))
    }

    pub fn identity(rows: usize, cols: usize) -> Self {
        let mut data = Self::zero(rows, cols);

        for i in 0..rows {
            for j in 0..rows {
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
        self.rows
    }

    pub fn cols_count(&self) -> usize {
        self.cols
    }

    pub fn get_column(&self, col_idx: usize) -> FloatN {
        let mut column = FloatN::zero(self.rows);
        for row_idx in 0..self.rows {
            column[row_idx] = self[row_idx][col_idx];
        }
        column
    }

    pub fn get_sub_mat(&self, i_start: usize, j_start: usize,
                       i_len: usize, j_len: usize) -> Option<FloatMat> {
        if i_start + i_len > self.rows || j_start + j_len > self.cols {
            return None;
        }

        let mut sub_mat = FloatMat::zero(i_len, j_len);
        for k in 0..i_len {
            for l in 0..j_len {
                sub_mat[k][l] = self[i_start + k][j_start + l];
            }
        }
        Some(sub_mat)
    }

    pub fn get_sub_mat_on(&self, i_start: usize, j_start: usize, mut i_len: usize,
                          mut j_len: usize, onto_rows: usize, onto_cols: usize)
                          -> Option<FloatMat> {
        // first ensure i and j are within a proper range
        if i_start > self.rows || j_start > self.cols {
            return None;
        }

        // then we might want to clamp k and l
        let delta_k = (self.rows) as i32 - (i_start + i_len) as i32;
        let delta_l = (self.cols) as i32 - (j_start + j_len) as i32 ;
        if delta_k < 0 {
            i_len -= delta_k.abs() as usize;
        }
        if delta_l < 0 {
            j_len -= delta_l.abs() as usize;
        }

        //Then we want to ensure O,P is big enought to fit data in
        if i_start + i_len > onto_rows || j_start + j_len > onto_cols {
            return None;
        }

        let mut sub_mat_on = FloatMat::zero(onto_rows, onto_cols);
        for a in i_start..i_len {
            for b in j_start..j_len {
                sub_mat_on[a][b] = self[a][b];
            }
        }
        Some(sub_mat_on)
    }

    //#endregion getters

    //#region data manipulation

    fn swap_rows(&mut self, i: usize, j: usize) {
        let row_i = self[i].clone();
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

    fn set_row(&mut self, index: usize, new_row: FloatN) {
        assert_eq!(self.cols, new_row.len(), "Error: new row is not of the proper size");
        self.data[index] = new_row;
    }

    pub fn set_column(&mut self, col_idx: usize, column: FloatN) {
        assert_eq!(self.rows, column.len(), "Error: new col is not of the proper size");
        for row_idx in 0..self.rows {
            self[row_idx][col_idx] = column[row_idx];
        }
    }

    //#endregion data manipulation

    //#region checks

    pub fn is_upper_triangular(&self) -> bool {
        for i in 0..self.rows {
            for j in 0..i {
                if self[i][j] != 0.0 {
                    return false;
                }
            }
        }
        true
    }

    pub fn is_lower_triangular(&self) -> bool {
        for i in 0..self.rows {
            for j in (i + 1)..self.cols {
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
        for i in 0..self.rows {
            for j in 0..self.cols {
                if (self[i][j] - other[i][j]).abs() > f64::EPSILON * 10.0 {
                    return false;
                }
            }
        }
        true
    }

    //#endregion checks

    //#region matrix operation

    pub fn transpose(&self) -> FloatMat {
        let mut data = Self::zero(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                data[j][i] = self[i][j];
            }
        }
        data
    }

    pub fn scale(&self, factor: f64) -> Self {
        let mut result = Self::zero(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[i][j] = self[i][j] * factor;
            }
        }
        result
    }

    pub fn add(&self, other: &Self) -> Self {
        let mut result = Self::zero(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
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

    pub fn mul(&self, other: &FloatMat) -> FloatMat {
        assert_eq!(self.cols, other.rows, "Error: Matrixs are not compatible for multiplication");

        let mut result = FloatMat::zero(self.rows, other.cols);

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
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

        for i in 0..self.rows {
            // Find pivot row
            let mut max_row = i;
            for j in i+1..self.rows {
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
                for k in 0..self.cols {
                    rref[i][k] /= pivot_coeff;
                }
            }

            // Make all rows below pivot row zero in current column
            for j in i+1..self.rows {
                let factor = rref[j][pivot];
                rref.add_scaled_row(j, i, -factor);
            }

            pivot += 1;
            if pivot >= self.cols {
                break;
            }
        }
        rref
    }

    pub fn reduced_row_echelon_form(&self) -> Self {
        let mut rref = self.row_echelon_form();
        let mut pivot = 0;

        for i in 0..self.rows {
            // Find pivot row
            let mut pivot_row = None;
            for j in pivot..self.cols {
                if rref[i][j] != 0.0 {
                    pivot_row = Some(j);
                    break;
                }
            }

            // If there is a pivot in this row
            if let Some(pivot_col) = pivot_row {
                // Make all other rows zero in current column
                for j in 0..self.rows {
                    if i != j && rref[j][pivot_col] != 0.0 {
                        let factor = rref[j][pivot_col];
                        rref.add_scaled_row(j, i, -factor);
                    }
                }

                pivot += 1;
                if pivot >= self.cols {
                    break;
                }
            }
        }
        rref
    }

    pub fn rank(&self) -> i32 {
        let ref rref = self.row_echelon_form();
        let mut rank = 0;
        for i in 0..self.rows {
            let mut is_zero_row = true;
            for j in 0..self.cols {
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

    pub fn qr_decomposition_gs(&self) -> (FloatMat, FloatMat) {
        let mut q = FloatMat::zero(self.rows, self.cols);
        let mut r = FloatMat::zero(self.cols, self.cols);

        for j in 0..self.cols {
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

    pub fn lq_decomposition_gs(&self) -> (FloatMat, FloatMat) {
        let (q_transpose, r_transpose) =
            self.transpose().qr_decomposition_gs();
        let q = q_transpose.transpose();
        let r = r_transpose.transpose();
        (r, q)
    }

    pub fn qr_decomposition_householder(&self) -> (FloatMat, FloatMat) {
        let mut q = FloatMat::identity(self.rows, self.cols);
        let mut r = self.get_sub_mat_on(0,0, self.cols, self.cols,
                                        self.cols, self.cols).unwrap();
        for j in 0..self.cols.min(self.rows-1) {
            let mut v = r.get_column(j);
            for k in 0..j {
                v[k] = 0.0;
            }
            let norm = v.magnitude();
            let sign = if v[0] > 0.0 { 1.0 } else { -1.0 };
            v[j] = v[j] + sign * norm;
            let dot = v.dot(&v);
            let h = FloatMat::identity(self.cols, self.cols)
                .sub(&v.kronecker(&v.scale(2.0 / dot)));
            r = h.mul(&r);
            q = q.mul(&h);
        }

        (q, r)
    }

    //#endregion matrix algorithm

    //#region squared matrix

    //#region checks

    pub fn is_symmetric(&self) -> bool {
        assert_eq!(self.rows, self.cols, "Error: Matrix is not square");
        for i in 0..self.rows {
            for j in 0..self.cols {
                if self[i][j] != self[j][i] {
                    return false;
                }
            }
        }
        true
    }

    pub fn is_identity(&self) -> bool {
        assert_eq!(self.rows, self.cols, "Error: Matrix is not square");
        for i in 0..self.rows {
            for j in 0..self.cols {
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
        assert_eq!(self.rows, self.cols, "Error: Matrix is not square");
        let mut rref = self.clone();
        //Gaussian elimination with maximum selection in column
        for i in 0..self.rows {
            // Find the row with maximum absolute value in the i-th column
            let mut j_max = i;
            for j in (i + 1)..self.rows {
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
            for j in (i + 1)..self.rows {
                let factor = rref[j][i] / rref[i][i];
                rref.add_scaled_row(j, i, -factor);
            }
        }

        rref
    }

    pub fn lu_decomposition_gaussian_elimination(&self) -> Option<(FloatMat, FloatMat)> {
        assert_eq!(self.rows, self.cols, "Error: Matrix is not square");
        let mut a = self.clone();
        let mut l = FloatMat::identity(self.rows, self.rows);
        let mut u = FloatMat::zero(self.rows, self.rows);

        for i in 0..self.rows {
            // Perform row swaps to avoid zero pivots
            for j in i..self.rows {
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
                if j == self.rows - 1 {
                    // Matrix is singular
                    return None;
                }
            }

            // Perform Gaussian elimination on current column
            for j in i+1..self.rows {
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

        for i in 0..self.rows {
            u.set_row(i, a[i].clone());
        }

        Some((l, u))
    }

    pub fn lu_decomposition_doolittle(&self) -> Option<(FloatMat, FloatMat)> {
        assert_eq!(self.rows, self.cols, "Error: Matrix is not square");
        let mut l = FloatMat::identity(self.rows, self.rows);
        let mut u = FloatMat::zero(self.rows, self.rows);

        for j in 0..self.rows {
            for i in j..self.rows {
                let sum = (0..j).fold(0.0, |acc, k| acc + l[j][k] * u[k][i]);
                u[j][i] = self[j][i] - sum;

                if j == i && l[j][i] == 0.0 {
                    return None;
                }
            }

            for i in (j + 1)..self.rows {
                let sum = (0..j).fold(0.0, |acc, k| acc + l[i][k] * u[k][j]);
                l[i][j] = (self[i][j] - sum) / u[j][j];
            }
        }

        Some((l, u))
    }

    pub fn lu_decomposition_crout(&self) -> Option<(FloatMat, FloatMat)> {
        assert_eq!(self.rows, self.cols, "Error: Matrix is not square");
        let mut u = FloatMat::identity(self.rows, self.rows);
        let mut l = FloatMat::zero(self.rows, self.rows);

        for j in 0..self.rows {
            for i in j..self.rows {
                let sum = (0..j).fold(0.0, |acc, k| acc + l[i][k] * u[k][j]);
                l[i][j] = self[i][j] - sum;

                if i == j && l[i][j] == 0.0 {
                    return None;
                }
            }

            for i in j..self.rows {
                let sum = (0..j).fold(0.0, |acc, k| acc + l[j][k] * u[k][i]);
                u[j][i] = (self[j][i] - sum) / l[j][j];
            }
        }

        Some((l, u))
    }

    pub fn det(&self) -> f64 {
        assert_eq!(self.rows, self.cols, "Error: Matrix is not square");
        let mut det = if self.rows % 2 == 0 {1.0} else {-1.0};

        // Triangulize the matrix
        let mat_triangulated = self.triangulate_gaussian();

        // Compute the determinant as the product of the diagonal elements
        for i in 0..self.rows {
            det *= mat_triangulated[i][i];
        }

        det
    }

    pub fn trace(&self) -> f64 {
        assert_eq!(self.rows, self.cols, "Error: Matrix is not square");

        let mut tr = 0.0;

        for i in 0..self.rows {
            tr += self[i][i];
        }

        tr
    }

    pub fn characteristic_polynomial(&self) -> Polynomial<1> {
        assert_eq!(self.rows, self.cols, "Error: Matrix is not square");
        //faddeev-leverrier method
        let mut poly = Polynomial::new();

        let mut c = Self::zero(self.rows, self.rows);
        let mut cn = 1.0;
        poly.add_term(cn, [self.rows ]);

        for i in 1..(self.rows + 1) {
            c = self.mul(&c.add(&Self::identity(self.rows, self.cols).scale(cn)));
            cn = c.trace() / (-(i as f64));
            poly.add_term(cn, [self.rows - i]);
        }

        poly
    }

    //#endregion matrix algorithms

    //#endregion squared matrix
}

//#endregion FloatMat

//#region Tests

#[cfg(test)]
mod float_mat_tests {
    use super::*;

    #[test]
    fn float_mat_new_test() {
        let row_a = FloatN::new(vec!(1.0,2.0,3.0));
        let row_b = row_a.scale(2.0);
        let row_c = row_a.scale(3.0);
        let mat = FloatMat::new(vec![row_a, row_b, row_c]);
        assert_eq!(mat[0][0], 1.0);
        assert_eq!(mat[1][1], 4.0);
        assert_eq!(mat[2][2], 9.0);
    }

    #[test]
    fn float_mat_identity_test() {
        let identity = FloatMat::identity(3,3);
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
        let row_a = FloatN::new(vec![1.0,2.0,3.0]);
        let row_b = row_a.scale(2.0);
        let mat = FloatMat::new(vec![row_a, row_b]);
        let mat_prime = mat.transpose();
        assert_eq!(mat_prime[0][0], 1.0);
        assert_eq!(mat_prime[0][1], 2.0);
        assert_eq!(mat_prime[2][0], 3.0);
    }

    #[test]
    fn float_mat_scale_test() {
        let row_a = FloatN::new(vec![1.0,2.0]);
        let row_b = row_a.scale(2.0);
        let mat = FloatMat::new(vec![row_a, row_b]);
        let mat_scaled = mat.scale(2.0);
        assert_eq!(mat_scaled[0][0], 2.0);
        assert_eq!(mat_scaled[1][1], 8.0);
    }

    #[test]
    fn float_mat_add_test() {
        let row_a = FloatN::new(vec![1.0,2.0]);
        let row_b = FloatN::new(vec![2.0,3.0]);
        let mat_a = FloatMat::new(vec![row_a.clone(), row_b.clone()]);
        let mat_b = FloatMat::new(vec![row_b, row_a]);
        let mat_add = mat_a.add(&mat_b);
        assert_eq!(mat_add[0][0], 3.0);
        assert_eq!(mat_add[1][1], 5.0);
    }

    #[test]
    fn float_mat_mul_test() {
        let row_a = FloatN::new(vec![1.0,2.0]);
        let row_b = row_a.scale(2.0);
        let mat_a = FloatMat::new(vec![row_a, row_b]);
        let mat_b = mat_a.scale(2.0);
        let mat_mul = mat_a.mul(&mat_b);
        assert_eq!(mat_mul[0][0], 10.0);
        assert_eq!(mat_mul[1][1], 40.0);
    }

    #[test]
    fn float_mat_swap_rows_test() {
        let row_a = FloatN::new(vec![1.0,2.0]);
        let row_b = row_a.scale(2.0);
        let mut mat_a = FloatMat::new(vec![row_a, row_b]);
        mat_a.swap_rows(0,1);
        assert_eq!(mat_a[0][0], 2.0);
        assert_eq!(mat_a[1][1], 2.0);
    }

    #[test]
    fn float_mat_scale_row_test() {
        let row_a = FloatN::new(vec![1.0,2.0]);
        let row_b = row_a.scale(2.0);
        let mut mat_a = FloatMat::new(vec![row_a, row_b]);
        mat_a.scale_row(0,2.0);
        assert_eq!(mat_a[0][0], 2.0);
    }

    #[test]
    fn float_mat_add_scale_row_test() {
        let row_a = FloatN::new(vec![1.0,2.0]);
        let row_b = row_a.scale(2.0);
        let mut mat_a = FloatMat::new(vec![row_a, row_b]);
        mat_a.add_scaled_row(0,1,2.0);
        assert_eq!(mat_a[0][0], 5.0);
    }

    #[test]
    fn float_mat_triangulate_gaussian_test() {
        let row_a = FloatN::new(vec![1.0,2.0,3.0]);
        let row_b = FloatN::new(vec![3.0,7.0,1.0]);
        let row_c = FloatN::new(vec![5.0,1.0,2.0]);
        let mat_a = FloatMat::new(vec![row_a, row_b, row_c]);
        let triangulated =  mat_a.triangulate_gaussian();
        assert_eq!(triangulated[0][0], 5.0);
        assert_eq!(triangulated[1][1], 6.4);
        assert_eq!(triangulated[2][2], 2.65625);
    }

    #[test]
    fn float_mat_lu_decomposition_gaussian_elimination_test() {
        let row_a = FloatN::new(vec![1.0,2.0,3.0]);
        let row_b = FloatN::new(vec![3.0,7.0,1.0]);
        let row_c = FloatN::new(vec![5.0,1.0,2.0]);
        let mat_a = FloatMat::new(vec![row_a, row_b, row_c]);
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
        let row_a = FloatN::new(vec![1.0,2.0,3.0]);
        let row_b = FloatN::new(vec![3.0,7.0,1.0]);
        let row_c = FloatN::new(vec![5.0,1.0,2.0]);
        let mat_a = FloatMat::new(vec![row_a, row_b, row_c]);
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
        let row_a = FloatN::new(vec![1.0,2.0,3.0]);
        let row_b = FloatN::new(vec![3.0,7.0,1.0]);
        let row_c = FloatN::new(vec![5.0,1.0,2.0]);
        let mat_a = FloatMat::new(vec![row_a, row_b, row_c]);
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
        let row_a = FloatN::new(vec![1.0,2.0,3.0]);
        let row_b = FloatN::new(vec![3.0,7.0,1.0]);
        let row_c = FloatN::new(vec![5.0,1.0,2.0]);
        let mat_a = FloatMat::new(vec![row_a, row_b, row_c]);
        let det = mat_a.det();
        assert_eq!(det, -85.0);
    }

    #[test]
    fn float_mat_row_echelon_form_test() {
        let row_a = FloatN::new(vec![1.0,2.0,3.0,4.0]);
        let row_b = FloatN::new(vec![3.0,7.0,1.0,2.0]);
        let row_c = FloatN::new(vec![5.0,1.0,2.0,1.0]);
        let mat_a = FloatMat::new(vec![row_a, row_b, row_c]);
        let echelon = mat_a.reduced_row_echelon_form();
        assert_eq!(format!("{:.2}", echelon[0][3]), "-0.36");
        assert_eq!(format!("{:.2}", echelon[1][3]), "0.26");
        assert_eq!(format!("{:.2}", echelon[2][3]), "1.28");
    }

    #[test]
    fn float_mat_is_symmetric_test() {
        let row_a = FloatN::new(vec![1.0, 2.0]);
        let row_b = FloatN::new(vec![2.0, 1.0]);
        let mat_a = FloatMat::new(vec![row_a.clone(), row_b.clone()]);
        let mat_b = FloatMat::new(vec![row_a.clone(), row_a.clone()]);
        assert_eq!(mat_a.is_symmetric(), true);
        assert_eq!(mat_b.is_symmetric(), false);
    }

    #[test]
    fn float_mat_rank_test() {
        let row_a = FloatN::new(vec![1.0,2.0,3.0,4.0]);
        let row_b = FloatN::new(vec![3.0,7.0,1.0,2.0]);
        let row_c = FloatN::new(vec![5.0,1.0,2.0,1.0]);
        let mat_a = FloatMat::new(vec![row_a.clone(), row_b, row_c, row_a]);
        let rank = mat_a.rank();
        assert_eq!(rank, 3);
    }

    #[test]
    fn float_mat_is_upper_lower_triangular_test() {
        let row_a = FloatN::new(vec![1.0,2.0,3.0]);
        let row_b = FloatN::new(vec![0.0,1.0,2.0]);
        let mat_a = FloatMat::new(vec![row_a, row_b]);
        assert_eq!(mat_a.is_upper_triangular(), true);
        assert_eq!(mat_a.is_lower_triangular(), false);
        let row_c = FloatN::new(vec![1.0, 0.0, 0.0]);
        let row_d = FloatN::new(vec![2.0, 1.0, 0.0]);
        let mat_b = FloatMat::new(vec![row_c, row_d]);
        assert_eq!(mat_b.is_upper_triangular(), false);
        assert_eq!(mat_b.is_lower_triangular(), true);
    }

    #[test]
    fn float_mat_is_identity_test() {
        let row_a = FloatN::new(vec![1.0,0.0,0.0]);
        let row_b = FloatN::new(vec![0.0,1.0,0.0]);
        let row_c = FloatN::new(vec![0.0, 0.0, 1.0]);
        let mut mat_a = FloatMat::new(vec![row_a, row_b, row_c]);
        assert_eq!(mat_a.is_identity(), true);
        mat_a.swap_entries(0,0,0,1);
        assert_eq!(mat_a.is_identity(), false);
    }

    #[test]
    fn float_mat_is_orthogonal_test() {
        let row_a = FloatN::new(vec![1.0,0.0,0.0]);
        let row_b = FloatN::new(vec![0.0,0.0,1.0]);
        let row_c = FloatN::new(vec![0.0,-1.0,0.0]);
        let mat_a = FloatMat::new(vec![row_a, row_b, row_c]);
        assert_eq!(mat_a.is_orthogonal(), true);
        let row_d = FloatN::new(vec![1.0,2.0,3.0]);
        let row_e = FloatN::new(vec![4.0,5.0,6.0]);
        let row_f = FloatN::new(vec![7.0,8.0,9.0]);
        let mat_b = FloatMat::new(vec![row_d, row_e, row_f]);
        assert_eq!(mat_b.is_orthogonal(), false);
    }

    #[test]
    fn float_mat_is_equal_test() {
        let mat_a = FloatMat::identity(2,2);
        let mat_b = FloatMat::identity(2,2);
        assert_eq!(mat_a.is_equal(&mat_b), true);
        let mut mat_c = FloatMat::identity(2,2);
        mat_c.swap_entries(0,0,0,1);
        assert_eq!(mat_a.is_equal(&mat_c), false);
    }

    #[test]
    fn float_mat_qr_decomposition_gs_test() {
        let row_a = FloatN::new(vec![1.0,2.0,3.0,4.0]);
        let row_b = FloatN::new(vec![3.0,7.0,1.0,2.0]);
        let row_c = FloatN::new(vec![5.0,1.0,2.0,1.0]);
        let mat_a = FloatMat::new(vec![row_a, row_b, row_c]);
        let (q, r) = mat_a.qr_decomposition_gs();
        assert_eq!(q.is_orthogonal(), true);
        assert_eq!(r.is_upper_triangular(), true);
        let mul = q.mul(&r);
        assert_eq!(mul.is_equal(&mat_a), true);
    }

    #[test]
    fn float_mat_lq_decomposition_gs_test() {
        let row_a = FloatN::new(vec![1.0,2.0,3.0,4.0]);
        let row_b = FloatN::new(vec![3.0,7.0,1.0,2.0]);
        let row_c = FloatN::new(vec![5.0,1.0,2.0,1.0]);
        let mat_a = FloatMat::new(vec![row_a, row_b, row_c]);
        let (l, q) = mat_a.lq_decomposition_gs();
        assert_eq!(q.is_orthogonal(), true);
        assert_eq!(l.is_lower_triangular(), true);
        let mul = l.mul(&q);
        assert_eq!(mul.is_equal(&mat_a), true);
    }

    #[test]
    fn float_mat_qr_decomposition_householder_test() {
        let row_a = FloatN::new(vec![1.0,2.0,3.0,4.0]);
        let row_b = FloatN::new(vec![3.0,7.0,1.0,2.0]);
        let row_c = FloatN::new(vec![5.0,1.0,2.0,1.0]);
        let mat_a = FloatMat::new(vec![row_a, row_b, row_c]);
        let (q, r) = mat_a.qr_decomposition_householder();
        assert_eq!(q.is_orthogonal(), true);
        assert_eq!(r.is_upper_triangular(), true);
        let mul = q.mul(&r);
        assert_eq!(mul.is_equal(&mat_a), true);
    }

    #[test]
    fn float_mat_char_poly_test_1() {
        let row_a = FloatN::new(vec![1.0,2.0]);
        let row_b = FloatN::new(vec![3.0,7.0]);
        let mat_a = FloatMat::new(vec![row_a, row_b]);
        let poly = mat_a.characteristic_polynomial();
        let mut expected = Polynomial::<1>::new();
        expected.add_term(1.0, [2]);
        expected.add_term(-8.0, [1]);
        expected.add_term(1.0, [0]);
        assert_eq!(poly, expected);
    }

    #[test]
    fn float_mat_char_poly_test_2() {
        let row_a = FloatN::new(vec![3.0,1.0,5.0]);
        let row_b = FloatN::new(vec![3.0,3.0,1.0]);
        let row_c = FloatN::new(vec![4.0,6.0,4.0]);
        let mat_a = FloatMat::new(vec![row_a, row_b, row_c]);
        let poly = mat_a.characteristic_polynomial();
        let mut expected = Polynomial::<1>::new();
        expected.add_term(1.0, [3]);
        expected.add_term(-10.0, [2]);
        expected.add_term(4.0, [1]);
        expected.add_term(-40.0, [0]);
        assert_eq!(poly, expected);
    }

    #[test]
    fn float_mat_char_poly_test_3() {
        let row_a = FloatN::new(vec![1.0,2.0,3.0,1.0]);
        let row_b = FloatN::new(vec![3.0,7.0,1.0,2.0]);
        let row_c = FloatN::new(vec![5.0,1.0,2.0,3.0]);
        let row_d = FloatN::new(vec![8.0,4.0,2.0,1.0]);
        let mat_a = FloatMat::new(vec![row_a, row_b, row_c, row_d]);
        let poly = mat_a.characteristic_polynomial();
        let mut expected = Polynomial::<1>::new();
        expected.add_term(1.0, [4]);
        expected.add_term(-11.0, [3]);
        expected.add_term(-11.0, [2]);
        expected.add_term(86.0, [1]);
        expected.add_term(317.0, [0]);
        assert_eq!(poly, expected);
    }
}

//#endregion Tests