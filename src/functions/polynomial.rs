use std::collections::HashMap;

//#region Polynomial

#[derive(Debug, PartialEq)]
pub struct Polynomial<const AMOUNT_OF_VAR: usize> {
    terms: HashMap<[usize; AMOUNT_OF_VAR], f64>,
}

impl<const AMOUNT_OF_VAR: usize> Polynomial<AMOUNT_OF_VAR> {

    //#region ctor

    pub fn new() -> Self {
        Self { terms: HashMap::new() }
    }

    //#endregion ctor

    //#region getters

    pub fn len(&self) -> usize {
        self.terms.len()
    }

    //#ednregion getters

    //#region data manipulation

    pub fn add_term(&mut self, coefficient: f64, exponents: [usize; AMOUNT_OF_VAR]) {
        if coefficient == 0.0 {
            return;
        }
        if let Some(current_coefficient) = self.terms.get_mut(&exponents) {
            *current_coefficient += coefficient;
        } else {
            self.terms.insert(exponents, coefficient);
        }
    }

    //#endregion data manipulation

    //#region pairwise operation

    pub fn add(&self, other: &Self) -> Self {
        let mut result = Self::new();

        for (term, coeff) in &self.terms {
            result.add_term(*coeff, *term);
        }

        for (term, coeff) in &other.terms {
            result.add_term(*coeff, *term);
        }

        result
    }

    pub fn sub(&self, other: &Self) -> Self {
        let mut result = Self::new();

        for (term, coeff) in &self.terms {
            result.add_term(*coeff, *term);
        }

        for (term, coeff) in &other.terms {
            result.add_term((-1.0) * (*coeff), *term);
        }

        result
    }

    pub fn mul(&self, other: &Self) -> Self {
        let mut result = Self::new();

        for (term1, coeff1) in &self.terms {
            for (term2, coeff2) in &other.terms {
                let mut term = [0; AMOUNT_OF_VAR];
                for i in 0..AMOUNT_OF_VAR {
                    term[i] = term1[i] + term2[i];
                }
                let coeff = coeff1 * coeff2;
                result.add_term(coeff, term);
            }
        }

        result
    }

    //#endregion pairwise operation

    //#region unary operation

    pub fn degree(&self) -> usize {
        let opt = self.terms.keys().map(|k| k.iter().sum()).max();
        return match opt {
            None => 0,
            Some(deg) => deg
        }
    }

    pub fn evaluate(&self, values: [f64; AMOUNT_OF_VAR]) -> f64 {
        let mut sum = 0.0;

        for (exponents, coefficient) in &self.terms {
            let mut term = *coefficient;
            for (i, &exp) in exponents.iter().enumerate() {
                term *= values[i].powi(exp as i32);
            }
            sum += term;
        }

        sum
    }

    pub fn differentiate(&self, derivatives: [usize; AMOUNT_OF_VAR]) -> Self {
        let mut result = Self::new();

        for (exponents, &coeff) in &self.terms {
            let mut derivative_coeff = coeff;
            let mut derivative_exponents = exponents.clone();
            let mut discard_term = false;
            for (i, &order) in derivatives.iter().enumerate() {
                if  derivative_exponents[i] < order {
                    discard_term = true;
                } else {
                    for _ in 1..=order {
                        derivative_coeff *= derivative_exponents[i] as f64;
                        derivative_exponents[i] -= 1;
                    }
                }
            }

            if !discard_term {
                result.add_term(derivative_coeff, derivative_exponents);
            }
        }

        result
    }

    pub fn integrate<const AMOUNT_OF_INTEGRATION: usize>(
        &self, integrations_vars: [usize; AMOUNT_OF_INTEGRATION],
        integration_constants: [f64; AMOUNT_OF_INTEGRATION]) -> Self {
        let mut result = Self::new();

        // Calculate the integral for each term
        for (exponents, &coeff) in &self.terms {
            let mut integrated_coeff = coeff;
            let mut integrated_exponents = exponents.clone();
            // Apply each integration
            for (_, &var_index) in integrations_vars.iter().enumerate() {
                integrated_coeff /= (integrated_exponents[var_index] + 1) as f64;
                integrated_exponents[var_index] += 1;
            }

            // Add the term with the integrated coefficients to the result
            result.add_term(integrated_coeff, integrated_exponents);
        }

        // Calculate the integration constants terms
        // Basically for N Integration, the Mth terms should be the associated constant
        // integrated on the range ]M,N]
        for (term_idx, &integration_constant) in integration_constants.iter().enumerate() {
            let mut integrated_coeff = integration_constant;
            let mut integrated_exponents = [0 as usize; AMOUNT_OF_VAR];

            for (integration_idx, &var_index) in integrations_vars.iter().enumerate() {
                if integration_idx <= term_idx {
                    continue;
                }
                integrated_coeff /= (integrated_exponents[var_index] + 1) as f64;
                integrated_exponents[var_index] += 1;
            }

            // Add the term with the integrated coefficients to the result
            result.add_term(integrated_coeff, integrated_exponents);
        }

        result
    }

    //#endregion unary operation
}

//#endregion Polynomial

//#region Tests

#[cfg(test)]
mod polynomial_test {
    use super::*;

    #[test]
    fn polynomial_new_add_term_test() {
        let mut poly = Polynomial::<3>::new();
        poly.add_term(1.0, [1,2,3]);
        poly.add_term(3.0, [2,0,4]);
        assert_eq!(poly.terms.get(&[1,2,3]), Some(&1.0));
        assert_eq!(poly.terms.get(&[3,2,1]), None);
        poly.add_term(2.0, [1,2,3]);
        assert_eq!(poly.terms.get(&[1,2,3]), Some(&3.0));
    }

    #[test]
    fn polynomial_degree_test() {
        let p1 = Polynomial::<2>::new();
        assert_eq!(p1.degree(), 0);

        let mut p2 = Polynomial::<2>::new();
        p2.add_term(1.0,[1, 0]);
        p2.add_term(2.0, [0, 1]);
        assert_eq!(p2.degree(), 1);

        let mut p3 = Polynomial::<2>::new();
        p3.add_term(1.0,[2, 0]);
        p3.add_term(2.0,[1, 1]);
        p3.add_term(3.0,[0, 4]);
        assert_eq!(p3.degree(), 4);
    }

    #[test]
    fn polynomial_add_test() {
        let mut p1 = Polynomial::<3>::new();
        let mut p2 = Polynomial::<3>::new();

        p1.add_term(2.0, [1, 2, 0]);
        p1.add_term(-1.0, [0, 0, 1]);
        p1.add_term(3.0, [0, 1, 0]);
        p1.add_term(2.0, [1,0,0]);

        p2.add_term(1.0, [1, 2, 0]);
        p2.add_term(2.0, [0, 0, 1]);
        p2.add_term(-1.0, [0, 1, 0]);
        p2.add_term(5.0, [1,1,1]);

        let p3 = p1.add(&p2);

        assert_eq!(p3.len(), 5);

        assert_eq!(p3.terms[&[1, 2, 0]], 3.0);
        assert_eq!(p3.terms[&[0, 0, 1]], 1.0);
        assert_eq!(p3.terms[&[0, 1, 0]], 2.0);
        assert_eq!(p3.terms[&[1, 0, 0]], 2.0);
        assert_eq!(p3.terms[&[1, 1, 1]], 5.0);
    }

    #[test]
    fn polynomial_sub_test() {
        let mut p1 = Polynomial::<2>::new();
        p1.add_term(3.0, [1, 2]);
        p1.add_term(-2.0, [0, 0]);

        let mut p2 = Polynomial::<2>::new();
        p2.add_term(1.0, [1, 2]);
        p2.add_term(-1.0, [0, 0]);

        let p3 = p1.sub(&p2);

        assert_eq!(p3.len(), 2);

        assert_eq!(p3.terms[&[1, 2]], 2.0);
        assert_eq!(p3.terms[&[0, 0]], -1.0);
    }

    #[test]
    fn polynomial_mul_test() {
        let mut p1 = Polynomial::<2>::new();
        p1.add_term(2.0, [2, 0]);
        p1.add_term(3.0, [1, 1]);
        p1.add_term(-1.0, [0, 2]);

        let mut p2 = Polynomial::<2>::new();
        p2.add_term(1.0, [1, 0]);
        p2.add_term(2.0, [0, 1]);
        p2.add_term(3.0, [1, 1]);

        let mut expected = Polynomial::<2>::new();
        expected.add_term(2.0, [3, 0]);
        expected.add_term(7.0, [2, 1]);
        expected.add_term(5.0, [1, 2]);
        expected.add_term(-2.0, [0, 3]);
        expected.add_term(6.0, [3, 1]);
        expected.add_term(9.0, [2, 2]);
        expected.add_term(-3.0, [1, 3]);

        assert_eq!(p1.mul(&p2), expected);
    }

    #[test]
    fn polynomial_evaluate_test() {
        let p = Polynomial::<2>::new();
        assert_eq!(p.evaluate([1.0, 2.0]), 0.0);

        let mut p = Polynomial::<2>::new();
        p.add_term(1.0, [0, 0]);
        p.add_term(2.0, [1, 0]);
        p.add_term(3.0, [0, 1]);
        p.add_term(4.0, [1, 1]);
        assert_eq!(p.evaluate([2.0, 3.0]), 38.0);
    }

    #[test]
    fn polynomial_differentiate_test() {
        let mut p = Polynomial::<3>::new();
        p.add_term(3.0, [1, 0, 2]);
        p.add_term(2.0, [0, 3, 1]);
        p.add_term(4.0, [2, 1, 0]);

        let dp_dx1 = p.differentiate([1, 0, 0]);
        let mut expected_dp_dx1 = Polynomial::<3>::new();
        expected_dp_dx1.add_term(3.0, [0, 0, 2]);
        expected_dp_dx1.add_term(8.0, [1, 1, 0]);

        let dp_dx2 = p.differentiate([0, 1, 0]);
        let mut expected_dp_dx2 = Polynomial::<3>::new();
        expected_dp_dx2.add_term(6.0, [0, 2, 1]);
        expected_dp_dx2.add_term(4.0, [2, 0, 0]);

        let dp_dx3 = p.differentiate([0, 0, 1]);
        let mut expected_dp_dx3 = Polynomial::<3>::new();
        expected_dp_dx3.add_term(6.0, [1, 0, 1]);
        expected_dp_dx3.add_term(2.0, [0, 3, 0]);

        let dp_dx1_dx1 = p.differentiate([2,0,0]);
        let mut expected_dp_dx1_dx1 = Polynomial::<3>::new();
        expected_dp_dx1_dx1.add_term(8.0, [0,1,0]);

        let dp_dx2_dx2 = p.differentiate([0,2,0]);
        let mut expected_dp_dx2_dx2 = Polynomial::<3>::new();
        expected_dp_dx2_dx2.add_term(12.0, [0,1,1]);

        let dp_dx3_dx3 = p.differentiate([0,0,2]);
        let mut expected_dp_dx3_dx3 = Polynomial::<3>::new();
        expected_dp_dx3_dx3.add_term(6.0, [1,0,0]);

        let dp_dx1_dx2 = p.differentiate([1,1,0]);
        let mut expected_dp_dx1_dx2 = Polynomial::<3>::new();
        expected_dp_dx1_dx2.add_term(8.0, [1,0,0]);

        let dp_dx2_dx3 = p.differentiate([0,1,1]);
        let mut expected_dp_dx2_dx3 = Polynomial::<3>::new();
        expected_dp_dx2_dx3.add_term(6.0, [0,2,0]);

        assert_eq!(dp_dx1, expected_dp_dx1);
        assert_eq!(dp_dx2, expected_dp_dx2);
        assert_eq!(dp_dx3, expected_dp_dx3);
        assert_eq!(dp_dx1_dx1, expected_dp_dx1_dx1);
        assert_eq!(dp_dx2_dx2, expected_dp_dx2_dx2);
        assert_eq!(dp_dx3_dx3, expected_dp_dx3_dx3);
        assert_eq!(dp_dx1_dx2, expected_dp_dx1_dx2);
        assert_eq!(dp_dx2_dx3, expected_dp_dx2_dx3);
    }

    #[test]
    fn polynomial_integrate_test() {
        let mut p1 = Polynomial::<3>::new();
        p1.add_term(3.0, [2, 0, 1]);
        p1.add_term(-4.0, [0, 3, 0]);
        p1.add_term(2.0, [1, 1, 1]);

        let p2 = p1.integrate([0, 1, 1], [1.0, 2.0, 0.0]);
        let mut expected = Polynomial::<3>::new();
        expected.add_term(1.0 / 2.0, [3, 2, 1]);
        expected.add_term(-1.0 / 5.0, [1, 5, 0]);
        expected.add_term(1.0 / 6.0, [2, 3, 1]);
        expected.add_term(1.0 / 2.0, [0, 2, 0]);
        expected.add_term(2.0, [0, 1, 0]);

        assert_eq!(p2, expected);
    }
}

//#endregion Tests