use crate::polynomial::Polynomial;

#[test]
fn polynomial_add() {
  let mut poly = Polynomial::new();
  poly.set_coefficient(0, 5.0);
  poly.set_coefficient(1, 6.0);
  poly.set_coefficient(2, 4.0);

  for i in -1000..1000 {
    let i = i as f64 * 0.001;
    assert!(approx_eq!(f64, (&poly + i).evaluate(i), 4.0*i.powi(2) + 6.0*i + 5.0 + i, epsilon=0.0001));
  }

  for i in -1000..1000 {
    let i = i as f64 * 0.001;
    poly = poly + i;
    assert!(approx_eq!(f64, poly.evaluate(i), 4.0*i.powi(2) + 6.0*i + 5.0 + i, epsilon=0.0001));
    poly.set_coefficient(0, 5.0);
  }

  for i in -1000..1000 {
    let i = i as f64 * 0.001;
    poly += i;
    assert!(approx_eq!(f64, poly.evaluate(i), 4.0*i.powi(2) + 6.0*i + 5.0 + i, epsilon=0.0001));
    poly.set_coefficient(0, 5.0);
  }

  for i in -1000..1000 {
    let i = i as f64 * 0.001;
    let mut addend = Polynomial::new();
    addend.set_coefficient(1, 5.0 * i);
    assert!(approx_eq!(f64, (&poly + addend).evaluate(i), 4.0*i.powi(2) + (6.0+5.0*i)*i + 5.0, epsilon=0.0001));
  }

  for i in -1000..1000 {
    let i = i as f64 * 0.001;
    let mut addend = Polynomial::new();
    addend.set_coefficient(1, 5.0 * i);
    assert!(approx_eq!(f64, (&poly + &addend).evaluate(i), 4.0*i.powi(2) + (6.0+5.0*i)*i + 5.0, epsilon=0.0001));
  }

  for i in -1000..1000 {
    let i = i as f64 * 0.001;
    let mut addend = Polynomial::new();
    addend.set_coefficient(1, 5.0 * i);
    poly = poly + addend;
    assert!(approx_eq!(f64, poly.evaluate(i), 4.0*i.powi(2) + (6.0+5.0*i)*i + 5.0, epsilon=0.0001));
    poly.set_coefficient(1, 6.0);
  }

  for i in -1000..1000 {
    let i = i as f64 * 0.001;
    let mut addend = Polynomial::new();
    addend.set_coefficient(1, 5.0 * i);
    poly = poly + &addend;
    assert!(approx_eq!(f64, poly.evaluate(i), 4.0*i.powi(2) + (6.0+5.0*i)*i + 5.0, epsilon=0.0001));
    poly.set_coefficient(1, 6.0);
  }

  for i in -1000..1000 {
    let i = i as f64 * 0.001;
    let mut addend = Polynomial::new();
    addend.set_coefficient(1, 5.0 * i);
    poly = poly + addend;
    assert!(approx_eq!(f64, poly.evaluate(i), 4.0*i.powi(2) + (6.0+5.0*i)*i + 5.0, epsilon=0.0001));
    poly.set_coefficient(1, 6.0);
  }

  for i in -1000..1000 {
    let i = i as f64 * 0.001;
    let mut addend = Polynomial::new();
    addend.set_coefficient(1, 5.0 * i);
    poly += addend;
    assert!(approx_eq!(f64, poly.evaluate(i), 4.0*i.powi(2) + (6.0+5.0*i)*i + 5.0, epsilon=0.0001));
    poly.set_coefficient(1, 6.0);
  }

  for i in -1000..1000 {
    let i = i as f64 * 0.001;
    let mut addend = Polynomial::new();
    addend.set_coefficient(1, 5.0 * i);
    poly += &addend;
    assert!(approx_eq!(f64, poly.evaluate(i), 4.0*i.powi(2) + (6.0+5.0*i)*i + 5.0, epsilon=0.0001));
    poly.set_coefficient(1, 6.0);
  }
}

#[test]
fn polynomial_sub() {
  let mut poly = Polynomial::new();
  poly.set_coefficient(0, 5.0);
  poly.set_coefficient(1, 6.0);
  poly.set_coefficient(2, 4.0);

  for i in -1000..1000 {
    let i = i as f64 * 0.001;
    assert!(approx_eq!(f64, (&poly - i).evaluate(i), 4.0*i.powi(2) + 6.0*i + 5.0 - i, epsilon=0.0001));
  }

  for i in -1000..1000 {
    let i = i as f64 * 0.001;
    poly = poly - i;
    assert!(approx_eq!(f64, poly.evaluate(i), 4.0*i.powi(2) + 6.0*i + 5.0 - i, epsilon=0.0001));
    poly.set_coefficient(0, 5.0);
  }

  for i in -1000..1000 {
    let i = i as f64 * 0.001;
    poly -= i;
    assert!(approx_eq!(f64, poly.evaluate(i), 4.0*i.powi(2) + 6.0*i + 5.0 - i, epsilon=0.0001));
    poly.set_coefficient(0, 5.0);
  }

  for i in -1000..1000 {
    let i = i as f64 * 0.001;
    let mut addend = Polynomial::new();
    addend.set_coefficient(1, 5.0 * i);
    assert!(approx_eq!(f64, (&poly - addend).evaluate(i), 4.0*i.powi(2) + (6.0-5.0*i)*i + 5.0, epsilon=0.0001));
  }

  for i in -1000..1000 {
    let i = i as f64 * 0.001;
    let mut addend = Polynomial::new();
    addend.set_coefficient(1, 5.0 * i);
    assert!(approx_eq!(f64, (&poly - &addend).evaluate(i), 4.0*i.powi(2) + (6.0-5.0*i)*i + 5.0, epsilon=0.0001));
  }

  for i in -1000..1000 {
    let i = i as f64 * 0.001;
    let mut addend = Polynomial::new();
    addend.set_coefficient(1, 5.0 * i);
    poly = poly - addend;
    assert!(approx_eq!(f64, poly.evaluate(i), 4.0*i.powi(2) + (6.0-5.0*i)*i + 5.0, epsilon=0.0001));
    poly.set_coefficient(1, 6.0);
  }

  for i in -1000..1000 {
    let i = i as f64 * 0.001;
    let mut addend = Polynomial::new();
    addend.set_coefficient(1, 5.0 * i);
    poly = poly - &addend;
    assert!(approx_eq!(f64, poly.evaluate(i), 4.0*i.powi(2) + (6.0-5.0*i)*i + 5.0, epsilon=0.0001));
    poly.set_coefficient(1, 6.0);
  }

  for i in -1000..1000 {
    let i = i as f64 * 0.001;
    let mut addend = Polynomial::new();
    addend.set_coefficient(1, 5.0 * i);
    poly = poly - addend;
    assert!(approx_eq!(f64, poly.evaluate(i), 4.0*i.powi(2) + (6.0-5.0*i)*i + 5.0, epsilon=0.0001));
    poly.set_coefficient(1, 6.0);
  }

  for i in -1000..1000 {
    let i = i as f64 * 0.001;
    let mut addend = Polynomial::new();
    addend.set_coefficient(1, 5.0 * i);
    poly -= addend;
    assert!(approx_eq!(f64, poly.evaluate(i), 4.0*i.powi(2) + (6.0-5.0*i)*i + 5.0, epsilon=0.0001));
    poly.set_coefficient(1, 6.0);
  }

  for i in -1000..1000 {
    let i = i as f64 * 0.001;
    let mut addend = Polynomial::new();
    addend.set_coefficient(1, 5.0 * i);
    poly -= &addend;
    assert!(approx_eq!(f64, poly.evaluate(i), 4.0*i.powi(2) + (6.0-5.0*i)*i + 5.0, epsilon=0.0001));
    poly.set_coefficient(1, 6.0);
  }
}

#[test]
fn polynomial_mul() {
  let mut poly = Polynomial::new();
  poly.set_coefficient(0, 5.0);
  poly.set_coefficient(1, 6.0);
  poly.set_coefficient(2, 4.0);

  for i in -1000..1000 {
    let i = i as f64 * 0.001;
    assert!(approx_eq!(f64, (&poly * i).evaluate(i), i*(4.0*i.powi(2) + 6.0*i + 5.0), epsilon=0.0001));
  }

  for i in -1000..1000 {
    let i = i as f64 * 0.001;
    poly = poly * i;
    assert!(approx_eq!(f64, poly.evaluate(i), (4.0*i.powi(2) + 6.0*i + 5.0)*i, epsilon=0.0001));
    poly.set_coefficient(0, 5.0);
    poly.set_coefficient(1, 6.0);
    poly.set_coefficient(2, 4.0);
  }

  for i in -1000..1000 {
    let i = i as f64 * 0.001;
    poly *= i;
    assert!(approx_eq!(f64, poly.evaluate(i), (4.0*i.powi(2) + 6.0*i + 5.0)*i, epsilon=0.0001));
    poly.set_coefficient(0, 5.0);
    poly.set_coefficient(1, 6.0);
    poly.set_coefficient(2, 4.0);
  }
}

#[test]
fn polynomial_div() {
  let mut poly = Polynomial::new();
  poly.set_coefficient(0, 5.0);
  poly.set_coefficient(1, 6.0);
  poly.set_coefficient(2, 4.0);

  for i in -1000..1000 {
    let i = i as f64 * 0.001;
    if approx_eq!(f64, i, 0.0, epsilon=0.001) {
      continue;
    }
    assert!(approx_eq!(f64, (&poly / i).evaluate(i), (4.0*i.powi(2) + 6.0*i + 5.0)/i, epsilon=0.0001));
  }

  for i in -1000..1000 {
    let i = i as f64 * 0.001;
    if approx_eq!(f64, i, 0.0, epsilon=0.001) {
      continue;
    }
    poly = poly / i;
    assert!(approx_eq!(f64, poly.evaluate(i), (4.0*i.powi(2) + 6.0*i + 5.0)/i, epsilon=0.0001));
    poly.set_coefficient(0, 5.0);
    poly.set_coefficient(1, 6.0);
    poly.set_coefficient(2, 4.0);
  }

  for i in -1000..1000 {
    let i = i as f64 * 0.001;
    if approx_eq!(f64, i, 0.0, epsilon=0.001) {
      continue;
    }
    poly /= i;
    assert!(approx_eq!(f64, poly.evaluate(i), (4.0*i.powi(2) + 6.0*i + 5.0)/i, epsilon=0.0001));
    poly.set_coefficient(0, 5.0);
    poly.set_coefficient(1, 6.0);
    poly.set_coefficient(2, 4.0);
  }
}
