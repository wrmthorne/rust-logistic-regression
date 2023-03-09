use ndarray::{Array, Array2, Dimension, concatenate, Ix1, Ix2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use clap::Parser;

fn sigmoid<T: Dimension>(z: &Array<f64, T>) -> Array<f64, T> {
    return z.mapv(|z| 1. / (1. + (-z).exp()));
}

fn forward(x: &Array<f64, Ix2>, w: &Array<f64, Ix1>) -> Array<f64, Ix1> {
    return sigmoid(&(x.dot(w)));
}

fn logit_loss(y_hat: &Array<f64, Ix1>, y: &Array<f64, Ix1>) -> f64 {
    const EPSILON: f64 = 1e-13;

    let y_hat_ln = y_hat.mapv(|y_hat| (y_hat + EPSILON).ln());
    let y_hat_ln_neg = y_hat.mapv(|y_hat| (1. - y_hat + EPSILON).ln());
    let loss = y * &y_hat_ln + (1. - y) * &y_hat_ln_neg;
    
    if let Some(loss) = loss.mean() {
        return -loss;
    }

    return f64::NAN;
}

fn gradient_descent(x: &Array<f64, Ix2>, y: &Array<f64, Ix1>, w: &Array<f64, Ix1>, lr: f64) -> Array<f64, Ix1> {
    let mut w_opt = w.clone();

    for i in 0..1000 {
        let y_hat = forward(x, &w_opt);
        let grad = x.t().dot(&(y_hat.clone() - y));
        w_opt = w_opt - lr * grad;

        if i % 100 == 0 {
            let loss = logit_loss(&y_hat, y);
            println!("loss: {}", loss);
        }
    }
    
    return w_opt;
}

#[derive(Parser, Default, Debug)]
#[clap(author="Liam Thorne", version, about)]
/// A learning project in logistic regression using Rust and ndarray
struct Arguments {
    #[clap(long, default_value="1e-3", about)]
    /// Model learning rate
    lr: f64,
}

fn main() {
    let args = Arguments::parse();
    println!("{:?}", args);
    
    let num_samples = 5;
    let num_dims = 2;

    // Generate samples
    let y = Array::random(num_samples, Uniform::new(0., 1.));

    let mut x = Array2::random((num_samples, num_dims), Uniform::new(-5., 5.));
    let x_ones = Array::<f64, Ix2>::ones((x.shape()[0], 1));
    x = concatenate![Axis(1), x_ones, x];
    
    let w = Array::random(x.shape()[1], Uniform::new(0., 10.));

    let w_opt = gradient_descent(&x, &y, &w, args.lr);
    println!("w_opt: {:?}", w_opt);
}