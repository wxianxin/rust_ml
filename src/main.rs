use ndarray::{Array, Array2, Axis};
use ndarray_rand::{rand_distr::StandardNormal, RandomExt};
use std::time::Instant;


const FEATURE_SIZE: usize = 30;
const H_SIZE: usize = 64;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn lstm(
    input: &Array2<f64>,
    weight_ih: &Array2<f64>,
    weight_hh: &Array2<f64>,
    bias_ih: &Array2<f64>,
    bias_hh: &Array2<f64>,
    h: &Array2<f64>,
    c: &Array2<f64>,
) -> (Array2<f64>, Array2<f64>) {
    let gates = input.dot(&weight_ih.t().to_owned())
        + bias_ih
        + h.dot(&weight_hh.t().to_owned())
        + bias_hh;
    // let (i, rest) = gates.view().split_at(Axis(1), H_SIZE);
    // let (f, rest) = rest.view().split_at(Axis(1), H_SIZE);
    // let (c_prime, o) = rest.view().split_at(Axis(1), H_SIZE);

    // let i = i.mapv(sigmoid);
    // let f = f.mapv(sigmoid);
    // let c_prime = c_prime.mapv(|x| x.tanh());
    // let o = o.mapv(sigmoid);
    let (ifo, c_prime) = gates.view().split_at(Axis(1), H_SIZE * 3);
    let ifo = ifo.mapv(sigmoid);
    let c_prime = c_prime.mapv(|x| x.tanh());
    let (i, fo) = ifo.view().split_at(Axis(1), H_SIZE);
    let (f, o) = fo.view().split_at(Axis(1), H_SIZE);

   //    let c = f * c.view() + i * c_prime.view();
   //    let h = o * c.mapv(|x| x.tanh());
    let c = f.to_owned() * c + i.to_owned() * c_prime;
    let h = o.to_owned() * c.mapv(|x| x.tanh());

    (h, c)
}

fn infer(x: Array2<f64>, param_dict: std::collections::HashMap<String, Array2<f64>>) {
    let mut h: Array2<f64> = Array::zeros((1, H_SIZE));
    let mut c: Array2<f64> = Array::zeros((1, H_SIZE));

    for _ in 0..100 {
        let (new_h, new_c) = lstm(
            &x,
            &param_dict["lstm1.weight_ih"],
            &param_dict["lstm1.weight_hh"],
            &param_dict["lstm1.bias_ih"],
            &param_dict["lstm1.bias_hh"],
            &h,
            &c,
        );
        h = new_h;
        c = new_c;
    }
}

fn main() {
    let mut param_dict = std::collections::HashMap::new();

    let x: Array2<f64> = Array::random((1, FEATURE_SIZE), StandardNormal);
    param_dict.insert(
        String::from("lstm1.weight_ih"),
        Array::random((H_SIZE * 4, FEATURE_SIZE), StandardNormal),
    );
    param_dict.insert(
        String::from("lstm1.weight_hh"),
        Array::random((H_SIZE * 4, H_SIZE), StandardNormal),
    );
    param_dict.insert(
        String::from("lstm2.weight_ih"),
        Array::random((H_SIZE * 4, H_SIZE), StandardNormal),
    );
    param_dict.insert(
        String::from("lstm2.weight_hh"),
        Array::random((H_SIZE * 4, H_SIZE), StandardNormal),
    );
    param_dict.insert(
        String::from("fc.weight"),
        Array::random((FEATURE_SIZE, H_SIZE), StandardNormal),
    );
    param_dict.insert(
        String::from("lstm1.bias_ih"),
        Array::random((H_SIZE * 4, 1), StandardNormal),
    );
    param_dict.insert(
        String::from("lstm1.bias_hh"),
        Array::random((H_SIZE * 4, 1), StandardNormal),
    );
    param_dict.insert(
        String::from("lstm2.bias_ih"),
        Array::random((H_SIZE * 4, 1), StandardNormal),
    );
    param_dict.insert(
        String::from("lstm2.bias_hh"),
        Array::random((H_SIZE * 4, 1), StandardNormal),
    );
    param_dict.insert(
        String::from("fc.bias"),
        Array::random((FEATURE_SIZE, 1), StandardNormal),
    );

    let start = Instant::now();
    infer(x, param_dict);
    println!("Elapsed: {:.2?}", start.elapsed());
}
