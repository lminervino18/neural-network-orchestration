use ndarray::prelude::*;

struct Layer {
    weights: Array2<f32>,
    biases: Array1<f32>,
    activation: fn(ArrayView1<f32>) -> Array1<f32>,
    activation_prime: fn(ArrayView1<f32>) -> Array1<f32>,
    input: Array1<f32>,
    linear_output: Array1<f32>,
}

impl Layer {
    fn new(
        neurons: usize,
        activation: fn(ArrayView1<f32>) -> Array1<f32>,
        activation_prime: fn(ArrayView1<f32>) -> Array1<f32>,
    ) -> Self {
        Self {
            weights: Array2::zeros((neurons, neurons)),
            biases: Array1::zeros(neurons),
            activation,
            activation_prime,
            input: Array1::zeros(neurons),
            linear_output: Array1::zeros(neurons),
        }
    }

    fn forward(&mut self, input: Array1<f32>) -> Array1<f32> {
        // me guardo el z para el backprop
        self.linear_output = self.weights.dot(&input) + &self.biases;
        (self.activation)(self.linear_output.view())
    }

    /* def backward(self, output_gradient, learning_rate):
    weights_gradient = np.dot(output_gradient, self.input.T)
    input_gradient = np.dot(self.weights.T, output_gradient)
    self.weights -= learning_rate * weights_gradient
    self.bias -= learning_rate * output_gradient
    return input_gradient */

    /* fn main() {
        let a: Array1<f64> = array![1.0, 2.0, 3.0];
        let b: Array1<f64> = array![4.0, 5.0];

        // Reshape 'a' to a column vector (3x1)
        let a_col_vec = a.insert_axis(Axis(1));

        // Reshape 'b' to a row vector (1x2)
        let b_row_vec = b.insert_axis(Axis(0));

        // Perform element-wise multiplication, which acts as the outer product
        let outer_product: Array2<f64> = &a_col_vec * &b_row_vec;

        println!("Vector a: {:?}", a);
        println!("Vector b: {:?}", b);
        println!("Outer product of a and b:\n{:?}", outer_product);

        // Expected result:
        // [[ 4.0,  5.0],
        //  [ 8.0, 10.0],
        //  [12.0, 15.0]]
    } */

    fn backward(&mut self, grad_output: &Array1<f32>) -> Array1<f32> {
        let delta = grad_output * (self.activation_prime)(self.linear_output.view());
        /***
         * grad_output <- outer product a^(l-1) * delta^(l)
         ***/
        let grad_weights =
            &self.input.view().insert_axis(Axis(1)) * &delta.view().insert_axis(Axis(0));
        /* let grad_weights = delta
        .view()
        .insert_axis(Axis(1))
        .dot(self.input.view().insert_axis(Axis(0))); */

        /***
         * move in the opposite direction of gradients (the gradient of the error with
         * respect to the biases equals delta)
         ***/
        self.weights -= &grad_weights;
        self.biases -= &delta;
        self.weights.t().dot(&delta)
    }
}

fn main() {}
