use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[derive(Debug, Clone)]
struct Sample {
    data: Vec<f64>,
    label: i8, // Note: Only 1 and -1 are allowed.
}

impl IntoPy<PyObject> for Sample {
    fn into_py(self, py: Python<'_>) -> PyObject {
        pyo3::types::PyTuple::new_bound(py, &[self.data.into_py(py), self.label.into_py(py)])
            .into_py(py)
    }
}

// Custom implementation since we take in a tuple, instead of an object.
impl FromPyObject<'_> for Sample {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let tuple = ob.downcast::<pyo3::types::PyTuple>()?;
        if tuple.len() != 2 {
            let quantifier = if tuple.len() < 2 { "few" } else { "many" };
            return Err(PyValueError::new_err(format!(
                "Tuple contains too {quantifier} elements. Expected two, got {}.",
                tuple.len()
            )));
        }

        let label: i8 = tuple.get_item(1)?.extract()?;
        let data: Vec<f64> = tuple.get_item(0)?.extract()?;
        if label != 1 && label != -1 {
            return Err(PyValueError::new_err(format!(
                "The only labels allowed are +1 and -1. Got {label}."
            )));
        }

        Ok(Self { data, label })
    }
}

#[derive(Debug, Clone, FromPyObject)]
struct SampleSet(Vec<Sample>);

impl SampleSet {
    fn extend(&mut self, other: SampleSet) {
        self.0.extend(other.0);
    }

    fn clear(&mut self) {
        self.0.clear()
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn is_dimension(&self, dimension: usize) -> bool {
        self.0.iter().all(|inner| inner.data.len() != dimension)
    }
}

impl IntoPy<PyObject> for SampleSet {
    fn into_py(self, py: Python<'_>) -> PyObject {
        self.0
            .into_iter()
            .map(|sample| sample.into_py(py))
            .collect::<Vec<PyObject>>()
            .into_py(py)
    }
}

#[derive(Debug, PartialEq)]
enum PerceptronState {
    Setup,
    Trained,
}

impl ToPyObject for PerceptronState {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        match self {
            Self::Setup => "Setup".into_py(py),
            Self::Trained => "Trained".into_py(py),
        }
    }
}

#[pyclass]
pub struct Perceptron {
    #[pyo3(get)]
    state: PerceptronState,

    #[pyo3(get)]
    dimensions: usize,

    #[pyo3(get)]
    training_data: SampleSet,

    #[pyo3(get)]
    model: Vec<f64>,
}

impl Perceptron {}

#[pymethods]
impl Perceptron {
    /// Constructor for our Perceptron object.
    ///
    /// # Example:
    /// p = Perceptron(2)
    /// p.add_training_samples([([1,2], 1), ([3,5], 1), ([-1, 4], -1), ([-7, 9], -1)])
    /// model = p.train(10)
    #[new]
    #[pyo3(signature =( dimensions, training_data = SampleSet(vec![]) ))]
    fn new(dimensions: usize, training_data: SampleSet) -> Self {
        Perceptron {
            dimensions,
            training_data,
            model: vec![0f64; dimensions],
            state: PerceptronState::Setup,
        }
    }

    /// Adds samples which will be used for training the model.
    /// The samples should be provided as a list of vectors.
    ///
    /// # Example:
    /// p = Perceptron(2)
    /// p.add_training_samples([([1,2], 1), ([3,5], 1), ([-1, 4], -1), ([-7, 9], -1)])
    fn add_training_samples(&mut self, samples: SampleSet) -> PyResult<()> {
        if self.state != PerceptronState::Setup {
            return Err(PyValueError::new_err(
                "Cannot add training samples after training has started.",
            ));
        }

        if samples.is_dimension(self.dimensions) {
            return Err(PyValueError::new_err(format!(
                "Training samples do not match the dimensions required for this Perceptron.\nProvide the correct length ({}d), or create a new Perceptron instance.", self.dimensions
            )));
        }

        self.training_data.extend(samples);

        Ok(())
    }

    /// Adds samples which will be used for training the model.
    /// The samples should be provided as a list of vectors.
    ///
    /// # Example:
    /// p = Perceptron(2)
    /// p.replace_training_samples([([1,2], 1), ([3,5], 1), ([-1, 4], -1), ([-7, 9], -1)])
    fn replace_training_samples(&mut self, samples: SampleSet) -> PyResult<()> {
        self.clear_training_samples();
        self.add_training_samples(samples)
    }

    /// Clear all existing training data.
    fn clear_training_samples(&mut self) {
        self.training_data.clear()
    }

    /// Start training our model.
    ///
    /// # Example:
    /// p = Perceptron(2)
    /// p.add_training_samples([([1,2], 1), ([3,5], 1), ([-1, 4], -1), ([-7, 9], -1)])
    /// model = p.train(10)
    fn train(&mut self, iterations: u32) -> Vec<f64> {
        assert!(
            !self.training_data.is_empty(),
            "Training dataset is empty. Cannot train on an empty set."
        );

        let mut count = 0;
        let mut theta = vec![0f64; self.dimensions]; // Our offset goes on the end.

        for _ in 0..iterations {
            let mut done = true;
            count += 1;

            for sample in self.training_data.0.iter() {
                let distance = signed_distance(&sample.data, &self.model);
                let margin = distance * sample.label as f64;

                if margin <= 0.0 || margin.is_nan() {
                    // Incorrectly classified
                    done = false;

                    // Add point to current model.
                    for (a, b) in theta.iter_mut().zip(&sample.data) {
                        *a += b * sample.label as f64;
                    }
                }

                // Add the current theta to the averaged theta:
                for (average, new_digit) in self.model.iter_mut().zip(&theta) {
                    *average += (*new_digit - *average) / (count + 1) as f64;
                }
            }

            // We can be done early if all points are classified correctly.
            if done {
                break;
            }
        }
        self.state = PerceptronState::Trained;
        normalize(&mut self.model);

        self.model.clone()
    }
}

fn signed_distance(point: &[f64], hyperplane: &[f64]) -> f64 {
    let normal = vector_length(hyperplane);
    let product = dot_product(point, hyperplane);

    product / normal
}

fn dot_product(vec1: &[f64], vec2: &[f64]) -> f64 {
    vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum()
}

fn vector_length(vector: &[f64]) -> f64 {
    vector.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn normalize(vector: &mut [f64]) {
    let length = vector_length(vector);
    for value in vector {
        *value /= length;
    }
}

/// Values exported to the Python module.
#[pymodule]
fn perceptron(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Perceptron>()?;

    Ok(())
}
