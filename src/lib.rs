use std::{collections::HashMap, fmt::Debug};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

type SampleSet = Vec<Vec<f64>>;

#[derive(Clone)]
struct AnnotatedSampleSet {
    a: SampleSet,
    b: SampleSet,
}

impl From<HashMap<char, SampleSet>> for AnnotatedSampleSet {
    fn from(value: HashMap<char, SampleSet>) -> Self {
        assert!(value.contains_key(&'a'), "Missing 'a' key.");
        assert!(value.contains_key(&'b'), "Missing 'b' key.");

        Self {
            a: match value.get(&'a') {
                Some(val) => val.to_vec(),
                None => vec![],
            },
            b: match value.get(&'a') {
                Some(val) => val.to_vec(),
                None => vec![],
            }
        }
    }
}

impl FromPyObject<'_> for AnnotatedSampleSet {
    fn extract_bound(ob: &Bound<PyAny>) -> PyResult<Self> {
        Ok(Self{
            a: ob.get_item('a')?.extract()?,
            b: ob.get_item('b')?.extract()?,
        })
    }
}

impl IntoPy<PyObject> for AnnotatedSampleSet {
    fn into_py(self, py: Python<'_>) -> PyObject {
        println!("Converting into Python!");
        HashMap::from([
            ('a', self.a),
            ('b', self.b),
        ]).into_py(py)
    }
}

impl AnnotatedSampleSet {
    fn new() -> Self{
        Self { a: vec![], b: vec![] }
    }

    fn clear(&mut self) {
        self.a.clear();
        self.b.clear();
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
struct Perceptron {
    #[pyo3(get)]
    state: PerceptronState,

    #[pyo3(get)]
    dimensions: usize,

    #[pyo3(get)]
    training_data: AnnotatedSampleSet,
    
    #[pyo3(get)]
    testing_data: AnnotatedSampleSet,

    #[pyo3(get)]
    model: Vec<f64>,
}

impl Perceptron {}

#[pymethods]
impl Perceptron {
    #[new]
    fn new(dimensions: usize) -> Self {
        Perceptron {
            dimensions,
            training_data: AnnotatedSampleSet::new(),
            testing_data: AnnotatedSampleSet::new(),
            model: vec![],
            state: PerceptronState::Setup,
        }
    }

    /// Adds samples which will be used for training the model.
    /// The samples should be provided as a list of vectors.
    /// 
    /// # Example:
    /// p = Perceptron()
    /// p.add_training_samples([[1,2], [3,5]], [-1, 4], [-7, 9]])
    fn add_training_samples(&mut self, a: SampleSet, b: SampleSet) -> PyResult<()> {
        if self.state != PerceptronState::Setup{ return Err(PyValueError::new_err("Cannot add training samples after training has started.")); }
        
        if !a.iter().all(|inner| {println!("Item of {} length.", inner.len()); inner.len() == self.dimensions})
        || !b.iter().all(|inner| inner.len() == self.dimensions) {
            return Err( PyValueError::new_err(format!("Training samples do not match the dimensions required for this Perceptron.\nProvide the correct length ({}d), or create a new Perceptron instance.", self.dimensions)));
        }
        
        self.training_data.a.extend(a);
        self.training_data.b.extend(b);

        Ok(())
    }

    /// Clear all existing training data.
    fn clear_training_samples(&mut self) {
        self.training_data.clear()
    }

    /// Adds samples which can later be used for testing.
    /// The samples should be provided as a list of vectors.
    /// 
    /// # Example:
    /// p = Perceptron()
    /// p.add_testing_samples([[1,2], [3,5]], [-1, 4], [-7, 9]])
    #[pyo3(signature = (a, b=vec![]))]
    fn add_testing_samples(&mut self, a: SampleSet, b: SampleSet) -> PyResult<()> {
        if !a.iter().all(|inner| inner.len() == self.dimensions)
        || !b.iter().all(|inner| inner.len() == self.dimensions) {
            return Err( PyValueError::new_err(format!("Testing samples do not match the dimensions required for this Perceptron.\nProvide the correct length ({}d), or create a new Perceptron instance.", self.dimensions)));
        }
        
        self.testing_data.a.extend(a);
        self.testing_data.b.extend(b);

        Ok(())
    }

    /// Clear all existing testing data.
    fn clear_testing_samples(&mut self) {
        self.testing_data.clear()
    }

    fn train(&mut self, iterations: u32) {
        // assert!(self.has_consistent_length(), "Not all vectors in the training/testing data is the same length.");
        let mut done = true;
        assert!(!self.training_data.a.is_empty() || !self.training_data.b.is_empty(), "Training dataset is empty. Cannot train on an empty set.");
        let mut count = 0;
        let theta = vec![0f64; self.dimensions]; // Our offset goes on the end.
        let theta_average = vec![0f64; self.dimensions]; // Our offset goes on the end.

        // let mut theta: vec

        for iter_index in 0..iterations {

            for point in self.training_data.a.iter() {
                // Calculate (signed) distance
                // If distance >= 0, point is classified correctly
                let distance = -5.0;
                if distance < 0.0 {
                    // Incorrectly classified
                    done = false;
                    // Theta += point
                }

                // Theta_average += theta (averaged by count)
                
                count += 1;
            }
            if done { break; } // We can be done early if all points are classified correctly.
        }

        self.model = theta_average;

        println!();
        self.state = PerceptronState::Trained;
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn perceptron(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Perceptron>()?;

    Ok(())
}
