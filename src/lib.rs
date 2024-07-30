use std::collections::HashMap;

use pyo3::prelude::*;
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

impl FromPyObject<'_> for  AnnotatedSampleSet {
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

#[pyclass]
struct Perceptron {
    #[pyo3(get, set)]
    index: i32,
    #[pyo3(get, set)]
    training_data: AnnotatedSampleSet,
    #[pyo3(get, set)]
    testing_data: AnnotatedSampleSet,
}

#[pymethods]
impl Perceptron {
    #[new]
    #[pyo3(signature = (
        train_data=HashMap::from([
            ('a', vec![]),
            ('b', vec![]),
        ]),
        test_data=HashMap::from([
            ('a', vec![]),
            ('b', vec![]),
        ])
    ))]

    fn new(train_data: HashMap<char, SampleSet>, test_data: HashMap<char, SampleSet>) -> Self {
        Perceptron { 
            index: 1,
            training_data: train_data.into(),
            testing_data: test_data.into()
        }
    }

    /// Adds samples which will be used for training the model.
    /// The samples should be provided as a list of vectors.
    /// 
    /// # Example:
    /// p = Perceptron()
    /// p.add_training_samples([[1,2], [3,5]], [-1, 4], [-7, 9]])
    fn add_training_samples(&mut self, a: SampleSet, b: SampleSet) {
        self.training_data.a.extend(a);
        self.training_data.b.extend(b);
    }

    /// Adds samples which can later be used for testing.
    /// The samples should be provided as a list of vectors.
    /// 
    /// # Example:
    /// p = Perceptron()
    /// p.add_testing_samples([[1,2], [3,5]], [-1, 4], [-7, 9]])
    #[pyo3(signature = (a, b=vec![]))]
    fn add_testing_samples(&mut self, a: SampleSet, b: SampleSet) {
        self.testing_data.a.extend(a);
        self.testing_data.b.extend(b);
    }

    fn train(&self) {
        println!("TODO: Implement training here.")
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn perceptron(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Perceptron>()?;

    Ok(())
}
