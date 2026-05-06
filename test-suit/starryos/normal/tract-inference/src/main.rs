use tract_onnx::prelude::*;

fn run_inference(
    model: &RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    image: &[f32],
    state: &[f32],
    z: &[f32],
) -> Vec<f32> {
    let img = tract_ndarray::Array4::from_shape_vec((1, 3, 64, 64), image.to_vec())
        .unwrap()
        .into_tensor();
    let st = tract_ndarray::Array2::from_shape_vec((1, state.len()), state.to_vec())
        .unwrap()
        .into_tensor();
    let zv = tract_ndarray::Array2::from_shape_vec((1, z.len()), z.to_vec())
        .unwrap()
        .into_tensor();

    let result = model
        .run(tvec![img.into(), st.into(), zv.into()])
        .expect("inference failed");

    result[0]
        .to_array_view::<f32>()
        .unwrap()
        .as_slice()
        .unwrap()
        .to_vec()
}

fn check(name: &str, actual: &[f32], expected: &[f32]) -> bool {
    let max_err = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, e)| (a - e).abs())
        .fold(0.0f32, f32::max);
    let ok = max_err < 0.02;
    if ok {
        println!("{}: PASS (max_err={:.6})", name, max_err);
    } else {
        println!(
            "{}: FAIL (max_err={:.6}, first3 actual={:?}, expected={:?})",
            name,
            max_err,
            &actual[..3],
            &expected[..3]
        );
    }
    ok
}

fn main() {
    println!("tract inference starting");

    let model = tract_onnx::onnx()
        .model_for_path("/usr/bin/simple.onnx")
        .expect("failed to load model")
        .into_optimized()
        .expect("failed to optimize")
        .into_runnable()
        .expect("failed to make runnable");

    println!("model loaded and optimized");

    // Reference outputs from tract-onnx on x86_64 host
    // zeros input
    let zeros_img = vec![0.0f32; 1 * 3 * 64 * 64];
    let zeros_st = vec![0.0f32; 14];
    let zeros_z = vec![0.0f32; 8];
    let ref_zeros: Vec<f32> = vec![
        -0.109215, 0.043383, -0.203812, 0.316391, 0.515393, -0.145230, 0.128388,
        0.335425, -0.032624, 0.587150, -0.014915, 0.176458, -0.109648, 0.056797,
    ];

    // ones input
    let ones_img = vec![1.0f32; 1 * 3 * 64 * 64];
    let ones_st = vec![1.0f32; 14];
    let ones_z = vec![1.0f32; 8];
    let ref_ones: Vec<f32> = vec![
        0.233426, 0.318312, -0.257784, 0.244134, 0.275732, 0.334169, -0.531516,
        -0.075019, 0.171543, 0.571759, 0.106811, 0.008956, 0.129034, 0.031104,
    ];

    let mut pass = true;
    pass &= check("zeros", &run_inference(&model, &zeros_img, &zeros_st, &zeros_z), &ref_zeros);
    pass &= check("ones", &run_inference(&model, &ones_img, &ones_st, &ones_z), &ref_ones);

    // Sanity: zeros and ones should produce different outputs
    let out_zeros = run_inference(&model, &zeros_img, &zeros_st, &zeros_z);
    let out_ones = run_inference(&model, &ones_img, &ones_st, &ones_z);
    let diff = out_zeros
        .iter()
        .zip(out_ones.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    if diff > 0.1 {
        println!("input_sensitivity: PASS (diff={:.4})", diff);
    } else {
        println!("input_sensitivity: FAIL (diff={:.4}, outputs too similar)", diff);
        pass = false;
    }

    if pass {
        println!("TEST PASSED");
    } else {
        println!("TEST FAILED");
    }
}
