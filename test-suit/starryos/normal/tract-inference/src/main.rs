use tract_onnx::prelude::*;

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

    // Input: [[1.0, 2.0, 3.0]]
    let input = tract_ndarray::arr2(&[[1.0f32, 2.0, 3.0]]);
    let input = input.into_tensor();

    let result = model.run(tvec![input.into()]).expect("inference failed");

    let output = result[0].to_array_view::<f32>().expect("failed to view output");
    println!("output: {:?}", output);

    // Expected: [[14.5, 31.5]]
    let expected = [6.0f32, 6.0];
    let actual = output.as_slice().unwrap();
    let ok = actual.iter().zip(expected.iter()).all(|(a, e)| (a - e).abs() < 1e-4);
    if ok {
        println!("TEST PASSED");
    } else {
        println!("TEST FAILED: expected {:?}, got {:?}", expected, actual);
    }
}
