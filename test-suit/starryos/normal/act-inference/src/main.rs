use std::time::Instant;
use tract_onnx::prelude::*;

fn main() {
    println!("=== StarryOS Real ACT Model Inference ===");

    let model_path = "/usr/bin/act_model.onnx";
    let meta = std::fs::metadata(model_path).expect("model file not found");
    println!("[1/3] Model file: {} ({} bytes)", model_path, meta.len());

    let t0 = Instant::now();
    let model = tract_onnx::onnx()
        .model_for_path(model_path)
        .expect("failed to load model");
    println!("[2/3] Loaded: {} nodes, {:.1}s", model.nodes.len(), t0.elapsed().as_secs_f64());

    let t1 = Instant::now();
    let optimized = model.into_optimized().expect("failed to optimize");
    println!("       Optimized: {} nodes, {:.1}s", optimized.nodes.len(), t1.elapsed().as_secs_f64());

    let t2 = Instant::now();
    let runnable = optimized.into_runnable().expect("failed to make runnable");
    println!("       Runnable: {:.1}s", t2.elapsed().as_secs_f64());

    println!("[3/3] Inference (image=1x3x224x224 zeros, state=1x2 zeros):");

    let img = tract_ndarray::Array4::<f32>::zeros((1, 3, 224, 224)).into_tensor();
    let st = tract_ndarray::Array2::<f32>::zeros((1, 2)).into_tensor();

    let t_inf = Instant::now();
    let result = runnable.run(tvec![img.into(), st.into()]).expect("inference failed");
    let dt = t_inf.elapsed();

    let output: Vec<f32> = result[0]
        .to_array_view::<f32>()
        .unwrap()
        .as_slice()
        .unwrap()
        .to_vec();

    println!("       Output ({} values):", output.len());
    for i in 0..8 {
        println!("         step {}: [{:.6}, {:.6}]", i, output[i * 2], output[i * 2 + 1]);
    }
    println!("       Inference time: {:.1}ms", dt.as_secs_f64() * 1000.0);

    println!("\nTEST PASSED");
}
