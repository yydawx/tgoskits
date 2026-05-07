use std::time::Instant;
use tract_onnx::prelude::*;

const IMG_SIZE: usize = 1 * 3 * 64 * 64; // 12288
const ST_SIZE: usize = 14;
const Z_SIZE: usize = 8;
const CASE_SIZE: usize = (IMG_SIZE + ST_SIZE + Z_SIZE) * 4; // 49240 bytes

static TEST_INPUTS: &[u8] = include_bytes!("test_inputs.bin");

fn load_f32s(data: &[u8], count: usize) -> Vec<f32> {
    data[..count * 4]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

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
    let ok = max_err < 0.001;
    if ok {
        println!("{}: PASS (max_err={:.8})", name, max_err);
    } else {
        println!(
            "{}: FAIL (max_err={:.8}, first3 actual={:?}, expected={:?})",
            name,
            max_err,
            &actual[..3],
            &expected[..3]
        );
    }
    ok
}

fn main() {
    println!("=== StarryOS ONNX Inference Cross-Validation ===");

    // 1. Verify model file exists and show size
    let model_path = "/usr/bin/simple.onnx";
    let model_meta = std::fs::metadata(model_path).expect("model file not found");
    println!("[1/4] Model file: {} ({} bytes)", model_path, model_meta.len());

    // 2. Load and optimize model — timed
    let t0 = Instant::now();
    let raw_model = tract_onnx::onnx()
        .model_for_path(model_path)
        .expect("failed to load model");
    let t_load = t0.elapsed();

    let node_count = raw_model.nodes.len();
    let input_count = raw_model.inputs.len();
    let output_count = raw_model.outputs.len();

    let t1 = Instant::now();
    let optimized = raw_model
        .into_optimized()
        .expect("failed to optimize");
    let t_opt = t1.elapsed();

    let opt_node_count = optimized.nodes.len();

    let t2 = Instant::now();
    let model = optimized
        .into_runnable()
        .expect("failed to make runnable");
    let t_runnable = t2.elapsed();

    println!(
        "[2/4] Model loaded: nodes={}, inputs={}, outputs={}",
        node_count, input_count, output_count
    );
    println!(
        "       Optimized: {} -> {} nodes ({:.1}ms load, {:.1}ms optimize, {:.1}ms runnable)",
        node_count,
        opt_node_count,
        t_load.as_secs_f64() * 1000.0,
        t_opt.as_secs_f64() * 1000.0,
        t_runnable.as_secs_f64() * 1000.0
    );

    let mut pass = true;

    // 3. Deterministic tests
    println!("[3/4] Deterministic tests:");

    let zeros_img = vec![0.0f32; IMG_SIZE];
    let zeros_st = vec![0.0f32; ST_SIZE];
    let zeros_z = vec![0.0f32; Z_SIZE];
    let ref_zeros: Vec<f32> = vec![
        -0.109215, 0.043383, -0.203812, 0.316391, 0.515393, -0.145230, 0.128388, 0.335425,
        -0.032624, 0.587150, -0.014915, 0.176458, -0.109648, 0.056797,
    ];

    let t_inf = Instant::now();
    let out_zeros = run_inference(&model, &zeros_img, &zeros_st, &zeros_z);
    let dt_zeros = t_inf.elapsed();
    pass &= check("zeros", &out_zeros, &ref_zeros);
    println!("       zeros inference: {:.1}ms", dt_zeros.as_secs_f64() * 1000.0);

    let ones_img = vec![1.0f32; IMG_SIZE];
    let ones_st = vec![1.0f32; ST_SIZE];
    let ones_z = vec![1.0f32; Z_SIZE];
    let ref_ones: Vec<f32> = vec![
        0.233426, 0.318312, -0.257784, 0.244134, 0.275732, 0.334169, -0.531516, -0.075019,
        0.171543, 0.571759, 0.106811, 0.008956, 0.129034, 0.031104,
    ];

    let t_inf = Instant::now();
    let out_ones = run_inference(&model, &ones_img, &ones_st, &ones_z);
    let dt_ones = t_inf.elapsed();
    pass &= check("ones", &out_ones, &ref_ones);
    println!("       ones  inference: {:.1}ms", dt_ones.as_secs_f64() * 1000.0);

    // Sensitivity
    let diff = out_zeros
        .iter()
        .zip(out_ones.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    if diff > 0.1 {
        println!("       sensitivity: PASS (diff={:.4})", diff);
    } else {
        println!("       sensitivity: FAIL (diff={:.4})", diff);
        pass = false;
    }

    // 4. Random cross-validation (10 cases)
    println!("[4/4] Random cross-validation (onnxruntime reference, seed=42):");

    let test_refs: [&[f32]; 10] = [
        &[
            0.00367326, -0.0153627, -0.114398, 0.336782, 0.19634, 0.275519, -0.579019,
            -0.175347, -0.167701, 0.74481, -0.243078, -0.0420059, 0.0902823, 0.228563,
        ],
        &[
            0.118605, 0.0430471, -0.089228, 0.243202, -0.0384407, 0.384782, -0.577861,
            0.0740427, -0.076888, 0.570403, -0.142815, 0.103263, 0.0743382, 0.0895536,
        ],
        &[
            0.0907632, 0.0153968, -0.0887958, 0.14705, 0.491619, -0.00892819, -0.201777,
            0.126157, -0.326588, 0.410625, -0.229705, 0.0126998, 0.135023, 0.19995,
        ],
        &[
            -0.000215247, 0.118395, -0.292333, 0.301229, 0.272215, 0.0936321, -0.145925,
            0.167353, -0.133194, 0.411986, -0.0200695, 0.0818884, -0.0407868, -0.0176758,
        ],
        &[
            0.177601, 0.218545, -0.160783, 0.294649, 0.604887, -0.0720627, -0.283713,
            0.0421668, 0.120536, 0.844466, 0.0361145, 0.157963, 0.0922555, 0.113665,
        ],
        &[
            0.119365, 0.260522, -0.184204, 0.317407, 0.490194, -0.248614, 0.152248, 0.370682,
            -0.123151, 0.269285, -0.0909297, 0.0800994, 0.087028, 0.133825,
        ],
        &[
            0.207075, 0.169025, -0.122342, 0.147505, 0.379753, -0.170778, 0.0893223, 0.307589,
            -0.287499, 0.209093, -0.27073, 0.0306133, 0.23553, 0.243054,
        ],
        &[
            0.0535414, 0.0570914, -0.156592, 0.0559606, 0.479057, -0.0811625, 0.000604421,
            0.408059, -0.281797, 0.143563, 0.0381973, 0.0736044, 0.0362682, 0.0621122,
        ],
        &[
            -0.0794414, 0.17228, -0.0704964, 0.214169, 0.34657, -0.131975, -0.0267224,
            0.350441, 0.26174, 0.176445, 0.00428378, -0.0172743, 0.0308232, -0.0559349,
        ],
        &[
            -0.212223, 0.0726351, -0.106473, 0.324146, 0.293305, 0.236462, -0.49473,
            -0.0727356, 0.084401, 0.596436, -0.153446, 0.00615631, -0.0293029, 0.0203267,
        ],
    ];

    let t_all = Instant::now();
    for i in 0..10 {
        let offset = i * CASE_SIZE;
        let img = load_f32s(&TEST_INPUTS[offset..], IMG_SIZE);
        let st = load_f32s(&TEST_INPUTS[offset + IMG_SIZE * 4..], ST_SIZE);
        let z = load_f32s(&TEST_INPUTS[offset + (IMG_SIZE + ST_SIZE) * 4..], Z_SIZE);
        let t1 = Instant::now();
        let out = run_inference(&model, &img, &st, &z);
        let dt = t1.elapsed();
        pass &= check(&format!("rand{}", i), &out, test_refs[i]);
        println!("        rand{}: {:.1}ms", i, dt.as_secs_f64() * 1000.0);
    }
    let dt_all = t_all.elapsed();
    println!(
        "       10 random inferences total: {:.1}ms",
        dt_all.as_secs_f64() * 1000.0
    );

    // Final verdict
    println!();
    if pass {
        println!(
            "RESULT: ALL 12 TESTS PASSED (model={} nodes, inference on StarryOS RISC-V)",
            node_count
        );
        println!("TEST PASSED");
    } else {
        println!("RESULT: SOME TESTS FAILED");
        println!("TEST FAILED");
    }
}
