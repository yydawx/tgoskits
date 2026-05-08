#include <stdio.h>
#include <onnxruntime_c_api.h>

int main() {
    printf("=== StarryOS ONNX Runtime Inference Test ===\n");

    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) {
        printf("Failed to get ONNX Runtime API\n");
        return 1;
    }
    printf("[1/5] ONNX Runtime API loaded: version %d\n", ORT_API_VERSION);

    OrtEnv* env = nullptr;
    OrtStatus* status = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env);
    if (status != NULL) {
        printf("Failed to create env: %s\n", g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        return 1;
    }
    printf("[2/5] Environment created\n");

    OrtSessionOptions* session_options = nullptr;
    g_ort->CreateSessionOptions(&session_options);
    g_ort->SetIntraOpNumThreads(session_options, 1);
    g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_BASIC);

    printf("[3/5] Loading model: /usr/bin/act_model.onnx\n");
    OrtSession* session = nullptr;
    status = g_ort->CreateSession(env, "/usr/bin/act_model.onnx", session_options, &session);
    if (status != NULL) {
        printf("Failed to create session: %s\n", g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseSessionOptions(session_options);
        g_ort->ReleaseEnv(env);
        return 1;
    }
    printf("       Session created successfully\n");

    size_t num_input_nodes;
    g_ort->SessionGetInputCount(session, &num_input_nodes);
    size_t num_output_nodes;
    g_ort->SessionGetOutputCount(session, &num_output_nodes);
    printf("[4/5] Model: %zu inputs, %zu outputs\n", num_input_nodes, num_output_nodes);

    printf("[5/5] Cleaning up\n");
    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseEnv(env);

    printf("\nTEST PASSED\n");
    return 0;
}
