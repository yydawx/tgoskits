#!/bin/bash
# Batch compile pruned ONNX models to cvimodel (BF16)
set -e
source sophgo-tpumilr/envsetup.sh 2>/dev/null || true

OUTDIR=pro57/prune/cvimodels
mkdir -p $OUTDIR

MODELS=(
    "act_baseline:act_model_det.onnx"
    "act_E3D4:pro57/prune/pruned_models/act_E3D4.onnx"
    "act_E2D4:pro57/prune/pruned_models/act_E2D4.onnx"
)

for entry in "${MODELS[@]}"; do
    NAME="${entry%%:*}"
    ONNX="${entry##*:}"
    MLIR="$OUTDIR/${NAME}.mlir"
    TPU_MLIR="$OUTDIR/${NAME}_tpu.mlir"
    CVIMODEL="$OUTDIR/${NAME}.cvimodel"

    echo "=== Compiling: $NAME ==="
    echo "  ONNX: $ONNX"

    # Step 1: ONNX → Top MLIR
    if [ ! -f "$MLIR" ]; then
        echo "  [1/3] model_transform ..."
        model_transform.py \
            --model_name "$NAME" \
            --model_def "$ONNX" \
            --input_shapes [[1,3,224,224],[1,2]] \
            --mlir "$MLIR"
        echo "  MLIR: $MLIR ($(wc -c < $MLIR) bytes)"
    else
        echo "  [1/3] MLIR exists, skip"
    fi

    # Step 2: Top MLIR → TPU MLIR (BF16)
    if [ ! -f "$TPU_MLIR" ]; then
        echo "  [2/3] tpuc-opt --convert-top-to-tpu --tpu_bf16 ..."
        tpuc-opt --convert-top-to-tpu --tpu_bf16 "$MLIR" -o "$TPU_MLIR"
        echo "  TPU MLIR: $TPU_MLIR ($(wc -c < $TPU_MLIR) bytes)"
    else
        echo "  [2/3] TPU MLIR exists, skip"
    fi

    # Step 3: TPU MLIR → cvimodel
    if [ ! -f "$CVIMODEL" ]; then
        echo "  [3/3] model_deploy ..."
        model_deploy.py \
            --mlir "$TPU_MLIR" \
            --quantize BF16 \
            --chip cv181x \
            --model "$CVIMODEL"
        echo "  cvimodel: $CVIMODEL ($(ls -lh $CVIMODEL | awk '{print $5}'))"
    else
        echo "  [3/3] cvimodel exists, skip"
    fi

    echo ""
done

echo "All models compiled:"
ls -lh $OUTDIR/*.cvimodel
