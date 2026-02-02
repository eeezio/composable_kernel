# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#!/bin/bash

# Script to run FMHA forward and backward tests with varying s_k values
# Records output and kills process after first output line

OUTPUT_DIR="test_results"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BWD_OUTPUT="$OUTPUT_DIR/fmha_bwd_no_mask_${TIMESTAMP}.log"
FWD_OUTPUT="$OUTPUT_DIR/fmha_fwd_no_mask_${TIMESTAMP}.log"

echo "Starting FMHA tests..."
echo "Backward test results will be saved to: $BWD_OUTPUT"
echo "Forward test results will be saved to: $FWD_OUTPUT"
echo ""

# Clear output files
> "$BWD_OUTPUT"
> "$FWD_OUTPUT"

# Loop through s_k values from 4 to 16
for s_k in {4..16}; do
    echo "========================================"
    echo "Testing with s_k=$s_k"
    echo "========================================"
    
    # Test backward pass
    echo "Running FMHA backward (s_k=$s_k)..."
    echo "=== Test with s_k=$s_k ===" >> "$BWD_OUTPUT"
    
    timeout 20s ./build2/bin/tile_example_fmha_bwd \
        -b=30720 -h=32 -s=1 -s_k=$s_k -d=128 -mask=0 -p_drop=1 -prec=bf16 -v=0 \
        2>&1 | head -n 50 >> "$BWD_OUTPUT"
    
    echo "" >> "$BWD_OUTPUT"
    echo "Backward test completed for s_k=$s_k"
    
    # Test forward pass
    echo "Running FMHA forward (s_k=$s_k)..."
    echo "=== Test with s_k=$s_k ===" >> "$FWD_OUTPUT"
    
    timeout 20s ./build2/bin/tile_example_fmha_fwd \
        -b=30720 -h=32 -s=1 -s_k=$s_k -d=128 -mask=0 -p_drop=1 -prec=bf16 -v=0 \
        2>&1 | head -n 50 >> "$FWD_OUTPUT"
    
    echo "" >> "$FWD_OUTPUT"
    echo "Forward test completed for s_k=$s_k"
    echo ""
done

echo "========================================"
echo "All tests completed!"
echo "========================================"
echo "Results saved to:"
echo "  Backward: $BWD_OUTPUT"
echo "  Forward:  $FWD_OUTPUT"
