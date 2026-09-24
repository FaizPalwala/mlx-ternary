def calc_tok_sec(bandwidth_gb_s, params_b, bits_per_param):
    # Calculate bytes per parameter
    bytes_per_param = bits_per_param / 8.0
    # Calculate total memory footprint in GB
    footprint_gb = params_b * bytes_per_param
    # Calculate tokens per second
    tok_sec = bandwidth_gb_s / footprint_gb
    return footprint_gb, tok_sec

bw = 273  # M4 Pro memory bandwidth in GB/s

for params in [3, 7, 14, 27, 30]:
    fp_gb, tok = calc_tok_sec(bw, params, 16)
    bp_gb, btok = calc_tok_sec(bw, params, 2)
    print(f"{params}B Model: FP16 ({fp_gb:.2f}GB) -> {tok:.1f} tok/s | 1.58b ({bp_gb:.2f}GB) -> {btok:.1f} tok/s")
