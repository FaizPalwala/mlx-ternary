import os

def generate_lut():
    lut = []
    for val in range(256):
        if val >= 243:
            # Padding/invalid values map to 0
            lut.append([0.0, 0.0, 0.0, 0.0, 0.0])
            continue
            
        w = []
        temp = val
        for _ in range(5):
            rem = temp % 3
            if rem == 0:
                w.append(0.0)
            elif rem == 1:
                w.append(1.0)
            elif rem == 2:
                w.append(-1.0)
            temp //= 3
        lut.append(w)
        
    code = "constant float16_t base3_lut[256][5] = {\n"
    for i, w in enumerate(lut):
        row = f"    {{ {w[0]:.1f}h, {w[1]:.1f}h, {w[2]:.1f}h, {w[3]:.1f}h, {w[4]:.1f}h }}"
        if i < 255:
            row += ","
        row += f" // {i}\n"
        code += row
    code += "};\n"
    
    os.makedirs("src/mlx_ternary/cpp", exist_ok=True)
    with open("src/mlx_ternary/cpp/base3_lut.h", "w") as f:
        f.write(code)

if __name__ == "__main__":
    generate_lut()
