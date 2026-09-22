import sys
import os

def main():
    if len(sys.argv) != 3:
        print("Usage: python metal_to_h.py <input.metal> <output.h>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    var_name = os.path.basename(input_file).replace('.', '_') + "_source"

    with open(input_file, 'r') as f:
        metal_code = f.read()

    with open(output_file, 'w') as f:
        f.write("#pragma once\n\n")
        f.write(f"const char* {var_name} = R\"METAL(\n")
        f.write(metal_code)
        f.write("\n)METAL\";\n")

if __name__ == "__main__":
    main()
