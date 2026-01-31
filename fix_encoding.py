import os

def fix_file(filepath):
    print(f"Fixing {filepath}...")
    content = ""
    try:
        # Try reading as utf-8 first
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            # Try reading as utf-16 (powershell default sometimes)
            with open(filepath, 'r', encoding='utf-16') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                # Fallback to ansi
                with open(filepath, 'r', encoding='mbcs') as f:
                    content = f.read()
            except Exception as e:
                print(f"Failed to read {filepath}: {e}")
                return

    # Write back as utf-8 (no BOM)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    start_dir = "."
    for root, dirs, files in os.walk(start_dir):
        if ".git" in root or ".idea" in root:
            continue
        for file in files:
            if file.endswith(".py") or file.endswith(".yaml"):
                fix_file(os.path.join(root, file))

if __name__ == "__main__":
    main()
