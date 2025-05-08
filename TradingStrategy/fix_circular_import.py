# This script fixes the circular import in backtest_engine.py
import os

def fix_circular_import():
    # Path to backtest_engine.py
    file_path = os.path.join('backtest', 'backtest_engine.py')
    
    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}")
        return False
    
    # Read the file content
    with open(file_path, 'r') as file:
        content = file.readlines()
    
    # Find and remove the problematic import
    fixed_content = []
    skip_line = False
    
    for line in content:
        # Skip the line with the circular import
        if "from backtest.backtest_engine import main" in line:
            skip_line = True
            print(f"Removed circular import: {line.strip()}")
            continue
        
        # If we've removed the sys.path.insert line, we can add it back but with a proper comment
        if "sys.path.insert" in line and "os.path.abspath" in line:
            fixed_content.append("# Add project root to path\n")
            fixed_content.append(line)
            continue
            
        fixed_content.append(line)
    
    # Write the fixed content back to the file
    with open(file_path, 'w') as file:
        file.writelines(fixed_content)
    
    print(f"Fixed circular import in {file_path}")
    return True

if __name__ == "__main__":
    if fix_circular_import():
        print("Now you can run the backtest with this command:")
        print("python -m backtest.backtest_engine --symbols GPUS --start-date 2025-05-02 --end-date 2025-05-07 --capital 10000")