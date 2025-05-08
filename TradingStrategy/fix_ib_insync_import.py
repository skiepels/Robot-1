# This script fixes the ib_insync import issue
import os

def fix_ib_connector():
    # Path to ib_connector.py
    file_path = os.path.join('src', 'data', 'ib_connector.py')
    
    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}")
        return False
    
    # Read the file content
    with open(file_path, 'r') as file:
        content = file.readlines()
    
    # Find and modify the import line
    fixed_content = []
    for line in content:
        if "from ib_insync import IB, Contract, Stock, Forex, Index, Bars, util" in line:
            # Replace with corrected import (remove Bars as it might not exist in the current version)
            fixed_content.append("from ib_insync import IB, Contract, Stock, Forex, Index, util\n")
            print("Fixed ib_insync import by removing 'Bars'")
        else:
            fixed_content.append(line)
    
    # Write the fixed content back to the file
    with open(file_path, 'w') as file:
        file.writelines(fixed_content)
    
    print(f"Fixed import in {file_path}")
    
    # Also check if IB package is installed
    try:
        import ib_insync
        print(f"ib_insync version: {ib_insync.__version__}")
    except ImportError:
        print("ib_insync is not installed. Installing...")
        import subprocess
        subprocess.call(["pip", "install", "ib_insync"])
        
    return True

def install_requirements():
    """Install required packages from the project"""
    import subprocess
    try:
        print("Installing required packages...")
        subprocess.call(["pip", "install", "-e", "."])
        print("Requirements installed successfully")
        return True
    except Exception as e:
        print(f"Error installing requirements: {e}")
        return False

if __name__ == "__main__":
    if fix_ib_connector():
        print("\nNow trying to install project requirements...")
        install_requirements()
        print("\nNow you can run the backtest with this command:")
        print("python -m backtest.backtest_engine --symbols GPUS --start-date 2025-05-02 --end-date 2025-05-07 --capital 10000")