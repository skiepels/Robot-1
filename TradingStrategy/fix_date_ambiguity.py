# This script fixes the date ambiguity issue in backtest_engine.py
import os

def fix_date_ambiguity():
    # Path to backtest_engine.py
    file_path = os.path.join('backtest', 'backtest_engine.py')
    
    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}")
        return False
    
    # Read the file content
    with open(file_path, 'r') as file:
        content = file.readlines()
    
    # Find and modify the problematic code
    fixed_content = []
    for i, line in enumerate(content):
        # Look for the line creating the 'date' column
        if "df['date'] = dates" in line:
            # Change to create a date_column instead of date
            fixed_content.append("            df['date_column'] = dates\n")
            print(f"Changed line {i+1}: 'df['date'] = dates' to 'df['date_column'] = dates'")
        # Update the merge line to use date_column instead of date
        elif "df = pd.merge(df, daily[['day_change_pct']], left_on='date', right_index=True, how='left')" in line:
            fixed_content.append("            df = pd.merge(df, daily[['day_change_pct']], left_on='date_column', right_index=True, how='left')\n")
            print(f"Changed line {i+1}: merged using 'date_column' instead of 'date'")
        # Update the line dropping the date column
        elif "df.drop('date', axis=1, inplace=True)" in line:
            fixed_content.append("            df.drop('date_column', axis=1, inplace=True)\n")
            print(f"Changed line {i+1}: dropping 'date_column' instead of 'date'")
        else:
            fixed_content.append(line)
    
    # Write the fixed content back to the file
    with open(file_path, 'w') as file:
        file.writelines(fixed_content)
    
    print(f"Fixed date ambiguity issue in {file_path}")
    return True

if __name__ == "__main__":
    if fix_date_ambiguity():
        print("\nNow you can run the backtest with this command:")
        print("python -m backtest.backtest_engine --symbols GPUS --start-date 2025-05-02 --end-date 2025-05-07 --capital 10000")