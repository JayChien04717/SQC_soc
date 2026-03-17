import os
import csv

def merge_csvs(folders, output_file):
    header_written = False
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = None
        
        for folder in folders:
            if not os.path.exists(folder):
                print(f"Warning: Folder {folder} does not exist. Skipping.")
                continue
                
            for filename in os.listdir(folder):
                if filename.endswith('.csv'):
                    filepath = os.path.join(folder, filename)
                    with open(filepath, 'r', newline='', encoding='utf-8') as infile:
                        reader = csv.reader(infile)
                        try:
                            header = next(reader)
                        except StopIteration:
                            continue # Empty file
                            
                        if not header_written:
                            writer = csv.writer(outfile)
                            writer.writerow(header)
                            header_written = True
                        
                        for row in reader:
                            if any(row): # Skip empty rows
                                writer.writerow(row)
    
    print(f"Successfully merged CSVs into {output_file}")

if __name__ == "__main__":
    base_dir = r"c:\Users\cluster\Desktop\SQC_soc-dev_mux"
    folders = [
        os.path.join(base_dir, "csvfolder"),
        os.path.join(base_dir, "csvfolder2")
    ]
    output_path = os.path.join(base_dir, "mic_temp.csv")
    
    merge_csvs(folders, output_path)
