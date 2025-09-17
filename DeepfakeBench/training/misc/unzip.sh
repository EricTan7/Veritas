

#!/bin/bash

# Iterate over all subdirectories in the current directory
for dir in */; do
  # Check if it is a directory
  if [[ -d "$dir" ]]; then
    echo "Entering directory: $dir"
    
    # Enter the subdirectory
    cd "$dir"
    
    # Iterate over all .tar.gz files in the subdirectory
    for zipfile in *.zip; do
        # Check if the file exists and is a .zip file
        if [[ -f "$zipfile" ]]; then
            echo "Extracting $zipfile ..."
            unzip "$zipfile" -d "${zipfile%.zip}"  # Extract to the current directory
        fi
    done
    
    # Go back to the parent directory
    cd ..
  fi
done

echo "Extraction completed in all subdirectories!"


#!/bin/bash

# Iterate over all subdirectories in the current directory
for dir in */; do
  # Check if it is a directory
  if [[ -d "$dir" ]]; then
    echo "Entering directory: $dir"
    
    # Enter the subdirectory
    cd "$dir"
    
    # Iterate over all .tar.gz files in the subdirectory
    for zipfile in *.tar.gz; do
        # Check if the file exists and is a .zip file
        if [[ -f "$zipfile" ]]; then
            echo "Extracting $zipfile ..."
            tar -zxvf "$zipfile"  # Extract to the current directory
        fi
    done
    
    # Go back to the parent directory
    cd ..
  fi
done

echo "Extraction completed in all subdirectories!"




#!/bin/bash

# Iterate over all .zip files in the current directory
for zipfile in *.zip; do
  # Check if the file exists and is a .zip file
  if [[ -f "$zipfile" ]]; then
    echo "Extracting $zipfile ..."
    unzip "$zipfile" -d "${zipfile%.zip}"  # Extract to the current directory
  fi
done

echo "Extraction completed!"


#!/bin/bash

# Iterate over all .zip files in the current directory
for zipfile in *.tar.gz; do
  # Check if the file exists and is a .zip file
  if [[ -f "$zipfile" ]]; then
    echo "Extracting $zipfile ..."
    tar -zxvf "$zipfile"  # Extract to the current directory
  fi
done

echo "Extraction completed!"



#!/bin/bash

# Iterate over all subdirectories in the current directory
for dir in */; do
  # Check if it is a directory
  if [[ -d "$dir" ]]; then
    echo "Entering directory: $dir"
    
    # Enter the subdirectory
    cd "$dir"
    
    # Iterate over all .zip files in the subdirectory
    for zipfile in *.zip; do
        # Check if the file exists and is a .zip file
        if [[ -f "$zipfile" ]]; then
            echo "Extracting $zipfile using 7z..."
            7z x "$zipfile"  # Extract to the current directory
        fi
    done
    
    # Go back to the parent directory
    cd ..
  fi
done

echo "Extraction completed in all subdirectories!"


