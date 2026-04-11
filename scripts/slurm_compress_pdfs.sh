#!/bin/bash
#SBATCH --job-name=compress_pdfs
#SBATCH --partition=rush
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16g
#SBATCH --time=01:00:00
#SBATCH --output=/home/zz865/pythonProject/autoresearch/analysis/iclr2025_baseline/compress_%j.out
#SBATCH --error=/home/zz865/pythonProject/autoresearch/analysis/iclr2025_baseline/compress_%j.err

cd /home/zz865/pythonProject/autoresearch
echo "Node: $(hostname), CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"

# Compress all PDFs > 10MB in parallel using GNU parallel / xargs
find analysis/iclr2025_baseline/pdfs -name "*.pdf" -size +10M | \
  xargs -P $SLURM_CPUS_PER_TASK -I{} bash -c '
    f="{}"
    orig_size=$(du -m "$f" | cut -f1)
    tmp="${f%.pdf}.tmp.pdf"
    gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/ebook \
       -dNOPAUSE -dBATCH -dQUIET \
       -dColorImageResolution=150 -dGrayImageResolution=150 -dMonoImageResolution=300 \
       -sOutputFile="$tmp" "$f" 2>/dev/null
    if [ -f "$tmp" ]; then
      new_size=$(du -m "$tmp" | cut -f1)
      if [ "$new_size" -le 10 ]; then
        mv "$tmp" "$f"
        echo "OK: $(basename $f) ${orig_size}MB -> ${new_size}MB"
      else
        # Try aggressive
        gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/screen \
           -dNOPAUSE -dBATCH -dQUIET \
           -dColorImageResolution=72 -dGrayImageResolution=72 \
           -sOutputFile="$tmp" "$f" 2>/dev/null
        new_size=$(du -m "$tmp" | cut -f1)
        if [ "$new_size" -le 10 ]; then
          mv "$tmp" "$f"
          echo "OK (aggressive): $(basename $f) ${orig_size}MB -> ${new_size}MB"
        else
          rm -f "$tmp"
          echo "SKIP: $(basename $f) ${orig_size}MB -> ${new_size}MB still too big"
        fi
      fi
    else
      echo "FAIL: $(basename $f)"
    fi
  '

echo "Done: $(date)"
echo "Remaining > 10MB:"
find analysis/iclr2025_baseline/pdfs -name "*.pdf" -size +10M -exec du -m {} \;
