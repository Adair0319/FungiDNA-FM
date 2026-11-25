cd ~/yy_projects/fungi_project/1kfg_datasets/genomes_unmasked

# 1) 统计 fasta.gz 文件数
file_count=$(find . -type f -name '*.fasta.gz' | wc -l)

# 2) 统计序列条数和总碱基数
read seq_count base_count < <(
  find . -type f -name '*.fasta.gz' -print0 |
  xargs -0 zcat |
  awk '
    /^>/ {seq++; next}
    {
      gsub(/[[:space:]]/, "", $0)
      bases += length($0)
    }
    END {print seq, bases}
  '
)

echo "FASTA文件数: $file_count"
echo "序列条数:   $seq_count"
echo "碱基总数:   $base_count"
