from Bio import SeqIO

# 定义ncRNA家族
families = [
    "5S_rRNA", "5_8S_rRNA", "tRNA", "ribozyme", "CD-box", "miRNA",
    "Intron_gpI", "Intron_gpII", "HACA-box", "riboswitch", "IRES",
    "leader", "scaRNA"
]

# 初始化统计字典
family_base_counts = {family: {"A": 0, "C": 0, "G": 0, "T": 0, "total": 0} for family in families}

# 数据文件夹路径
data_folder = 'H:/Google download/MSADN/MSADN-main/nRC_Ten_Fold_Data/'

# 处理Train和Test文件夹中的所有文件
for set_type in ['Train', 'Test']:
    for i in range(10):
        file_path = f"{data_folder}{set_type}_{i}"
        try:
            with open(file_path, "r") as file:
                # 解析FASTA文件
                records = SeqIO.parse(file, "fasta")
                for record in records:
                    # 从记录的ID中提取家族名称
                    description = record.description
                    family = description.split()[-1]
                    if family in family_base_counts:
                        # 计算每个碱基的数量
                        sequence = str(record.seq).upper()
                        family_base_counts[family]["A"] += sequence.count('A')
                        family_base_counts[family]["C"] += sequence.count('C')
                        family_base_counts[family]["G"] += sequence.count('G')
                        family_base_counts[family]["T"] += sequence.count('T')
                        # 计算总碱基数
                        family_base_counts[family]["total"] += len(sequence)
        except FileNotFoundError:
            print(f"File {file_path} not found.")

# 输出结果
for family, counts in family_base_counts.items():
    total_bases = counts.pop("total")
    if total_bases > 0:
        percentages = {base: (count / total_bases) * 100 for base, count in counts.items()}
    else:
        percentages = {base: 0 for base in counts.keys()}  # 避免零除错误
    print(f"Family: {family}, Counts: {counts}, Percentages: {percentages}")
