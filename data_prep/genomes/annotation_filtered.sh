# 切换到你的 annotation 根目录
cd ~/yy_projects/fungi_project/1kfg_datasets/annotation_filtered

# 遍历当前目录下的所有文件夹
for sp_dir in */; do
    # 确保只处理目录，忽略文件
    [ -d "${sp_dir}" ] || continue
    
    # 去除目录名末尾的斜杠，便于后续拼接路径 (例如将 "Aaoar1/" 变为 "Aaoar1")
    sp=${sp_dir%/}
    
    echo "正在处理: ${sp} ..."
    
    # 定义源路径和目标路径
    src_dir="${sp}/Annotation/Mycocosm/Annotation/Filtered_Models___best__"
    dest_dir="${sp}/Filtered_Models___best__"
    
    # 检查源文件夹是否存在
    if [ -d "${src_dir}" ]; then
        # 1. 将 Filtered_Models___best__ 移动到物种文件夹的第一层级
        mv "${src_dir}" "${dest_dir}"
        
        # 2. 删除原有的 Annotation 文件夹（这会连同里面的 All_models... 一起删掉）
        rm -rf "${sp}/Annotation"
        
        echo "  -> [成功] ${sp} 目录已重新整理。"
    elif [ -d "${dest_dir}" ]; then
        # 如果外层已经有了该文件夹，说明可能已经处理过
        echo "  -> [跳过] ${sp} 似乎已经整理完毕。"
    else
        # 如果既没有在外层也没有在内层找到，发出警告
        echo "  -> [警告] 在 ${sp} 中未找到 Filtered_Models___best__，请检查原始结构。"
    fi
done

echo "所有物种文件夹整理完毕！"

