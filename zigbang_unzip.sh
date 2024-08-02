source ~/.bashrc

source="../../data/testset_sample_split_rest/23"
target="../../data/testset_sample_split_rest/23_unzip"

mkdir "${target}"

for file in $source/*; do
        if [[ "$file" == *.zip ]]; then
            #echo "${file}"
            #echo "${target}"
            f="${file##*/}"
            #echo "${f%.*}"
            unzip "${file}" -d "${target}/${f%.*}"
        fi
done