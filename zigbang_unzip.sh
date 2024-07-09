source ~/.bashrc

source="../../data/aug_imgs_split/7"
target="../../data/aug_imgs_split/7_unzip"

for file in $source/*; do
        if [[ "$file" == *.zip ]]; then
            #echo "${file}"
            #echo "${target}"
            unzip "${file}" -d "${target}"                
        fi
done

