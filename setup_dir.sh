# mkdirs
mkdir exps
mkdir preps
mkdir configs
mkdir notebooks
mkdir tools
mkdir tools/utils
mkdir tools/models
mkdir tools/features
mkdir mnt
# mkdir mnt/figs
# mkdir mnt/importances
# mkdir mnt/logs
# mkdir mnt/oofs
# mkdir mnt/submissions
# mkdir mnt/trained_models
# mkdir mnt/inputs
# mkdir mnt/inputs/origin
# mkdir mnt/inputs/

# set .gitignore
if [ -e ".gitignore" ]; then
    echo ".gitignore already exists!"
else
    touch .gitignore
    echo "tags*" >> .gitignore
    echo "mnt/" >> .gitignore
fi
