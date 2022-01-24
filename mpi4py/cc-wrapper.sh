#! /bin/bash

# perlmutter cc is a wrapper around nvcc which does not like some args

echo "got $@"

args=("$@")
for ((i=0; i<"${#args[@]}"; ++i)); do
    case ${args[i]} in
        '-march=x86-64') unset args[i];;
        '-fwrapv') unset args[i];;
        '-Wno-unused-result') unset args[i];;
    esac
done
echo cc "${args[@]}"
cc "${args[@]}"
