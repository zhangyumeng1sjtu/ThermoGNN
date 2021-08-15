#!/bin/bash

for pdb_dir in `ls data/pdbs/demo/*`
do
    python ThermoGNN/tools/hhblits.py -i $pdb_dir \
                                      -db ../hhsuite_db/UniRef30_2020_06 \
                                      -o data/hhm/demo/ \
                                      --cpu 40
done
