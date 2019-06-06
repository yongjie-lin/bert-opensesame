expt="aux"
layers="0 1 2 3 4 5 6 7 8 9 10 11"
data_dir="data_${expt}"
expt_dir=""
mode="eval"
bs=32

bert_model="bbu"
best_bbu=""
python predict.py \
    --mode ${mode} \
    --bert_model ${bert_model} \
    --expt ${expt} \
    --batch_size ${bs} \
    --output_layers ${layers}\
    --data_path ${data_dir} \
    --expt_path ${expt_dir} \
    --load_checkpt ${expt_dir}/${best_bbu} \
    |& tee -a ${expt_dir}/${mode}_${bert_model}.log
