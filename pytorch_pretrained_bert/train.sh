date=`date '+%Y%m%d-%H%M%S'`
expt="aux"
layers="0 1 2 3 4 5 6 7 8 9 10 11"
blu_layers="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23"
data_dir="data_${expt}"
expt_dir="experiments/${expt}_${date}"
mode="train"
bs=32
epochs=1
nth_n=9

mkdir -p ${expt_dir}

bert_model="blu"
python predict.py \
    --mode ${mode} \
    --bert_model ${bert_model} \
    --expt ${expt} \
    --batch_size ${bs} \
    --n_epochs ${epochs} \
    --output_layers ${blu_layers}\
    --eager_eval \
    --data_path ${data_dir} \
    --expt_path ${expt_dir} \
    |& tee -a ${expt_dir}/${mode}_${bert_model}.log
