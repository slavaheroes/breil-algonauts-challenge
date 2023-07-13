# we pass case to this bash script and it runs sequentially left and right sides
# usage: sh train_sh.sh -c 1 -g 0

while getopts c:g: flag
do
    case "${flag}" in
        c) case=${OPTARG};;
        g) gpu=${OPTARG}
    esac
done

python train.py --subj $case --side 'left' --gpu $gpu;
python train.py --subj $case --side 'right' --gpu $gpu;