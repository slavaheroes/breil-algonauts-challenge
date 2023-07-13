max=8
for i in `seq 1 $max`
do
    echo "$i"
    python prediction.py --subj $i --side 'left'
    python prediction.py --subj $i --side 'right'
done

echo All done