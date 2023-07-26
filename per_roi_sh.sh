max=8
for i in `seq 1 $max`
do
    echo "$i"
    python clip_sam_roi_wise_submission.py --subj $i
done

echo All done