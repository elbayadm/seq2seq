#! /usr/bin/env bash
# Flag '-b' for submitting to the besteffort queue

BQ=''
while getopts 'btk' flag; do
    echo "Reading flag "$flag
    case "${flag}" in
        b) BQ='true' ;;
        *) error "Unexpected option ${flag}" ;;
    esac
done
echo besteffort $BQ
shift $((OPTIND-1))

JOB=$1

echo traininig $JOB
mkdir -p 'save/'$JOB

if [ $BQ ]; then
    echo "Submitting to besteffort queue"
    cmd="oarsub -l \"walltime=160:0:0\" -n $JOB \
           -t besteffort -t idempotent \
           -O  save/$JOB/stdout -E save/$JOB/stderr\
           'python nmt.py -c config/'$JOB'.yaml'"
    echo 'Running' $cmd
    eval $cmd
else
    echo "Submitting to default queue"
    cmd="oarsub -l \"walltime=160:0:0\" -n $JOB \
            -O  save/$JOB/stdout -E save/$JOB/stderr\
            'python nmt.py -c config/'$JOB'.yaml'"
    eval $cmd
fi

