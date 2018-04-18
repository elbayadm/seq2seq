#! /usr/bin/env bash
# Flag '-b' for submitting to the besteffort queue
# Flag '-k' to tolerate K40s
# Flag '-t' to request a titan_x

BQ=''
TX=''
K40=''
while getopts 'btk' flag; do
    echo "Reading flag "$flag
    case "${flag}" in
        b) BQ='true' ;;
        t) TX='true' ;;
        k) K40='true' ;;
        *) error "Unexpected option ${flag}" ;;
    esac
done
echo titanx:$TX besteffort $BQ k40 tolereance $K40
shift $((OPTIND-1))

JOB=$1
MEM=$2

echo traininig $JOB
mkdir -p 'save/'$JOB

if [ $TX ]; then
    oarprop="\"gpumodel='titan_x_pascal' or gpumodel='titan_x'\""
    echo $oarprop
else 
    if [ $K40 ]; then
        oarprop="\"gpumem>"$MEM"\""
    else
        oarprop="\"not gpumodel='k40m' and gpumem>"$MEM"\""
    fi
fi
echo "OAR requirements:" $oarprop 

if [ $BQ ]; then
    echo "Submitting to besteffort queue"
    cmd="oarsub -l \"walltime=160:0:0\" -n $JOB \
           -t besteffort -t idempotent \
           -p $oarprop \
           -O  save/$JOB/stdout -E save/$JOB/stderr\
           'source activate safe && python nmt.py -c config/'$JOB'.yaml'"
    echo 'Running' $cmd
    eval $cmd
else
    echo "Submitting to default queue"
    cmd="oarsub -l \"walltime=160:0:0\" -n $JOB \
            -O  save/$JOB/stdout -E save/$JOB/stderr\
            -p $oarprop\
            'source activate safe && python nmt.py -c config/'$JOB'.yaml'"
    eval $cmd
fi

