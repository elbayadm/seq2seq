#! /usr/bin/env zsh
# Run in edgar
track_myjobs_edgar(){
    jobs=("${(@f)$(oarstat | grep melbayad | grep W=100)}")
    for job in $jobs; do 
        #jobn=${${job%,T=*}#*N=}
        jid="$(cut -d' ' -f1 <<< $job)"
        jobn=$(oarstat -j  $jid -f | grep 'stderr_file')
        jobn="$(cut -d' ' -f7 <<< $jobn)"
        jobn="$(cut -d'/' -f2 <<< $jobn)"
        echo 'Job:' $jobn '('$jid')'
        python scripts/show.py -f $jobn
    done
}

track_myjobs_lig(){
    jobs=("${(@f)$(myjobs | grep nmt)}")
    for job in $jobs; do 
        jobn="$(cut -d' ' -f3 <<< $job)"
        server="$(cut -d' ' -f1 <<< $job)"
        echo $jobn '('$server')'
        python scripts/show.py -f $jobn
    done
}

host=${HOST:r:r}$HOSTNAME
case $host in 
    edgar) track_myjobs_edgar;;
    decore*) track_myjobs_lig;;
    dvorak*) track_myjobs_lig;;
    hyperion*) track_myjobs_lig;;
    ceos*) track_myjobs_lig;;
    *)  echo "Unknown whereabouts!!" ;;
esac


