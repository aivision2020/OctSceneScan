DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$DIR/:$DIR/cvil/slam:$DIR/mlildl:$DIR/pose_metrics
echo $PYTHONPATH
