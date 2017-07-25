
tmux new -s app
tmux a -t app

source activate rllab3
cd ~/button-showdown
uwsgi --socket 0.0.0.0:8080 --protocol=http --callable app -w wsgi 
